import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput

from transformers.generation.utils import (
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput
)

from tools.prob_vis import log_token_distribution

print("VCD multinomial")

def _normalized_wasserstein_distance(probs_a: torch.Tensor, probs_b: torch.Tensor) -> torch.Tensor:
    """カテゴリ分布間の正規化された1次元Wasserstein距離を計算する。"""
    # 1次元に限定することで密な輸送行列を作る必要がなくなる。
    # 累積和を使えば離散かつ等間隔の支持集合におけるEarth-Mover距離を厳密に求められる。
    cdf_diff = torch.cumsum(probs_a - probs_b, dim=-1)
    print(cdf_diff[0,0:20])
    print("-------------")
    emd = torch.abs(cdf_diff).sum(dim=-1)
    # print(emd.shape)
    print("-------------")
    print(emd)
    vocab_size = probs_a.size(-1)
    max_distance = max(vocab_size - 1, 1)
    denom = torch.tensor(float(max_distance), device=probs_a.device, dtype=probs_a.dtype)
    normalized = torch.clamp(emd / denom, min=0.0, max=1.0)
    # print(normalized)
    return normalized


def _embedding_sinkhorn_distance(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    embeddings: torch.nn.Embedding,
    *,
    topk: int = 32064,
    lamda: float = 5,
    iters: int = 20,
) -> torch.Tensor:
    weight = embeddings.weight.to(device=probs_a.device, dtype=probs_a.dtype)
    k = min(topk, weight.size(0))
    distances = []

    for a_row, b_row in zip(probs_a, probs_b):
        # 1) 両分布それぞれのTop-Kを取り、和集合をサポートとする
        idx_a = torch.topk(a_row, k).indices
        idx_b = torch.topk(b_row, k).indices
        support_idx = torch.unique(torch.cat([idx_a, idx_b], dim=0)).sort().values

        # 2) サポート上に確率を再正規化（マスを残した部分だけで分布を作る）
        a = a_row[support_idx].clamp_min(0)
        b = b_row[support_idx].clamp_min(0)
        # tv = 0.5 * (a - b).abs().sum()
        # print("TV:", tv.item())
        a = a / a.sum().clamp_min(1e-8)
        b = b / b.sum().clamp_min(1e-8)

        # 3) サポートの埋め込みからL2距離コスト行列を構築
        emb = weight[support_idx]
        # emb = F.normalize(emb, dim=-1)
        # cost = (1.0 - emb @ emb.t()).clamp_min(0.0)  # cosine cost in [0,2]
        cost = torch.cdist(emb, emb, p=2)
        # print(f"mean:{cost.mean()}, max:{cost.max()}, size:{cost.shape}")

        # 4) Sinkhorn用にカーネル行列Kを作成（εでスムーズ化）
        K = torch.exp(-cost * lamda)
        # print(f"{K[0,:10]}")

        # 5) 反復Bregman投影でカップリングを近似（u,vはスケーリング係数）
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(iters):
            u = a / (K @ v)
            v = b / (K.t() @ u)
            # print(f"u:{u[:3]}, v:{v[:3]}")
        # print("------------")

        # 6) 得られた輸送計画transportとコストで期待コストを計算し、平均コストで正規化
        transport_plan = torch.outer(u, v) * K
        dist = (transport_plan * cost).sum()
        # print(f"mean:{transport_plan.mean()}, max:{transport_plan.max()}")
        # distances.append(torch.clamp(dist / cost.mean(), min=0.0, max=1.0))
        distances.append(dist)
        # print(distances)
    # print(distances)
    return torch.stack(distances)

def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id


    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    is_start=True
    step_count = 0
    # Keep a separate cache/state for the contrastive (noisy image) forward pass.
    # If we rebuild model_kwargs_cd from model_kwargs every step, it will reuse the
    # original-image past_key_values and become identical after step 0.
    model_kwargs_cd = model_kwargs.copy()

    # auto-regressive generation
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        ## For contrastive decoding initial
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )

        ## cd_comments: forward pass of the model with distorted image input
        model_inputs_cd = self.prepare_inputs_for_generation(input_ids, **model_kwargs_cd)

        if 'image_cd' in model_kwargs_cd:
            if 'pixel_values' in model_inputs_cd:
                model_inputs_cd['pixel_values'] = model_kwargs_cd["image_cd"]
                # print("pi") #全部こっちを通過すればいい
            elif 'images' in model_inputs_cd:
                model_inputs_cd['images'] = model_kwargs_cd["image_cd"]
                print("im")
            else:
                print("else")

        elif is_start and ('cd_input_embed' in model_kwargs_cd):
            model_inputs_cd['inputs_embeds'] = model_kwargs_cd['cd_input_embed']
            is_start=False
        
        # if ('image_cd' not in model_kwargs) == ('cd_input_embed' not in model_kwargs):
        #     raise ValueError ("Only one of image_cd or cd_input_embed must be given !!!")
        
        outputs_cd = self(
            **model_inputs_cd,
            vcd_decoding=True,
            return_dict=True,
            output_attentions=output_attentions_wo_img,
            output_hidden_states=output_hidden_states_wo_img,
        )
        next_token_logits_cd = outputs_cd.logits[:, -1, :]
        
        ## cd_comments: pre-process logits from contrastive inputs
        # version 1  set cutoff for Adaptive Plausibility Constraints
        # probs = nn.functional.softmax(next_token_logits, dim=-1)
        # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values
        
        # version 2 set cutoff for Adaptive Plausibility Constraints
        # cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
        
        # diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
        # cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
        cd_alpha = model_kwargs.get("cd_alpha", 1.0)
        cd_beta = model_kwargs.get("cd_beta", 0.1)
        # use_simple_diff = getattr(self, "_use_simple_diff", False)
        # if not use_simple_diff:
        #     use_simple_diff = model_kwargs.get("use_simple_diff", False)
        
        ######################################################################### KL
        # probs = nn.functional.softmax(next_token_logits, dim=-1)
        # probs_cd = nn.functional.softmax(next_token_logits_cd, dim=-1)

        # # KL divergence D_KL(p || q)
        # # 数値安定化のため eps を入れる
        # eps = 1e-8
        # kl = (probs * (torch.log(probs + eps) - torch.log(probs_cd + eps))).sum(dim=-1)
        # print(kl)
        # h_unsq = kl.unsqueeze(-1)

        # cd_alpha_adaptive = h_unsq
        # # Encourage exploration when distributions agree
        # cd_beta_adaptive = (1.0 - cd_alpha_adaptive).clamp(min=1e-6)

        # max_logit = next_token_logits.max(dim=-1, keepdim=True).values
        # cutoff = torch.log(cd_beta_adaptive) + max_logit

        # cd_logits = next_token_logits + cd_alpha_adaptive * (next_token_logits - next_token_logits_cd)
        # cutoff_cd_logits = cd_logits.masked_fill(next_token_logits < cutoff, -float("inf"))

        
        ######################################################################### Proposed
        # if use_simple_diff:
        probs = nn.functional.softmax(next_token_logits, dim=-1)
        probs_cd = nn.functional.softmax(next_token_logits_cd, dim=-1)
        # Use a Wasserstein distance to gauge distribution mismatch instead of entropy only.
        # wasserstein = _normalized_wasserstein_distance(probs, probs_cd)
        # Try an embedding-aware OT distance (Top-K support) if embeddings are available.
        ot_topk = model_kwargs.get("ot_topk", 3206)

        # print("max_abs_diff:", (probs - probs_cd).abs().max().item())
        # print("tv_full:", 0.5*(probs - probs_cd).abs().sum(-1).mean().item())

        # print(f"p:{probs}")
        # print(f"q:{probs_cd}")
        emb_module = self.get_input_embeddings()
        if emb_module is not None and ot_topk > 0:
            wasserstein = _embedding_sinkhorn_distance(
                probs, probs_cd, emb_module, topk=ot_topk
            )

        h_unsq = wasserstein.unsqueeze(-1)

        cd_alpha_adaptive = h_unsq
        # Encourage exploration when distributions agree (
        cd_beta_adaptive = (1.0 - cd_alpha_adaptive).clamp(min=1e-6)
        # cd_beta_adaptive = cd_alpha_adaptive

        max_logit = next_token_logits.max(dim=-1, keepdim=True).values
        cutoff = torch.log(cd_beta_adaptive) + max_logit

        cd_logits = next_token_logits + cd_alpha_adaptive * (next_token_logits - next_token_logits_cd)
        cutoff_cd_logits = cd_logits.masked_fill(next_token_logits < cutoff, -float("inf"))

        ######################################################################### default
        # # else: #デフォ
        # cutoff_base = torch.log(
        #     torch.tensor(cd_beta, device=next_token_logits.device, dtype=next_token_logits.dtype)
        # ) + next_token_logits.max(dim=-1, keepdim=True).values
        
        # diffs = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
        # cd_logits = diffs.masked_fill(next_token_logits < cutoff_base, -float("inf"))
        # cutoff_cd_logits = cd_logits

        #########################################################################

        ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
        cutoff_cd_logits = logits_processor(input_ids, cutoff_cd_logits)
        cutoff_cd_logits = logits_warper(input_ids, cutoff_cd_logits)

        next_token_scores = cutoff_cd_logits
        cd_probs = nn.functional.softmax(cutoff_cd_logits, dim=-1)
        next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)

        next_tokens = next_tokens.to(device=input_ids.device)

        #########################################################################

        # # top5確率分布を保存
        # tokenizer = getattr(self, "cd_tokenizer", None)
        # meta = {
        #     "mode": "vcd_sample",
        #     "logits": "next_token_logits",
        #     "contrastive": "True",
        # }
        # log_token_distribution(
        #     next_token_logits,
        #     tokenizer,
        #     step=step_count,
        #     meta=meta,
        # )
        # step_count += 1
        #########################################################################


        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        ## cd_comments: update model_kwargs_cd for contrastive decoding
        model_kwargs_cd = self._update_model_kwargs_for_generation(
            outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin._sample = sample
