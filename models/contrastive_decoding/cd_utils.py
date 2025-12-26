import torch
from torch import nn


def contrastive_decoding (processor, model_kwargs, next_token_logits, next_token_logits_cd, results, description, is_beam=False):
    cd_alpha = model_kwargs.get("cd_alpha") 
    cd_beta = model_kwargs.get("cd_beta") 

    # use_simple_diffを取得: model_kwargsから
    use_simple_diff = model_kwargs.get("use_simple_diff", False)
    
    if use_simple_diff:
        # p_v のエントロピーを計算
        probs = nn.functional.softmax(next_token_logits, dim=-1)  # [B, V]
        entropy = torch.distributions.Categorical(probs=probs).entropy()  # [B]

        # 最大エントロピーで正規化（0〜1）
        # vocab_size = probs.size(-1)
        # print(f"vocab_size: {vocab_size}")
        # max_entropy = torch.log(torch.tensor(vocab_size, dtype=probs.dtype,
                                            # device=probs.device))
        # h = (entropy / max_entropy).clamp(0.0, 1.0)  # [B]
        # print(f"h: {h}")
        print(f"entropy: {entropy}")
        h = torch.clamp(entropy, max=torch.tensor(1.0, device=entropy.device, dtype=entropy.dtype))
        # モデル側で設定するベース値（kwargs から取れるようにしてもOK）
        # alpha_min = model_kwargs.get("cd_alpha_min", 0.2 * cd_alpha)
        # alpha_max = model_kwargs.get("cd_alpha_max", 1.5 * cd_alpha)
        # beta_min  = model_kwargs.get("cd_beta_min",  0.5 * cd_beta)
        # beta_max  = model_kwargs.get("cd_beta_max",  0.95)  # β<=1 にしておく
        alpha_min = 0.0
        alpha_max = 10.0
        beta_min = 0.0
        beta_max = 10.0

        # 線形補間で α(h), β(h) を決める
        # h: [B] -> [B,1] にして logit とブロードキャスト
        h_unsq = h.unsqueeze(-1)  # [B, 1]

        # cd_alpha_adaptive = alpha_min + h_unsq * (alpha_max - alpha_min)  # [B,1]
        # cd_beta_adaptive  = beta_min  + h_unsq * (beta_max  - beta_min)   # [B,1]
        # cd_alpha_adaptive = alpha_max * h_unsq
        # cd_beta_adaptive = 1.0 - beta_max * h_unsq
        cd_alpha_adaptive = h_unsq
        cd_beta_adaptive = 1.0 - h_unsq
        # print(f"cd_alpha_adaptive: {cd_alpha_adaptive}, cd_beta_adaptive: {cd_beta_adaptive}")
        # logit 空間で cutoff を計算
        max_logit = next_token_logits.max(dim=-1, keepdim=True).values  # [B,1]
        cutoff = torch.log(cd_beta_adaptive) + max_logit                # [B,1]

        # CD ロジット
        cd_logits = next_token_logits + cd_alpha_adaptive * (next_token_logits - next_token_logits_cd)
        cd_logits = cd_logits.masked_fill(next_token_logits < cutoff, -float("inf"))

        kl_d = None


    else:
        # 既存のKL divergenceベースの実装
        p_v = nn.functional.softmax(next_token_logits, dim=-1)
        p_d = nn.functional.softmax(next_token_logits_cd, dim=-1)
        
        kl_d = 0.5 * ((torch.log2(torch.abs(p_v - p_d) ** cd_alpha + 1)) * (p_v + p_d)).sum(dim=-1).unsqueeze(-1)

        kld_alpha = 1 - kl_d 
        
        cutoff = kl_d * p_v.max(dim=-1, keepdim=True).values

        ##############################
        diffs = (1 + kld_alpha) * next_token_logits - kld_alpha * next_token_logits_cd
        cd_logits = diffs.masked_fill(p_v < cutoff, -float("inf"))

    next_token_logits = cd_logits
                    
    if processor is not None:
        final_probs = nn.functional.softmax(cd_logits, dim=-1)    
        next_img_probs = nn.functional.softmax(next_token_logits, dim=-1)
        next_desc_probs = nn.functional.softmax(next_token_logits_cd, dim=-1)
        
        final_tokens = torch.argmax(final_probs, dim=-1)
        from_img_tokens = torch.argmax(next_img_probs, dim=-1)
        from_desc_tokens = torch.argmax(next_desc_probs, dim=-1)
        
        img_token = processor.decode(from_img_tokens, skip_special_tokens=True)
        desc_token = processor.decode(from_desc_tokens, skip_special_tokens=True)
        final_token = processor.decode(final_tokens, skip_special_tokens=True)
        
        img_prob1 = next_img_probs[0, from_img_tokens].item()
        img_prob2 = next_img_probs[0, from_desc_tokens].item()
        img_prob3 = next_img_probs[0, final_tokens].item()
        
        desc_prob1 = next_desc_probs[0, from_img_tokens].item()
        desc_prob2 = next_desc_probs[0, from_desc_tokens].item()
        desc_prob3 = next_desc_probs[0, final_tokens].item()

        final_prob1 = final_probs[0, from_img_tokens].item()
        final_prob2 = final_probs[0, from_desc_tokens].item()
        final_prob3 = final_probs[0, final_tokens].item()
        
        kl_value = kl_d.item() if kl_d is not None else 0.0
        results.append({
            "kl_value": kl_value, "img_token": img_token, "img_prob1": img_prob1, "img_prob2": img_prob2, "img_prob3": img_prob3,
            "desc_token": desc_token,  "desc_prob1": desc_prob1, "desc_prob2": desc_prob2, "desc_prob3": desc_prob3,
            "final_token": final_token, "final_prob1": final_prob1, "final_prob2": final_prob2, "final_prob3": final_prob3,
        })
        description.append(final_token)
        # print(final_token)

    return cd_logits, results, description
