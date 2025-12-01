import torch
from torch import nn


def contrastive_decoding (processor, model_kwargs, next_token_logits, next_token_logits_cd, results, description, is_beam=False):
    cd_alpha = model_kwargs.get("cd_alpha") 
    cd_beta = model_kwargs.get("cd_beta") 

    # use_simple_diffを取得: model_kwargsから
    use_simple_diff = model_kwargs.get("use_simple_diff", False)
    
    if use_simple_diff:
        # # print("use_simple_diff is True")
        # # version 2: logit空間でcutoffを計算（vcd_sample.pyと統一）
        # cutoff = torch.log(torch.tensor(cd_beta, device=next_token_logits.device)) + next_token_logits.max(dim=-1, keepdim=True).values
        # cd_logits = next_token_logits + cd_alpha * (next_token_logits - next_token_logits_cd)
        # cd_logits = cd_logits.masked_fill(next_token_logits < cutoff, -float("inf"))
        # # cd_logits = cd_logits.masked_fill(probs < cutoff, -float("inf"))
        
        # エントロピーベースの適応的パラメータ調整（オプション）
        # エントロピーを計算
        probs = nn.functional.softmax(next_token_logits, dim=-1)
        entropy = torch.distributions.Categorical(probs=probs).entropy()
        # print(f"entropy: {entropy}")
        # エントロピーを正規化（最大エントロピー = log(vocab_size)で割る）
        # max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=torch.float32, device=entropy.device))
        # print(f"max_entropy: {max_entropy}")
        # normalized_entropy = entropy / max_entropy  # 0-1の範囲
        normalized_entropy = entropy
        # エントロピーが高ければα/βを大きく、低ければベース値(1.0)に近づける
        # entropy_factor = normalized_entropy.unsqueeze(-1)
        entropy_factor = normalized_entropy

        cd_alpha_adaptive = entropy_factor * cd_alpha
        cd_beta_adaptive = entropy_factor * cd_beta
        # print(f" {cd_alpha_adaptive}, {cd_beta_adaptive}")
        # version 2: logit空間でcutoffを計算（vcd_sample.pyと統一）
        if isinstance(cd_beta_adaptive, torch.Tensor):
            beta_tensor = cd_beta_adaptive
        else:
            beta_tensor = torch.tensor(cd_beta_adaptive, device=next_token_logits.device)
        cutoff = torch.log(beta_tensor) + next_token_logits.max(dim=-1, keepdim=True).values
        cd_logits = next_token_logits + cd_alpha_adaptive * (next_token_logits - next_token_logits_cd)
        cd_logits = cd_logits.masked_fill(next_token_logits < cutoff, -float("inf"))
        
        kl_d = None  # 単純差分モードではkl_dは計算しない

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
