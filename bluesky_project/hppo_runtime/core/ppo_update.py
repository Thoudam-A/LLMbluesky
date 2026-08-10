from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class PPOUpdateMetrics:
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    approx_kl_discrete: float = 0.0
    approx_kl_continuous: float = 0.0
    clip_fraction: float = 0.0
    clip_fraction_discrete: float = 0.0
    clip_fraction_continuous: float = 0.0
    explained_variance: float = 0.0
    discrete_entropy: float = 0.0
    continuous_entropy: float = 0.0
    discrete_prob_mean: float = 0.0
    continuous_logp_mean: float = 0.0
    candidate_entropy: float = 0.0
    approx_kl_candidate: float = 0.0
    clip_fraction_candidate: float = 0.0
    nan_detected: float = 0.0


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if var_y.item() < 1e-8:
        return 0.0
    return float((1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)).item())


def _finite_or_raise(tensors: list[torch.Tensor], fail_on_nonfinite: bool) -> bool:
    all_finite = all(torch.isfinite(tensor).all().item() for tensor in tensors)
    if (not all_finite) and fail_on_nonfinite:
        raise RuntimeError("Non-finite tensor detected during PPO update")
    return all_finite


def _ppo_update_discrete_parameters(model, optimizer_actor, optimizer_critic, batch, cfg) -> PPOUpdateMetrics:
    """PPO for the joint behavior/candidate and discrete-parameter action."""
    metrics = PPOUpdateMetrics()
    local_obs = batch["local_obs"]
    global_state = batch["global_state"]
    presence_mask = batch["presence_mask"]
    target_index = batch["target_index"]
    action_mask = batch["action_mask"]
    action_prior = batch.get("action_prior", torch.ones_like(action_mask))
    prior_weight = batch.get("prior_weight", torch.zeros(action_mask.shape[0], device=action_mask.device))
    macro_action = batch["macro_action"]
    branches = batch["parameter_branch"]
    parameter_indices = batch["parameter_action_index"]
    parameter_mask = batch["parameter_mask"]
    old_macro_logp = batch["discrete_logp"]
    old_parameter_logp = batch["parameter_logp"]
    candidate_mode = batch.get("candidate_mode", torch.zeros_like(old_macro_logp))
    candidate_features = batch.get("candidate_features")
    candidate_mask = batch.get("candidate_mask")
    candidate_index = batch.get("candidate_index", torch.full_like(macro_action, -1))
    old_candidate_logp = batch.get("candidate_logp", torch.zeros_like(old_macro_logp))
    advantages = batch["advantage"]
    returns = batch["return"]
    old_values = batch["value"]
    if cfg.normalize_advantage and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    with torch.no_grad():
        initial_values = model.critic(global_state, presence_mask, target_index).squeeze(-1)

    actor_losses = []
    critic_losses = []
    joint_kls = []
    behavior_kls = []
    parameter_kls = []
    clip_fracs = []
    discrete_entropies = []
    parameter_entropies = []
    candidate_entropies = []
    discrete_prob_means = []
    parameter_logp_means = []
    candidate_kls = []
    candidate_clip_fracs = []
    nan_detected = False
    early_stop = False

    for _ in range(cfg.epochs):
        indices = torch.randperm(local_obs.shape[0], device=local_obs.device)
        for start in range(0, local_obs.shape[0], cfg.mini_batch_size):
            mb_idx = indices[start : start + cfg.mini_batch_size]
            if mb_idx.numel() == 0:
                continue
            obs = local_obs[mb_idx]
            mask = action_mask[mb_idx].clone()
            invalid = mask.sum(dim=-1) <= 0.0
            if invalid.any():
                mask[invalid] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], device=mask.device)
            macro = macro_action[mb_idx]
            active_branch = branches[mb_idx]
            action_index = parameter_indices[mb_idx]
            param_mask = parameter_mask[mb_idx]
            candidate_active = candidate_mode[mb_idx] > 0.5

            macro_logits = model.discrete_actor(obs)
            guided_logits = macro_logits + prior_weight[mb_idx].unsqueeze(-1) * torch.log(action_prior[mb_idx].clamp_min(1e-8))
            macro_dist = torch.distributions.Categorical(logits=guided_logits.masked_fill(mask <= 0.0, -1e9))
            new_macro_logp = macro_dist.log_prob(macro)
            macro_entropy = macro_dist.entropy()
            macro_probs = macro_dist.probs

            if candidate_active.any():
                features = candidate_features[mb_idx]
                candidate_valid = candidate_mask[mb_idx]
                selector_prior = features[..., 16].clamp_min(1e-8)
                selector_logits = model.candidate_selector(obs, features) + cfg.candidate_prior_weight * torch.log(selector_prior)
                selector_dist = torch.distributions.Categorical(logits=selector_logits.masked_fill(candidate_valid <= 0.0, -1e9))
                new_candidate_logp = selector_dist.log_prob(candidate_index[mb_idx].clamp_min(0))
                selector_entropy = selector_dist.entropy()
            else:
                new_candidate_logp = torch.zeros_like(new_macro_logp)
                selector_entropy = torch.zeros_like(macro_entropy)

            parameter_logits = model.parameter_actor(obs)
            new_parameter_logp, parameter_entropy = model.parameter_policy.evaluate_parameters(
                parameter_logits, active_branch, action_index, param_mask
            )
            valid_parameter = (active_branch >= 0).float()
            new_behavior_logp = torch.where(candidate_active, new_candidate_logp, new_macro_logp)
            old_behavior_logp = torch.where(candidate_active, old_candidate_logp[mb_idx], old_macro_logp[mb_idx])
            new_joint_logp = new_behavior_logp + new_parameter_logp
            old_joint_logp = old_behavior_logp + old_parameter_logp[mb_idx]
            ratio = torch.exp(new_joint_logp - old_joint_logp)
            advantage = advantages[mb_idx]
            surrogate_1 = ratio * advantage
            surrogate_2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantage
            actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

            fallback_active = (~candidate_active).float()
            macro_entropy_mean = (macro_entropy * fallback_active).sum() / fallback_active.sum().clamp_min(1.0)
            selector_entropy_mean = (selector_entropy * candidate_active.float()).sum() / candidate_active.float().sum().clamp_min(1.0)
            parameter_entropy_mean = (parameter_entropy * valid_parameter).sum() / valid_parameter.sum().clamp_min(1.0)
            entropy_bonus = -(
                cfg.entropy_coef_discrete * (macro_entropy_mean + selector_entropy_mean)
                + cfg.entropy_coef_continuous * parameter_entropy_mean
            )
            values = model.critic(global_state[mb_idx], presence_mask[mb_idx], target_index[mb_idx]).squeeze(-1)
            value_clipped = old_values[mb_idx] + torch.clamp(values - old_values[mb_idx], -cfg.value_clip_ratio, cfg.value_clip_ratio)
            critic_loss = 0.5 * torch.max((values - returns[mb_idx]) ** 2, (value_clipped - returns[mb_idx]) ** 2).mean()
            loss = actor_loss + entropy_bonus + cfg.value_loss_coef * critic_loss
            finite_tensors = [macro_logits, *parameter_logits, new_joint_logp, values, actor_loss, critic_loss, loss]
            if not _finite_or_raise(finite_tensors, cfg.fail_on_nonfinite):
                nan_detected = True
                continue
            optimizer_actor.zero_grad(set_to_none=True)
            optimizer_critic.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor_parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.critic_parameters(), cfg.max_grad_norm)
            optimizer_actor.step()
            optimizer_critic.step()

            with torch.no_grad():
                joint_log_ratio = new_joint_logp - old_joint_logp
                behavior_log_ratio = new_behavior_logp - old_behavior_logp
                parameter_log_ratio = new_parameter_logp - old_parameter_logp[mb_idx]
                joint_kl = float(((torch.exp(joint_log_ratio) - 1.0) - joint_log_ratio).mean().item())
                behavior_kl = float(((torch.exp(behavior_log_ratio) - 1.0) - behavior_log_ratio).mean().item())
                parameter_kl = float(
                    ((((torch.exp(parameter_log_ratio) - 1.0) - parameter_log_ratio) * valid_parameter).sum()
                    / valid_parameter.sum().clamp_min(1.0)).item()
                )
                clip_fraction = float((torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean().item())
                if candidate_active.any():
                    candidate_log_ratio = new_candidate_logp - old_candidate_logp[mb_idx]
                    candidate_kl = float(
                        ((((torch.exp(candidate_log_ratio) - 1.0) - candidate_log_ratio) * candidate_active.float()).sum()
                        / candidate_active.float().sum()).item()
                    )
                    candidate_clip = float(
                        (((torch.abs(torch.exp(candidate_log_ratio) - 1.0) > cfg.clip_ratio).float() * candidate_active.float()).sum()
                        / candidate_active.float().sum()).item()
                    )
                else:
                    candidate_kl = 0.0
                    candidate_clip = 0.0
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                joint_kls.append(joint_kl)
                behavior_kls.append(behavior_kl)
                parameter_kls.append(parameter_kl)
                clip_fracs.append(clip_fraction)
                discrete_entropies.append(float(macro_entropy_mean.item()))
                parameter_entropies.append(float(parameter_entropy_mean.item()))
                candidate_entropies.append(float(selector_entropy_mean.item()))
                discrete_prob_means.append(float(macro_probs.max(dim=-1).values.mean().item()))
                parameter_logp_means.append(float(new_parameter_logp.mean().item()))
                candidate_kls.append(candidate_kl)
                candidate_clip_fracs.append(candidate_clip)
                if cfg.enable_kl_early_stop and cfg.target_kl_discrete > 0.0 and joint_kl > cfg.target_kl_discrete:
                    early_stop = True
                    break
        if early_stop:
            break

    metrics.actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
    metrics.critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
    metrics.entropy = float(np.mean(np.asarray(discrete_entropies) + np.asarray(parameter_entropies) + np.asarray(candidate_entropies))) if discrete_entropies else 0.0
    metrics.approx_kl = float(np.mean(joint_kls)) if joint_kls else 0.0
    metrics.approx_kl_discrete = float(np.mean(behavior_kls)) if behavior_kls else 0.0
    metrics.approx_kl_continuous = float(np.mean(parameter_kls)) if parameter_kls else 0.0
    metrics.clip_fraction = float(np.mean(clip_fracs)) if clip_fracs else 0.0
    metrics.clip_fraction_discrete = metrics.clip_fraction
    metrics.clip_fraction_continuous = metrics.clip_fraction
    metrics.explained_variance = explained_variance(initial_values, returns)
    metrics.discrete_entropy = float(np.mean(discrete_entropies)) if discrete_entropies else 0.0
    metrics.continuous_entropy = float(np.mean(parameter_entropies)) if parameter_entropies else 0.0
    metrics.discrete_prob_mean = float(np.mean(discrete_prob_means)) if discrete_prob_means else 0.0
    metrics.continuous_logp_mean = float(np.mean(parameter_logp_means)) if parameter_logp_means else 0.0
    metrics.candidate_entropy = float(np.mean(candidate_entropies)) if candidate_entropies else 0.0
    metrics.approx_kl_candidate = float(np.mean(candidate_kls)) if candidate_kls else 0.0
    metrics.clip_fraction_candidate = float(np.mean(candidate_clip_fracs)) if candidate_clip_fracs else 0.0
    metrics.nan_detected = 1.0 if nan_detected else 0.0
    return metrics


def ppo_update(model, optimizer_actor, optimizer_critic, batch: dict[str, torch.Tensor], cfg) -> PPOUpdateMetrics:
    if cfg.parameter_policy_type == "discrete":
        return _ppo_update_discrete_parameters(model, optimizer_actor, optimizer_critic, batch, cfg)
    metrics = PPOUpdateMetrics()
    local_obs = batch["local_obs"]
    global_state = batch["global_state"]
    presence_mask = batch["presence_mask"]
    target_index = batch["target_index"]
    action_mask = batch["action_mask"]
    action_prior = batch.get("action_prior", torch.ones_like(action_mask))
    prior_weight = batch.get("prior_weight", torch.zeros(action_mask.shape[0], device=action_mask.device))
    macro_action = batch["macro_action"]
    raw_params = batch["raw_params"]
    selected_param_index = batch["selected_param_index"]
    old_discrete_logp = batch["discrete_logp"]
    old_continuous_logp = batch["continuous_logp"]
    advantages = batch["advantage"]
    returns = batch["return"]
    old_values = batch["value"]
    candidate_mode = batch.get("candidate_mode", torch.zeros_like(old_values))
    candidate_features = batch.get("candidate_features")
    candidate_mask = batch.get("candidate_mask")
    candidate_index = batch.get("candidate_index", torch.full_like(macro_action, -1))
    old_candidate_logp = batch.get("candidate_logp", torch.zeros_like(old_values))

    if cfg.normalize_advantage and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    with torch.no_grad():
        initial_values = model.critic(global_state, presence_mask, target_index).squeeze(-1)
    batch_size = local_obs.shape[0]
    actor_losses = []
    critic_losses = []
    approx_kls = []
    approx_kls_discrete = []
    approx_kls_continuous = []
    clip_fracs = []
    clip_fracs_discrete = []
    clip_fracs_continuous = []
    entropies = []
    discrete_entropy_acc = []
    continuous_entropy_acc = []
    discrete_prob_means = []
    continuous_logp_means = []
    candidate_entropy_acc = []
    approx_kls_candidate = []
    clip_fracs_candidate = []
    nan_detected = False
    early_stop = False

    for _ in range(cfg.epochs):
        epoch_indices = torch.randperm(batch_size, device=local_obs.device)
        for start in range(0, batch_size, cfg.mini_batch_size):
            mb_idx = epoch_indices[start : start + cfg.mini_batch_size]
            if mb_idx.numel() == 0:
                continue
            mb_obs = local_obs[mb_idx]
            mb_global = global_state[mb_idx]
            mb_presence = presence_mask[mb_idx]
            mb_target = target_index[mb_idx]
            mb_mask = action_mask[mb_idx]
            mb_prior = action_prior[mb_idx].clamp_min(1e-8)
            mb_prior_weight = prior_weight[mb_idx]
            invalid_action_rows = mb_mask.sum(dim=-1) <= 0.0
            if invalid_action_rows.any():
                mb_mask = mb_mask.clone()
                mb_mask[invalid_action_rows] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], device=mb_mask.device)
            mb_macro = macro_action[mb_idx]
            mb_raw_params = raw_params[mb_idx]
            mb_selected_param_index = selected_param_index[mb_idx]
            mb_old_discrete_logp = old_discrete_logp[mb_idx]
            mb_old_continuous_logp = old_continuous_logp[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            mb_old_values = old_values[mb_idx]
            mb_candidate_mode = candidate_mode[mb_idx]
            mb_candidate_features = candidate_features[mb_idx] if candidate_features is not None else None
            mb_candidate_mask = candidate_mask[mb_idx] if candidate_mask is not None else None
            mb_candidate_index = candidate_index[mb_idx]
            mb_old_candidate_logp = old_candidate_logp[mb_idx]

            logits = model.discrete_actor(mb_obs)
            guided_logits = logits + mb_prior_weight.unsqueeze(-1) * torch.log(mb_prior)
            masked_logits = guided_logits.masked_fill(mb_mask <= 0.0, -1e9)
            dist_discrete = torch.distributions.Categorical(logits=masked_logits)
            new_discrete_logp = dist_discrete.log_prob(mb_macro)
            discrete_entropy = dist_discrete.entropy()
            probs = torch.softmax(masked_logits, dim=-1)

            candidate_active = mb_candidate_mode > 0.5
            if candidate_active.any():
                selector_prior = mb_candidate_features[..., 16].clamp_min(1e-8)
                selector_logits = model.candidate_selector(mb_obs, mb_candidate_features) + cfg.candidate_prior_weight * torch.log(selector_prior)
                selector_logits = selector_logits.masked_fill(mb_candidate_mask <= 0.0, -1e9)
                selector_dist = torch.distributions.Categorical(logits=selector_logits)
                new_candidate_logp = selector_dist.log_prob(mb_candidate_index.clamp_min(0))
                candidate_entropy = selector_dist.entropy()
            else:
                new_candidate_logp = torch.zeros_like(mb_adv)
                candidate_entropy = torch.zeros_like(mb_adv)

            mean, log_std = model.continuous_actor(mb_obs)
            branch_logps = []
            branch_entropies = []
            for branch in range(mean.shape[-1]):
                branch_logp = model.parameter_policy.branch_log_prob(
                    mean[:, branch : branch + 1],
                    log_std[:, branch : branch + 1],
                    mb_raw_params[:, branch : branch + 1],
                ).squeeze(-1)
                branch_logps.append(branch_logp)
                branch_entropies.append(-branch_logp)
            branch_logps = torch.stack(branch_logps, dim=-1)
            branch_entropies = torch.stack(branch_entropies, dim=-1)
            branch_mask = torch.zeros_like(branch_logps)
            valid_param = (mb_selected_param_index >= 0).float()
            valid_idx = mb_selected_param_index.clamp_min(0).unsqueeze(-1)
            branch_mask.scatter_(1, valid_idx, 1.0)
            branch_mask = branch_mask * valid_param.unsqueeze(-1)
            new_continuous_logp = (branch_logps * branch_mask).sum(dim=-1)
            new_continuous_entropy = (branch_entropies * branch_mask).sum(dim=-1)

            values = model.critic(mb_global, mb_presence, mb_target).squeeze(-1)
            ratio_discrete = torch.exp(new_discrete_logp - mb_old_discrete_logp)
            ratio_continuous = torch.exp(new_continuous_logp - mb_old_continuous_logp)
            ratio_candidate = torch.exp(new_candidate_logp - mb_old_candidate_logp)
            surrogate_discrete_1 = ratio_discrete * mb_adv
            surrogate_discrete_2 = torch.clamp(ratio_discrete, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
            surrogate_cont_1 = valid_param * (ratio_continuous * mb_adv)
            surrogate_cont_2 = valid_param * (torch.clamp(ratio_continuous, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv)
            fallback_active = (~candidate_active).float()
            if fallback_active.sum().item() > 0:
                discrete_loss = -(torch.min(surrogate_discrete_1, surrogate_discrete_2) * fallback_active).sum() / fallback_active.sum()
            else:
                discrete_loss = torch.zeros((), dtype=mb_adv.dtype, device=mb_adv.device)
            candidate_surrogate_1 = ratio_candidate * mb_adv
            candidate_surrogate_2 = torch.clamp(ratio_candidate, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_adv
            if candidate_active.any():
                candidate_loss = -(torch.min(candidate_surrogate_1, candidate_surrogate_2) * candidate_active).sum() / candidate_active.float().sum()
                candidate_entropy_mean = (candidate_entropy * candidate_active).sum() / candidate_active.float().sum()
            else:
                candidate_loss = torch.zeros((), dtype=mb_adv.dtype, device=mb_adv.device)
                candidate_entropy_mean = torch.zeros((), dtype=mb_adv.dtype, device=mb_adv.device)
            if valid_param.sum().item() > 0:
                continuous_loss = -(torch.min(surrogate_cont_1, surrogate_cont_2).sum() / valid_param.sum().clamp_min(1.0))
                continuous_entropy_mean = new_continuous_entropy.sum() / valid_param.sum().clamp_min(1.0)
            else:
                continuous_loss = torch.zeros((), dtype=mb_adv.dtype, device=mb_adv.device)
                continuous_entropy_mean = torch.zeros((), dtype=mb_adv.dtype, device=mb_adv.device)
            actor_loss = discrete_loss + candidate_loss + continuous_loss
            discrete_entropy_mean = (discrete_entropy * fallback_active).sum() / fallback_active.sum().clamp_min(1.0)
            entropy_bonus = -(cfg.entropy_coef_discrete * (discrete_entropy_mean + candidate_entropy_mean) + cfg.entropy_coef_continuous * continuous_entropy_mean)
            value_clipped = mb_old_values + torch.clamp(values - mb_old_values, -cfg.value_clip_ratio, cfg.value_clip_ratio)
            critic_loss = 0.5 * torch.max((values - mb_returns) ** 2, (value_clipped - mb_returns) ** 2).mean()
            loss = actor_loss + entropy_bonus + cfg.value_loss_coef * critic_loss

            finite = _finite_or_raise(
                [
                    logits,
                    mean,
                    log_std,
                    new_discrete_logp,
                    new_continuous_logp,
                    values,
                    actor_loss,
                    critic_loss,
                    loss,
                ],
                cfg.fail_on_nonfinite,
            )
            if not finite:
                nan_detected = True
                continue

            optimizer_actor.zero_grad(set_to_none=True)
            optimizer_critic.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor_parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.critic_parameters(), cfg.max_grad_norm)
            optimizer_actor.step()
            optimizer_critic.step()

            with torch.no_grad():
                discrete_log_ratio = new_discrete_logp - mb_old_discrete_logp
                approx_kl_discrete = float(
                    ((torch.exp(discrete_log_ratio) - 1.0) - discrete_log_ratio).mean().item()
                )
                if valid_param.sum().item() > 0:
                    continuous_log_ratio = new_continuous_logp - mb_old_continuous_logp
                    approx_kl_continuous = float(
                        ((((torch.exp(continuous_log_ratio) - 1.0) - continuous_log_ratio) * valid_param).sum()
                        .div(valid_param.sum())).item()
                    )
                    clip_fraction_continuous = float((((torch.abs(ratio_continuous - 1.0) > cfg.clip_ratio).float() * valid_param).sum().div(valid_param.sum())).item())
                else:
                    approx_kl_continuous = 0.0
                    clip_fraction_continuous = 0.0
                approx_kl = approx_kl_discrete + approx_kl_continuous
                if candidate_active.any():
                    candidate_log_ratio = new_candidate_logp - mb_old_candidate_logp
                    approx_kl_candidate = float(
                        ((((torch.exp(candidate_log_ratio) - 1.0) - candidate_log_ratio) * candidate_active).sum()
                        .div(candidate_active.float().sum())).item()
                    )
                    clip_fraction_candidate = float(
                        (((torch.abs(ratio_candidate - 1.0) > cfg.clip_ratio).float() * candidate_active).sum()
                        .div(candidate_active.float().sum())).item()
                    )
                else:
                    approx_kl_candidate = 0.0
                    clip_fraction_candidate = 0.0
                approx_kl += approx_kl_candidate
                clip_fraction_discrete = float(
                    (((torch.abs(ratio_discrete - 1.0) > cfg.clip_ratio).float() * fallback_active).sum()
                    .div(fallback_active.sum().clamp_min(1.0))).item()
                )
                components = 1.0 + float(valid_param.sum().item() > 0) + float(candidate_active.any().item())
                clip_fraction = (clip_fraction_discrete + clip_fraction_continuous + clip_fraction_candidate) / components
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                approx_kls.append(approx_kl)
                approx_kls_discrete.append(approx_kl_discrete)
                approx_kls_continuous.append(approx_kl_continuous)
                clip_fracs.append(clip_fraction)
                clip_fracs_discrete.append(clip_fraction_discrete)
                clip_fracs_continuous.append(clip_fraction_continuous)
                entropies.append(float((discrete_entropy_mean + candidate_entropy_mean + continuous_entropy_mean).item()))
                discrete_entropy_acc.append(float(discrete_entropy_mean.item()))
                continuous_entropy_acc.append(float(continuous_entropy_mean.item()))
                candidate_entropy_acc.append(float(candidate_entropy_mean.item()))
                approx_kls_candidate.append(approx_kl_candidate)
                clip_fracs_candidate.append(clip_fraction_candidate)
                discrete_prob_means.append(float(probs.max(dim=-1).values.mean().item()))
                continuous_logp_means.append(float(new_continuous_logp.mean().item()))

                if cfg.enable_kl_early_stop and (
                    (cfg.target_kl_discrete > 0.0 and approx_kl_discrete > cfg.target_kl_discrete)
                    or (cfg.target_kl_continuous > 0.0 and approx_kl_continuous > cfg.target_kl_continuous)
                ):
                    early_stop = True
                    break
        if early_stop:
            break

    metrics.actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
    metrics.critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
    metrics.entropy = float(np.mean(entropies)) if entropies else 0.0
    metrics.approx_kl = float(np.mean(approx_kls)) if approx_kls else 0.0
    metrics.approx_kl_discrete = float(np.mean(approx_kls_discrete)) if approx_kls_discrete else 0.0
    metrics.approx_kl_continuous = float(np.mean(approx_kls_continuous)) if approx_kls_continuous else 0.0
    metrics.clip_fraction = float(np.mean(clip_fracs)) if clip_fracs else 0.0
    metrics.clip_fraction_discrete = float(np.mean(clip_fracs_discrete)) if clip_fracs_discrete else 0.0
    metrics.clip_fraction_continuous = float(np.mean(clip_fracs_continuous)) if clip_fracs_continuous else 0.0
    metrics.explained_variance = explained_variance(initial_values, returns)
    metrics.discrete_entropy = float(np.mean(discrete_entropy_acc)) if discrete_entropy_acc else 0.0
    metrics.continuous_entropy = float(np.mean(continuous_entropy_acc)) if continuous_entropy_acc else 0.0
    metrics.discrete_prob_mean = float(np.mean(discrete_prob_means)) if discrete_prob_means else 0.0
    metrics.continuous_logp_mean = float(np.mean(continuous_logp_means)) if continuous_logp_means else 0.0
    metrics.candidate_entropy = float(np.mean(candidate_entropy_acc)) if candidate_entropy_acc else 0.0
    metrics.approx_kl_candidate = float(np.mean(approx_kls_candidate)) if approx_kls_candidate else 0.0
    metrics.clip_fraction_candidate = float(np.mean(clip_fracs_candidate)) if clip_fracs_candidate else 0.0
    metrics.nan_detected = 1.0 if nan_detected else 0.0
    return metrics
