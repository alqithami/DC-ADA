"""
DC-Ada Methods and Baselines

This module implements:
1. SharedPolicy: Baseline using a single shared policy (no adaptation)
2. DC-Ada: Data-Centric Collaborative Adaptation using zeroth-order optimization
3. RandomPerturbation: Random parameter perturbation baseline
4. LocalFineTuning: Local gradient-based fine-tuning
5. ObservationNormalization: Simple observation normalization baseline

All methods use the same budget (total environment steps) for fair comparison.
"""

import torch
import numpy as np
import time
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Any

from .policy import SharedPolicy
from .transformation import TransformationLayer, RobotTransformationModule


class BaseMethod:
    """
    Base class for all adaptation methods.
    
    Provides common functionality for running episodes and logging results.
    """
    
    def __init__(
        self,
        env,
        obs_dim: int,
        action_dim: int = 2,
        total_budget: int = 50000,
        hidden_dim: int = 256,
        seed: int = 42,
        pretrained_path: str = None,
        # Common hyperparameters that appear across method blocks in YAML.
        # Keeping them here prevents "unexpected kwarg" failures and makes
        # the experiment runner robust to config changes.
        transform_init_std: float = 0.0,
        **unused_kwargs,
    ):
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.total_budget = total_budget
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.transform_init_std = float(transform_init_std)

        # Preserve unused kwargs for debugging / provenance.
        # We intentionally do not raise here to keep batch sweeps running.
        self._unused_kwargs = dict(unused_kwargs) if unused_kwargs else {}
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create shared policy
        self.policy = SharedPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            import os
            if os.path.exists(pretrained_path):
                # Checkpoint compatibility matters when obs_dim is fixed.
                # If a user supplies an incompatible checkpoint, fail fast
                # with a clear error.
                # Prefer safe weight-only loading when supported by the
                # installed PyTorch version.
                try:
                    state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
                except TypeError:
                    state = torch.load(pretrained_path, map_location='cpu')
                self.policy.load_state_dict(state)
                print(f"  Loaded pretrained policy from: {pretrained_path}")
            else:
                print(f"  Warning: Pretrained path not found: {pretrained_path}")
        
        # Set policy to eval mode (we're not training it)
        self.policy.eval()
        
        # Tracking
        self.steps_used = 0
        self.episode_count = 0
        self.results = []
        self.communication_bytes = 0
        
    def run_episode(
        self,
        policy: Optional[SharedPolicy] = None,
        transforms: Optional[List[TransformationLayer]] = None,
        deterministic: bool = False,
        max_steps_override: Optional[int] = None,
        reset_seed: Optional[int] = None,
        reset_rng_state: Optional[tuple] = None,
    ) -> Tuple[float, int, bool, Dict]:
        """
        Run a single episode and return results.
        
        Args:
            policy: Policy to use (defaults to self.policy)
            transforms: Optional transformation layers for each robot
            deterministic: Whether to use deterministic actions
            
        Returns:
            total_reward: Sum of rewards over episode
            steps: Number of steps taken
            success: Whether task was completed successfully
            info: Final info dict from environment
        """
        if policy is None:
            policy = self.policy

        # Common Random Numbers (CRN) support:
        # - If reset_seed is provided, we deterministically seed the env before
        #   reset(). This is the preferred mechanism.
        # - reset_rng_state is kept for backward compatibility.
        if reset_seed is not None:
            try:
                self.env.seed(int(reset_seed))
            except Exception:
                pass
        elif reset_rng_state is not None:
            try:
                self.env.rng.set_state(reset_rng_state)
            except Exception:
                pass

        obs_list, info = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Apply transformations if provided
            if transforms is not None:
                transformed_obs = []
                for i, obs in enumerate(obs_list):
                    obs_tensor = torch.from_numpy(obs).float()
                    with torch.no_grad():
                        trans_obs = transforms[i](obs_tensor)
                        trans_obs = torch.nan_to_num(trans_obs, nan=0.0, posinf=0.0, neginf=0.0)
                        trans_obs = torch.clamp(trans_obs, -10.0, 10.0)
                    transformed_obs.append(trans_obs.detach().cpu().numpy())
                obs_list = transformed_obs
            
            # Get actions from policy
            actions = policy.get_actions(obs_list, deterministic=deterministic)
            
            # Step environment
            obs_list, reward, done, info = self.env.step(actions)
            
            total_reward += reward
            steps += 1

            # IMPORTANT: Budget is counted in *environment* steps (joint steps),
            # not per-robot steps. This matches typical RL reporting and avoids
            # unintentionally shortening runs by a factor of num_robots.
            self.steps_used += 1

            # Optional truncation (used by DC-Ada candidate evaluations)
            if max_steps_override is not None and steps >= max_steps_override:
                break
            
        # Minimal communication model: a single scalar (mission reward) is
        # broadcast per rollout. We count this to avoid misleading "0 bytes"
        # reports.
        self.communication_bytes += 8  # float64

        return total_reward, steps, info.get('success', False), info
    
    def run(self) -> List[Dict]:
        """
        Run the method until budget is exhausted.
        
        Returns:
            List of episode result dictionaries
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def log_episode(
        self,
        reward: float,
        steps: int,
        success: bool,
        info: Dict,
        wall_time: float = 0.0
    ):
        """Log episode results."""
        self.episode_count += 1
        self.results.append({
            'episode': self.episode_count,
            'steps_used': self.steps_used,
            'reward': float(reward),
            'episode_length': int(steps),
            'success': bool(success),
            'wall_time': float(wall_time),
            'communication_bytes': self.communication_bytes,
            **{k: v for k, v in info.items() if isinstance(v, (int, float, bool))}
        })
    
    def get_results(self) -> List[Dict]:
        """Get all logged results."""
        return self.results


class SharedPolicyMethod(BaseMethod):
    """
    Baseline: Use shared policy without any adaptation.
    
    This represents the lower bound - what happens when we don't adapt
    to heterogeneous robot configurations at all.
    """
    
    def __init__(self, env, obs_dim: int, **kwargs):
        super().__init__(env, obs_dim, **kwargs)
        self.method_name = "shared_policy"
        
    def run(self) -> List[Dict]:
        """Run shared policy baseline."""
        episode_num = 0
        while self.steps_used < self.total_budget:
            start_time = time.time()
            episode_num += 1
            episode_seed = int(self.seed) + int(episode_num)
            reward, steps, success, info = self.run_episode(
                deterministic=True,
                reset_seed=episode_seed,
            )
            wall_time = time.time() - start_time
            self.log_episode(reward, steps, success, info, wall_time)
            
        return self.results


class DCADAMethod(BaseMethod):
    """
    DC-Ada: Data-Centric Collaborative Adaptation
    
    Uses zeroth-order optimization to adapt per-robot observation
    transformation layers while keeping the shared policy frozen.

    Implementation note
    -------------------
    Earlier iterations of this repo used an ES-style antithetic gradient
    estimator on transformation parameters. This version implements a
    more review-friendly accept/reject variant (Appendix-style): for each
    robot, we sample M perturbations, evaluate short rollouts under Common
    Random Numbers (CRN), and apply the best perturbation if it improves
    the baseline.
    """
    
    def __init__(
        self,
        env,
        obs_dim: int,
        num_candidates: int = 5,
        noise_scale: float = 0.05,
        latent_dim: int = 32,
        transform_init_std: float = 0.0,
        adaptation_interval: int = 5,
        step_size: float = 1.0,
        acceptance_margin: float = 0.0,
        candidate_rollout_fraction: float = 0.25,
        deterministic_rollouts: bool = True,
        **kwargs
    ):
        super().__init__(env, obs_dim, **kwargs)
        self.method_name = "dc_ada"
        self.num_candidates = num_candidates
        self.noise_scale = noise_scale
        self.latent_dim = latent_dim
        self.transform_init_std = float(transform_init_std)
        self.adaptation_interval = adaptation_interval
        self.step_size = step_size
        self.acceptance_margin = acceptance_margin
        self.candidate_rollout_fraction = candidate_rollout_fraction
        self.deterministic_rollouts = deterministic_rollouts
        
        # Create transformation layers for each robot
        self.transforms = [
            TransformationLayer(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                init_std=self.transform_init_std,
            )
            for _ in range(env.num_robots)
        ]
        
        # Track performance history for each robot
        self.reward_history = []
        self.adaptation_count = 0
        
    def run(self) -> List[Dict]:
        """Run DC-Ada adaptation."""
        episode_num = 0

        # Candidate rollouts are shorter than full episodes for efficiency.
        env_max_steps = int(getattr(self.env, 'max_steps', 500))
        cand_steps = max(1, int(env_max_steps * float(self.candidate_rollout_fraction)))

        while self.steps_used < self.total_budget:
            start_time = time.time()
            episode_num += 1

            # Deterministic per-episode seeding keeps runs reproducible and
            # makes candidate comparisons lower-variance.
            episode_seed = int(self.seed) + int(episode_num)

            # Run nominal episode with current transforms
            nominal_reward, steps, success, info = self.run_episode(
                transforms=self.transforms,
                deterministic=self.deterministic_rollouts,
                reset_seed=episode_seed,
            )

            self.reward_history.append(nominal_reward)

            # Periodically perform adaptation using short CRN rollouts
            if episode_num % self.adaptation_interval == 0:
                self._adapt_accept_reject(
                    episode_seed=episode_seed,
                    candidate_steps=cand_steps,
                )

            wall_time = time.time() - start_time

            # Add adaptation info to log
            info['adaptation_count'] = self.adaptation_count
            info['candidate_rollout_steps'] = int(cand_steps)
            self.log_episode(nominal_reward, steps, success, info, wall_time)

        return self.results

    def _adapt_accept_reject(self, episode_seed: int, candidate_steps: int):
        """One accept/reject adaptation round (zeroth-order, per robot).

        For each robot i, we:
          1) Evaluate a *baseline* short rollout with current transforms.
          2) Sample M perturbations ε_m ~ N(0, I) and evaluate candidates.
          3) Apply the best perturbation if it improves the baseline by
             at least `acceptance_margin`.

        All rollouts use the same `episode_seed` to implement CRN.
        """
        if self.steps_used >= self.total_budget:
            return

        self.adaptation_count += 1

        for robot_idx in range(self.env.num_robots):
            if self.steps_used >= self.total_budget:
                break

            # Baseline (short) rollout under CRN
            baseline_reward, _, _, _ = self.run_episode(
                transforms=self.transforms,
                deterministic=self.deterministic_rollouts,
                max_steps_override=candidate_steps,
                reset_seed=episode_seed,
            )

            current_params = self.transforms[robot_idx].get_params_vector().clone()
            best_reward = float(baseline_reward)
            best_eps: Optional[torch.Tensor] = None

            for _ in range(int(self.num_candidates)):
                if self.steps_used >= self.total_budget:
                    break

                eps = torch.randn_like(current_params)
                candidate_params = current_params + (self.noise_scale * eps)

                cand_transform = self.transforms[robot_idx].clone()
                cand_transform.set_params_vector(candidate_params)

                candidate_transforms = self.transforms.copy()
                candidate_transforms[robot_idx] = cand_transform

                cand_reward, _, _, _ = self.run_episode(
                    transforms=candidate_transforms,
                    deterministic=self.deterministic_rollouts,
                    max_steps_override=candidate_steps,
                    reset_seed=episode_seed,
                )

                if float(cand_reward) > best_reward:
                    best_reward = float(cand_reward)
                    best_eps = eps

            # Accept if improvement exceeds margin
            if best_eps is not None and (best_reward - float(baseline_reward)) > float(self.acceptance_margin):
                update = self.step_size * self.noise_scale * best_eps
                new_params = current_params + update
                self.transforms[robot_idx].set_params_vector(new_params)


class RandomPerturbationMethod(BaseMethod):
    """
    Baseline: Random parameter perturbation without selection.
    
    This helps isolate the contribution of the zeroth-order optimization
    in DC-Ada by showing what happens with random changes.
    """
    
    def __init__(
        self,
        env,
        obs_dim: int,
        noise_scale: float = 0.01,
        latent_dim: int = 32,
        transform_init_std: float = 0.0,
        perturbation_interval: int = 10,
        **kwargs
    ):
        super().__init__(env, obs_dim, **kwargs)
        self.method_name = "random_perturbation"
        self.noise_scale = noise_scale
        self.latent_dim = latent_dim
        self.transform_init_std = float(transform_init_std)
        self.perturbation_interval = perturbation_interval
        
        # Create transformation layers
        self.transforms = [
            TransformationLayer(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                init_std=self.transform_init_std,
            )
            for _ in range(env.num_robots)
        ]
        
    def run(self) -> List[Dict]:
        """Run random perturbation baseline."""
        episode_num = 0
        
        while self.steps_used < self.total_budget:
            start_time = time.time()
            episode_num += 1

            episode_seed = int(self.seed) + int(episode_num)
            
            # Randomly perturb transforms periodically
            if episode_num % self.perturbation_interval == 0:
                for i in range(len(self.transforms)):
                    self.transforms[i] = self.transforms[i].perturb(self.noise_scale)
            
            reward, steps, success, info = self.run_episode(
                transforms=self.transforms,
                deterministic=True,
                reset_seed=episode_seed,
            )
            wall_time = time.time() - start_time
            self.log_episode(reward, steps, success, info, wall_time)
            
        return self.results


class LocalFineTuningMethod(BaseMethod):
    """Baseline: Local gradient-based fine-tuning of transformation layers.

    This is intentionally *not* a placeholder. We fine-tune ONLY the per-robot
    observation transforms using a standard policy-gradient objective while
    keeping the shared policy frozen.

    Compared to DC-Ada:
      - DC-Ada: zeroth-order (gradient-free) transform updates
      - Local fine-tuning: first-order policy gradients through the frozen policy
        network into the transforms
    """

    def __init__(
        self,
        env,
        obs_dim: int,
        learning_rate: float = 3e-4,
        latent_dim: int = 32,
        finetune_interval: int = 1,
        finetune_steps: int = 1,
        training_rollout_fraction: float = 0.25,
        deterministic_eval: bool = True,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        grad_clip: float = 1.0,
        **kwargs,
    ):
        super().__init__(env, obs_dim, **kwargs)
        self.method_name = "local_finetuning"
        self.learning_rate = float(learning_rate)
        self.latent_dim = int(latent_dim)
        self.finetune_interval = int(finetune_interval)
        self.finetune_steps = int(finetune_steps)
        self.training_rollout_fraction = float(training_rollout_fraction)
        self.deterministic_eval = bool(deterministic_eval)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.grad_clip = float(grad_clip)

        # Freeze policy parameters (baseline fine-tunes transforms only).
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.policy.eval()

        # Per-robot transforms
        self.transforms = [
            TransformationLayer(
                obs_dim=obs_dim,
                latent_dim=self.latent_dim,
                init_std=self.transform_init_std,
            )
            for _ in range(env.num_robots)
        ]

        # One optimizer per robot (local updates)
        self.optimizers = [
            torch.optim.Adam(t.parameters(), lr=self.learning_rate)
            for t in self.transforms
        ]

    def run(self) -> List[Dict]:
        """Run local fine-tuning baseline."""
        episode_num = 0

        env_max_steps = int(getattr(self.env, 'max_steps', 500))
        train_steps = max(1, int(env_max_steps * self.training_rollout_fraction))

        while self.steps_used < self.total_budget:
            start_time = time.time()
            episode_num += 1
            episode_seed = int(self.seed) + int(episode_num)

            # 1) Training rollout (short) for gradient updates
            _, _, _, _, traj = self._rollout_collect(
                reset_seed=episode_seed,
                max_steps_override=train_steps,
                deterministic_actions=False,
            )

            if episode_num % self.finetune_interval == 0:
                for _ in range(self.finetune_steps):
                    self._update_transforms(traj)

            # 2) Evaluation rollout (full episode) for logging
            reward, steps, success, info = self.run_episode(
                transforms=self.transforms,
                deterministic=self.deterministic_eval,
                reset_seed=episode_seed,
            )

            wall_time = time.time() - start_time
            info['training_rollout_steps'] = int(train_steps)
            self.log_episode(reward, steps, success, info, wall_time)

        return self.results

    def _rollout_collect(
        self,
        reset_seed: int,
        max_steps_override: Optional[int] = None,
        deterministic_actions: bool = False,
    ) -> Tuple[float, int, bool, Dict, Dict]:
        """Roll out an episode and collect (obs, actions, rewards)."""
        try:
            self.env.seed(int(reset_seed))
        except Exception:
            pass

        obs_list, info = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        obs_traj: List[List[np.ndarray]] = []
        act_traj: List[List[np.ndarray]] = []
        rew_traj: List[float] = []

        while not done and self.steps_used < self.total_budget:
            # Save current raw observations
            obs_traj.append([o.copy() for o in obs_list])

            # Transform observations (no_grad for rollout speed)
            transformed_obs = []
            for i, obs in enumerate(obs_list):
                obs_t = torch.from_numpy(obs).float()
                with torch.no_grad():
                    trans_obs = self.transforms[i](obs_t)
                    trans_obs = torch.nan_to_num(trans_obs, nan=0.0, posinf=0.0, neginf=0.0)
                    trans_obs = torch.clamp(trans_obs, -10.0, 10.0)
                transformed_obs.append(trans_obs.detach().cpu().numpy())

            # Sample actions from the frozen policy
            actions = self.policy.get_actions(transformed_obs, deterministic=deterministic_actions)
            act_traj.append([a.copy() for a in actions])

            obs_list, reward, done, info = self.env.step(actions)

            total_reward += float(reward)
            rew_traj.append(float(reward))
            steps += 1
            self.steps_used += 1

            if max_steps_override is not None and steps >= int(max_steps_override):
                break

        # Minimal scalar reward broadcast
        self.communication_bytes += 8

        traj = {
            'obs': obs_traj,
            'actions': act_traj,
            'rewards': rew_traj,
        }
        return total_reward, steps, info.get('success', False), info, traj

    def _update_transforms(self, traj: Dict):
        """One REINFORCE-style update on transform parameters (recomputed graph).

        Numerical stability + efficiency
        -------------------------------
        Earlier versions used nested Python loops (time × robots) with a policy
        forward pass per (t, i), which is slow and can amplify numerical
        problems. This implementation:

          - Vectorizes policy evaluation across all (time, robot) pairs
          - Sanitizes non-finite observations/actions
          - Skips updates if the objective becomes non-finite

        The shared policy is frozen; gradients flow only into the per-robot
        transformation layers.
        """
        rewards: List[float] = traj.get('rewards', [])
        obs_traj: List[List[np.ndarray]] = traj.get('obs', [])
        act_traj: List[List[np.ndarray]] = traj.get('actions', [])

        T = len(rewards)
        if T <= 0:
            return

        # ------------------------------------------------------------------
        # Discounted returns (team reward)
        # ------------------------------------------------------------------
        returns: List[float] = []
        R = 0.0
        for r in reversed(rewards):
            r_f = float(r)
            if not np.isfinite(r_f):
                r_f = 0.0
            R = r_f + self.gamma * R
            returns.append(R)
        returns.reverse()

        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns for variance reduction (skip if degenerate)
        std = torch.std(returns_t)
        if torch.isfinite(std) and float(std) > 1e-6:
            returns_t = (returns_t - torch.mean(returns_t)) / (std + 1e-8)

        # ------------------------------------------------------------------
        # Build batched tensors: apply each robot's transform to its trajectory
        # ------------------------------------------------------------------
        trans_obs_list: List[torch.Tensor] = []
        act_list: List[torch.Tensor] = []

        for i in range(self.env.num_robots):
            obs_i_np = np.stack([obs_traj[t][i] for t in range(T)], axis=0).astype(np.float32)
            act_i_np = np.stack([act_traj[t][i] for t in range(T)], axis=0).astype(np.float32)

            # Sanitize
            if not np.all(np.isfinite(obs_i_np)):
                obs_i_np = np.nan_to_num(obs_i_np, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.all(np.isfinite(act_i_np)):
                act_i_np = np.nan_to_num(act_i_np, nan=0.0, posinf=0.0, neginf=0.0)

            obs_i = torch.from_numpy(obs_i_np)
            act_i = torch.from_numpy(act_i_np)

            trans_i = self.transforms[i](obs_i)
            trans_i = torch.nan_to_num(trans_i, nan=0.0, posinf=0.0, neginf=0.0)
            trans_i = torch.clamp(trans_i, -10.0, 10.0)

            trans_obs_list.append(trans_i)
            act_list.append(act_i)

        # Concatenate as (R*T, dim) where R=num_robots
        obs_all = torch.cat(trans_obs_list, dim=0)
        act_all = torch.cat(act_list, dim=0)

        # Policy eval (vectorized). Policy params are frozen but computation must
        # keep grad w.r.t. obs_all so transforms get gradients.
        log_prob, entropy, _ = self.policy.evaluate_actions(obs_all, act_all)

        if not torch.isfinite(log_prob).all() or not torch.isfinite(entropy).all():
            # Skip pathological update rather than injecting NaNs into parameters
            return

        # Reshape back to (R, T) with the same concatenation order
        Rn = int(self.env.num_robots)
        logp = log_prob.view(Rn, T)
        ent = entropy.view(Rn, T)

        logp_mean_t = torch.mean(logp, dim=0)  # (T,)
        ent_mean_t = torch.mean(ent, dim=0)    # (T,)

        # REINFORCE objective (maximize logp * return) => minimize negative
        loss = torch.mean((-logp_mean_t * returns_t) - (self.entropy_coef * ent_mean_t))

        if not torch.isfinite(loss):
            return

        # Zero grads (one optimizer per robot)
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        loss.backward()

        # Clip + step
        for transform in self.transforms:
            torch.nn.utils.clip_grad_norm_(transform.parameters(), self.grad_clip)

        for opt in self.optimizers:
            opt.step()

        # Post-step sanitation (prevents catastrophic NaNs from crashing sweeps)
        with torch.no_grad():
            for transform in self.transforms:
                for p in transform.parameters():
                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0)
                    p.data.clamp_(-5.0, 5.0)



class ObservationNormalizationMethod(BaseMethod):
    """
    Baseline: Simple observation normalization.
    
    Normalizes observations using running mean and standard deviation.
    This is a common technique that can help with heterogeneous observations.
    """
    
    def __init__(self, env, obs_dim: int, **kwargs):
        super().__init__(env, obs_dim, **kwargs)
        self.method_name = "obs_normalization"
        
        # Running statistics
        self.obs_mean = np.zeros(obs_dim)
        self.obs_var = np.ones(obs_dim)
        self.obs_count = 1e-4
        
    def run(self) -> List[Dict]:
        """Run observation normalization baseline."""
        episode_num = 0
        while self.steps_used < self.total_budget:
            start_time = time.time()

            episode_num += 1
            episode_seed = int(self.seed) + int(episode_num)
            try:
                self.env.seed(int(episode_seed))
            except Exception:
                pass

            obs_list, info = self.env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            
            while not done:
                # Update statistics and normalize
                normalized_obs = []
                for obs in obs_list:
                    self._update_stats(obs)
                    norm_obs = self._normalize(obs)
                    normalized_obs.append(norm_obs)
                
                # Get actions
                actions = self.policy.get_actions(normalized_obs, deterministic=True)
                
                # Step
                obs_list, reward, done, info = self.env.step(actions)
                total_reward += reward
                steps += 1
                # Budget is counted in environment (joint) steps.
                self.steps_used += 1
            
            wall_time = time.time() - start_time
            self.communication_bytes += 8  # scalar reward broadcast
            self.log_episode(total_reward, steps, info.get('success', False), info, wall_time)
            
        return self.results
    
    def _update_stats(self, obs: np.ndarray):
        """Update running mean and variance."""
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += delta * delta2
        
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        std = np.sqrt(self.obs_var / self.obs_count + 1e-8)
        return (obs - self.obs_mean) / std


# Factory function
def create_method(
    method_name: str,
    env,
    obs_dim: int,
    **kwargs
) -> BaseMethod:
    """
    Factory function to create methods by name.
    
    Args:
        method_name: Name of the method
        env: Environment instance
        obs_dim: Observation dimension
        **kwargs: Additional arguments for the method
        
    Returns:
        Method instance
    """
    methods = {
        'shared_policy': SharedPolicyMethod,
        'dc_ada': DCADAMethod,
        'random_perturbation': RandomPerturbationMethod,
        'local_finetuning': LocalFineTuningMethod,
        'obs_normalization': ObservationNormalizationMethod,
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")
    
    return methods[method_name](env, obs_dim, **kwargs)
