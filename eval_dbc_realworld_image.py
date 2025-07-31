import hydra
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import collections
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.dataset.robomimic_dataset import RobomimicImageDataset
from cleandiffuser.utils import report_parameters

from src.utils import set_seed, Logger


def inference(args, envs, dataset, agent):
    """Evaluate a trained agent and optionally save a video."""
    # ---------------- Start Rollout ----------------
    episode_steps = []

    solver = args.solver

    for i in range(args.eval_episodes // args.num_envs):
        ep_reward = [0.0] * args.num_envs
        obs, t = envs.reset(), 0

        while t < args.max_episode_steps:
            obs_dict = {}
            for k in obs.keys():
                obs_seq = obs[k].astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                nobs = dataset.normalizer['obs'][k].normalize(obs_seq)
                obs_dict[k] = nobs = torch.tensor(nobs, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)
            with torch.no_grad():
                condition = obs_dict
                if args.nn == "pearce_mlp":
                    # run sampling (num_envs, action_dim)
                    prior = torch.zeros((args.num_envs, args.action_dim), device=args.device)
                elif args.nn == 'dit':
                    # run sampling (num_envs, args.action_steps, action_dim)
                    prior = torch.zeros((args.num_envs, args.action_steps, args.action_dim), device=args.device)
                else:
                    raise ValueError("NN type not supported")

                if not args.diffusion_x:
                    naction, _ = agent.sample(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps, solver=solver,
                                        condition_cfg=condition, w_cfg=1.0, use_ema=True)
                else:
                    naction, _ = agent.sample_x(prior=prior, n_samples=args.num_envs, sample_steps=args.sample_steps, solver=solver,
                                        condition_cfg=condition, w_cfg=1.0, use_ema=True, extra_sample_steps=args.extra_sample_steps)

            # unnormalize prediction
            naction = naction.detach().to('cpu').clip(-1., 1.).numpy()  # (num_envs, action_dim)
            action_pred = dataset.normalizer['action'].unnormalize(naction)
            action = action_pred.reshape(args.num_envs, 1, args.action_dim)  # (num_envs, 1, action_dim)

            if args.abs_action:
                action = dataset.undo_transform_action(action)
            obs, reward, _, _ = envs.step(action)
            ep_reward += reward
            t += args.action_steps

        success = [1.0 if s > 0 else 0.0 for s in ep_reward]
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    return {'mean_step': np.nanmean(episode_steps), 'mean_reward': np.nanmean(episode_rewards), 'mean_success': np.nanmean(episode_success)}


@hydra.main(config_path="configs/dbc/robomimic_image", config_name="lift")
def pipeline(args):
    # --------------------- Create Path -----------------------
    set_seed(args.seed)

    save_path = f'{args.log_dir}/{args.pipeline_name}/{args.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Wandb Init -----------------------
    logger = Logger(pathlib.Path(save_path), args)

    # ---------------- Create Dataset --------------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = RobomimicImageDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
                                    n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
                                    pad_after=args.action_steps-1, abs_action=args.abs_action)
    print(dataset)

    # --------------- Create Diffusion Model -----------------
    if args.nn == "pearce_mlp":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import PearceMlp

        nn_diffusion = PearceMlp(act_dim=args.action_dim, To=args.obs_steps, emb_dim=256, hidden_dim=512).to(args.device)
        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop, use_group_norm=args.use_group_norm,
            use_seq=args.use_seq).to(args.device)
    elif args.nn == "dit":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import DiT1d

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=args.use_seq).to(args.device)
        nn_diffusion = DiT1d(
            args.action_dim, emb_dim=256*args.obs_steps, d_model=320, n_heads=10, depth=2, timestep_emb_type="fourier").to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    # ----------------- Diffusion Agent ----------------------
    if args.diffusion == "sde":
        from cleandiffuser.diffusion.sde import SDE
        # from src.contraction_diffusion import DiscreteDiffusionSDE as SDE
        args.diffusion_x = False  # SDE does not support diffusion_x
        agent = SDE(nn_diffusion, nn_condition, predict_noise=False,
                    optim_params={"lr": args.lr},
                    diffusion_steps=args.sample_steps,
                    device=args.device)
                    # eigen_weight=args.loss_weights.eigen_max,
                    # lambda_contr=args.lambda_contr,
                    # jacobian_weight=args.loss_weights.jacobian,
                    # loss_type=args.loss_type)
    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM
        agent = EDM(nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
                    optim_params={"lr": args.lr})
    else:
        raise NotImplementedError


    # ------------ Realworld Environment ---------------
    env = RealWorldEnvWrapper()

    # ----------------- Inference ----------------------
    if args.model_path:
        agent.load(args.model_path)
    else:
        raise ValueError(f'Empty model for inference at {args.model_path}')

    agent.model.eval()
    agent.model_ema.eval()

    metrics = {'step': args.gradient_steps}
    metrics.update(inference(args, env, dataset, agent))
    logger.log(metrics, category='inference')


if __name__ == "__main__":
    pipeline()
