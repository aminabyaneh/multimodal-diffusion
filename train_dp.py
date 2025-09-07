import os
import pdb
import time
import hydra
import pathlib

import warnings
warnings.filterwarnings('ignore')

import numpy as np

import torch

from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters

from src.utils import set_seed, Logger
from src.realworld_dataset import RealWorldImageDataset


@hydra.main(config_path="configs/dp", config_name="realworld_image_eef_pos")
def pipeline(args):
    # ---------------- Startup Setups ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.log_dir), args)

    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    full_dataset = RealWorldImageDataset(
        dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
        n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
        pad_after=args.action_steps-1, abs_action=args.abs_action)

    # Split into train/val
    val_ratio = getattr(args, "val_ratio", 0.1)
    total_len = len(full_dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    dataset = train_dataset
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
    )

    # --------------- Create Diffusion Model -----------------
    if args.nn == "chi_unet":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiUNet1d

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=256, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=args.use_seq).to(args.device)

        nn_diffusion = ChiUNet1d(
            args.action_dim, 256, args.obs_steps, model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
            obs_as_global_cond=True, timestep_emb_type="positional").to(args.device)

    elif args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiTransformer

        # embedding_dim = 256 # image embedding dimension
        # d_model = 256 # transformer model dimension
        # n_heads = 4 # number of attention heads
        # num_layers = 4 # number of transformer layers

        # More complex transformer
        embedding_dim = 1024 # image embedding dimension
        d_model = 256 # transformer model dimension
        n_heads = 8 # number of attention heads
        num_layers = 8 # number of transformer layers

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=embedding_dim, rgb_model_name=args.rgb_model, resize_shape=args.resize_shape,
            crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)

        nn_diffusion = ChiTransformer(
            args.action_dim, embedding_dim, args.horizon, args.obs_steps, d_model=d_model, nhead=n_heads, num_layers=num_layers,
            timestep_emb_type="positional").to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}, only 'dit' is supported for now.")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    if args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM
        agent = EDM(nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
                    optim_params={"lr": args.lr})

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)
    else:
        raise NotImplementedError


    # ----------------- Training ----------------------
    n_gradient_step = 0
    diffusion_loss_list = []
    start_time = time.time()

    # dataset loop
    for batch in loop_dataloader(dataloader):
        # get condition
        nobs = batch['obs']

        condition = {}
        for k in nobs.keys():
            condition[k] = nobs[k][:, :args.obs_steps, :].to(args.device)

        # get action
        naction = batch['action'].to(args.device)

        # update diffusion
        diffusion_loss = agent.update(naction, condition)['loss']
        lr_scheduler.step()
        diffusion_loss_list.append(diffusion_loss)

        if n_gradient_step % args.log_freq == 0:
            # validation
            diffusion_loss_val_list = []
            action_loss_val_list = []

            with torch.no_grad():
                # switch to eval mode
                agent.model.eval()
                agent.model_ema.eval()

                # loop over val dataset
                for val_batch in val_dataloader:
                    # get validation condition
                    val_condition = {}
                    val_obs = val_batch['obs']

                    for k in val_obs.keys():
                        obs_seq = val_obs[k].cpu().numpy().astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                        val_condition[k] = torch.tensor(obs_seq[:, :args.obs_steps, :], device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)

                    # get validation action
                    val_naction = val_batch['action'].to(args.device)

                    val_diffusion_loss = agent.loss(val_naction, val_condition).item()
                    diffusion_loss_val_list.append(val_diffusion_loss)

                    # now get actions too
                    val_condition_n = {}
                    for k in val_obs.keys():
                        obs_seq = val_obs[k].cpu().numpy().astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                        val_condition_n[k] = torch.tensor(obs_seq, device=args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)

                    batch_size = val_naction.shape[0]
                    prior = torch.zeros((batch_size, args.horizon, args.action_dim), device=args.device)

                    # sample from ema model
                    naction, _ = agent.sample(prior=prior, n_samples=batch_size, sample_steps=args.sample_steps, solver=args.solver,
                                                condition_cfg=val_condition_n, w_cfg=1.0, use_ema=True)

                    # unnormalize prediction
                    action_pred = naction.detach().to('cpu').clip(-1., 1.).numpy()  # (num_envs, action_dim)

                    # compute action mse
                    action_mse = np.mean((action_pred - val_naction.detach().to('cpu').numpy())**2)
                    action_loss_val_list.append(action_mse)

            val_diffusion_loss = float(np.mean(diffusion_loss_val_list))
            val_action_loss = float(np.mean(action_loss_val_list))

            metrics = {
                'step': n_gradient_step,
                'total_time': time.time() - start_time,
                'avg_diffusion_loss': float(np.mean(diffusion_loss_list)),
                'val_diffusion_loss': val_diffusion_loss,
                'val_action_loss': val_action_loss
            }
            logger.log(metrics, category='train')
            diffusion_loss_list = []

            # back to train mode
            agent.model.train()
            agent.model_ema.train()

        if n_gradient_step % args.save_freq == 0:
            logger.save_agent(agent=agent, identifier=n_gradient_step)

        n_gradient_step += 1
        if n_gradient_step > args.gradient_steps:
            # finish
            break

if __name__ == "__main__":
    pipeline()











