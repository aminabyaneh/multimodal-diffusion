"""
eval_server.py

Starts a diffusion server which the client can query to get robot actions.
Adopted from: https://github.com/moojink/openvla-oft/blob/main/vla-scripts/deploy.py
"""

# ruff: noqa: E402
import json_numpy
json_numpy.patch()

import json
import logging
import traceback

import draccus
import torch
import uvicorn

import hydra
import os
import pathlib

import numpy as np

from dataclasses import dataclass
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.realworld_dataset import RealWorldImageDataset
from cleandiffuser.utils import report_parameters

from src.utils import set_seed, Logger


def get_policy(args):
   # ---------------- Create Logger ----------------
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.log_dir), args)

    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = RealWorldImageDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
                                    n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
                                    pad_after=args.action_steps-1, abs_action=args.abs_action)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

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

    # ----------------- Inference ----------------------
    if args.model_path:
        agent.load(args.model_path)
    else:
        raise ValueError(f'Empty model for inference at {args.model_path}')

    agent.model.eval()
    agent.model_ema.eval()

    print(f"Loaded model from {args.model_path}")
    return agent, logger, args, dataset


class PolicyServer:
    def __init__(self, cfg, args) -> pathlib.Path:
        """
        A simple server for Diffusion Policy models; exposes `/act` to predict an action for a given observation.
        """
        self.num_envs = 1 # Real-world environments are single-agent, so we only need one environment

        # deploy configuration
        self.cfg = cfg

        # load model, logger, and args
        self.agent, self.logger, self.args, self.dataset = get_policy(args)

    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            instruction = observation["instruction"] # no need for diffusion

            # set the diffusion solver
            solver = self.args.solver

            # normalize observation
            print(f"Observation keys: {observation.keys()}")
            obs = observation # ???

            obs_dict = {}
            for k in obs.keys():
                obs_seq = obs[k].astype(np.float32)  # (num_envs, obs_steps, obs_dim)
                nobs = self.dataset.normalizer['obs'][k].normalize(obs_seq)
                obs_dict[k] = nobs = torch.tensor(nobs, device=self.args.device, dtype=torch.float32)  # (num_envs, obs_steps, obs_dim)

            # form conditioning
            with torch.no_grad():
                condition = obs_dict
                if self.args.nn == "pearce_mlp":
                    # run sampling (num_envs, action_dim)
                    prior = torch.zeros((self.num_envs, self.args.action_dim), device=self.args.device)
                elif self.args.nn == 'dit':
                    # run sampling (num_envs, args.action_steps, action_dim)
                    prior = torch.zeros((self.num_envs, self.args.action_steps, self.args.action_dim), device=self.args.device)
                else:
                    raise ValueError("NN type not supported")

                if not self.args.diffusion_x:
                    naction, _ = self.agent.sample(prior=prior, n_samples=self.num_envs, sample_steps=self.args.sample_steps, solver=solver,
                                                   condition_cfg=condition, w_cfg=1.0, use_ema=True)
                else:
                    naction, _ = self.agent.sample_x(prior=prior, n_samples=self.num_envs, sample_steps=self.args.sample_steps, solver=solver,
                                                     condition_cfg=condition, w_cfg=1.0, use_ema=True, extra_sample_steps=self.args.extra_sample_steps)

            # unnormalize prediction
            naction = naction.detach().to('cpu').clip(-1., 1.).numpy()  # (num_envs, action_dim)
            action_pred = self.dataset.normalizer['action'].unnormalize(naction)
            action = action_pred.reshape(self.num_envs, 1, self.args.action_dim)  # (num_envs, 1, action_dim)

            if self.args.abs_action:
                action = self.dataset.undo_transform_action(action)

            print(f"Action shape: {action.shape}, Action: {action}")
            action = action

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)

        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # server configuration
    host: str = "10.122.196.156" # Host IP Address
    port: int = 8777 # Host Port

@draccus.wrap()
def deploy(cfg: DeployConfig, args) -> None:
    server = PolicyServer(cfg, args)
    server.run(cfg.host, port=cfg.port)

@hydra.main(config_path="configs/dbc", config_name="realworld_image")
def main(args):
    deploy(args)

if __name__ == "__main__":
    main()