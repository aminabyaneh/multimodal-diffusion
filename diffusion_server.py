"""
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

from src.utils import set_seed, crop_resize


def get_policy(args):
    """
    Initialize and return a diffusion policy model for real-world execution.

    Args:
        args: Configuration object containing the model attributes and checkpoint path.

    Returns:
        tuple: A tuple containing:
            - agent: Configured diffusion agent ready for inference
            - args: Updated configuration arguments
            - dataset: Initialized dataset object

    Raises:
        ValueError: If nn type is not 'dit' or 'chi_transformer' or if model_path is empty
        NotImplementedError: If diffusion type is not supported
    """
    # ---------------- Startup Setup ----------------
    set_seed(args.seed)

    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = RealWorldImageDataset(dataset_path, horizon=args.horizon, shape_meta=args.shape_meta,
                                    n_obs_steps=args.obs_steps, pad_before=args.obs_steps-1,
                                    pad_after=args.action_steps-1, abs_action=args.abs_action)

    # --------------- Create Diffusion Model -----------------
    # Do not use sequence model for real-world exectution
    args.use_seq = True

    if args.nn == "dit": # DBC with DiT backbone
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import DiT1d

        # Note: Dimensions fixed for now to match pretrained model
        embedding_dim = 256 # image embedding dimension
        d_model = 320 # transformer model dimension
        n_heads = 10 # number of attention heads
        depth = 2 # number of transformer blocks

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=embedding_dim, rgb_model_name=args.rgb_model, rgb_weights=args.rgb_weights,
            resize_shape=args.resize_shape, crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=args.use_seq).to(args.device)

        nn_diffusion = DiT1d(
            args.action_dim, emb_dim=embedding_dim*args.obs_steps,
            d_model=d_model, n_heads=n_heads, depth=depth,
            timestep_emb_type="fourier").to(args.device)

    elif args.nn == "chi_transformer": # DP with Chi_Transformer backbone
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiTransformer

        # Note: Dimensions fixed for now to match pretrained model
        embedding_dim = 1024 # image embedding dimension
        d_model = 256 # transformer model dimension
        n_heads = 8 # number of attention heads
        num_layers = 8 # number of transformer layers

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta, emb_dim=embedding_dim, rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape, crop_shape=args.crop_shape, random_crop=args.random_crop,
            use_group_norm=args.use_group_norm, use_seq=args.use_seq, keep_horizon_dims=True).to(args.device)

        nn_diffusion = ChiTransformer(
            args.action_dim, embedding_dim, args.horizon, args.obs_steps, d_model=d_model,
            nhead=n_heads, num_layers=num_layers, timestep_emb_type="positional").to(args.device)

    else:
        raise ValueError(f"Invalid nn type {args.nn}, only 'dit' and 'chi_transformer' is supported for real-world experiments.")

    # check model parameters
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")
    print(f"======================= Parameter Report of Condition Model =======================")
    report_parameters(nn_condition)
    print(f"===================================================================================")

    # ----------------- Diffusion Agent ----------------------
    if args.diffusion == "sde":
        from src.contraction_diffusion import DiscreteDiffusionSDE as SDE
        args.diffusion_x = False  # SDE does not support diffusion_x
        agent = SDE(nn_diffusion, nn_condition, predict_noise=False,
                    optim_params={"lr": args.lr},
                    diffusion_steps=args.sample_steps,
                    device=args.device,
                    eigen_weight=args.loss_weights.eigen_max,
                    lambda_contr=args.lambda_contr,
                    jacobian_weight=args.loss_weights.jacobian,
                    loss_type=args.loss_type)

    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM
        agent = EDM(nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=args.device,
                    optim_params={"lr": args.lr})

    else:
        raise NotImplementedError(f"Diffusion type {args.diffusion} not supported.")

    # ----------------- Inference ----------------------
    if args.model_path:
        agent.load(args.model_path)
    else:
        raise ValueError(f'Empty model for inference at {args.model_path}')

    agent.model.eval()
    agent.model_ema.eval()

    print(f"Loaded model from {args.model_path}")
    return agent, args, dataset


class PolicyServer:
    def __init__(self, cfg, args) -> pathlib.Path:
        """
        A simple server for Diffusion Policy models; exposes `/act` to predict an action for a given observation.
        """
        self.num_envs = 1 # real-world environments are single-agent, so we only need one environment

        # deploy configuration
        self.cfg = cfg

        # load model, and args
        self.agent, self.args, self.dataset = get_policy(args)

    def generate_action_from_observation(self, payload: Dict[str, Any]) -> str:
        """ Receives an observation from the client, and returns a predicted action.

        Args:
            payload: A dictionary with keys "observation" and "instruction". The observation is itself a dictionary
                     with keys "full_image", "depth_image", and "eef_pos". The values are numpy arrays or lists.
                     The instruction is a string.
        """
        try:
            if double_encode := "encoded" in payload:
                # support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # unpack the payload
            obs_list = payload["observations"]
            instruction = obs_list[0]["instruction"] # no need for diffusion, get from first obs
            print(f"Received {len(obs_list)} observations, instruction: {instruction}")

            # set the diffusion solver
            solver = self.args.solver

            # process multiple observations for sequence
            obs_dict = {}

            # initialize lists to collect observations across time steps
            agentview_obs = []
            depth_obs = []
            eef_pos_obs = []

            for i, obs in enumerate(obs_list):
                print(f"Processing observation {i+1}/{len(obs_list)}")

                # process agentview_rgb
                if "agentview_rgb" in obs and self.args.shape_meta.obs.get("agentview", False):
                    obs_seq = np.array(obs["agentview_rgb"]).astype(np.uint8)
                    obs_seq = crop_resize(obs_seq, is_depth=False, output_size=(224, 224))
                    agentview_obs.append(obs_seq)

                # process agentview_depth
                if "agentview_depth" in obs and self.args.shape_meta.obs.get("depth", False):
                    obs_seq = np.array(obs["agentview_depth"]).astype(np.float32)
                    obs_seq = crop_resize(obs_seq, is_depth=True, output_size=(224, 224))
                    depth_obs.append(obs_seq)

                # process eef_pos
                if "eef_pos" in obs and self.args.shape_meta.obs.get("ee_pos", False):
                    obs_seq = np.array(obs["eef_pos"]).astype(np.float32)
                    eef_pos_obs.append(obs_seq)

            # stack observations and normalize
            if agentview_obs:
                agentview_stacked = np.stack(agentview_obs, axis=0)  # (obs_steps, H, W, C)
                print(f"[agentview] Stacked shape: {agentview_stacked.shape}")

                nobs = self.dataset.normalizer['obs']['agentview'].normalize(agentview_stacked)
                nobs = nobs.transpose(0, 3, 1, 2)  # (obs_steps, C, H, W)
                nobs = np.expand_dims(nobs, 0)  # (1, obs_steps, C, H, W)
                obs_dict['agentview'] = torch.tensor(nobs, device=self.args.device, dtype=torch.float32)
                print(f"[agentview] Final tensor shape: {obs_dict['agentview'].shape}")

            if depth_obs:
                depth_stacked = np.stack(depth_obs, axis=0)  # (obs_steps, H, W, C)
                print(f"[depth] Stacked shape: {depth_stacked.shape}")

                nobs = self.dataset.normalizer['obs']['depth'].normalize(depth_stacked)
                nobs = nobs.transpose(0, 3, 1, 2)  # (obs_steps, C, H, W)
                nobs = np.repeat(nobs, 3, axis=1)  # Expand to 3 channels
                nobs = np.expand_dims(nobs, 0)  # (1, obs_steps, C, H, W)
                obs_dict['depth'] = torch.tensor(nobs, device=self.args.device, dtype=torch.float32)
                print(f"[depth] Final tensor shape: {obs_dict['depth'].shape}")

            if eef_pos_obs:
                eef_pos_stacked = np.stack(eef_pos_obs, axis=0)  # (obs_steps, 3)
                print(f"[ee_pos] Stacked shape: {eef_pos_stacked.shape}")

                nobs = self.dataset.normalizer['obs']['ee_pos'].normalize(eef_pos_stacked)
                nobs = np.expand_dims(nobs, 0)  # (1, obs_steps, 3)
                obs_dict['ee_pos'] = torch.tensor(nobs, device=self.args.device, dtype=torch.float32)
                print(f"[ee_pos] Final tensor shape: {obs_dict['ee_pos'].shape}")

            # form conditioning
            with torch.no_grad():
                condition = obs_dict

                # only 1 action for real-world
                prior = torch.zeros((1, self.args.action_steps, self.args.action_dim), device=self.args.device)
                nactions, _ = self.agent.sample(prior=prior, n_samples=1, sample_steps=self.args.sample_steps, solver=solver,
                                                condition_cfg=condition, w_cfg=1.0, use_ema=True)

            # unnormalize prediction
            nactions = nactions.detach().cpu().clip(-1., 1.).numpy()
            actions_pred = self.dataset.normalizer['action'].unnormalize(nactions)
            actions = actions_pred.reshape(self.num_envs, 1, self.args.action_dim)

            if self.args.abs_action:
                actions = self.dataset.undo_transform_action(actions)

            print(f"Sending actions with the shape: {actions.shape}")

            if double_encode:
                return JSONResponse(json_numpy.dumps(actions))
            else:
                return JSONResponse(actions)

        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str, port: int) -> None:
        """ Runs the FastAPI server.

        Args:
            host: Host IP address to run the server on.
            port: Host port to run the server on.
        """
        self.app = FastAPI()
        self.app.post("/act")(self.generate_action_from_observation)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    """ Configuration for the diffusion policy server.

    NOTE: You can override these parameters from the command line based on the local network.

    Attributes:
        host: Host IP address to run the server on. Default is "10.69.55.168".
        port: Host port to run the server on. Default is 8777.
    """

    host: str = "10.69.55.168"
    port: int = 8777


@draccus.wrap()
def deploy(cfg: DeployConfig, args) -> None:
    """ Deploys a diffusion policy server. Call with `python eval_server.py --config-path <path-to-config>`.

    Args:
        cfg: Configuration for the server (host, port).
        args: Command line arguments. See `configs/dbc/realworld_image_eef_pos.yaml` for an example.
    """
    server = PolicyServer(cfg, args)
    server.run(cfg.host, port=cfg.port)


@hydra.main(config_path="configs/dbc", config_name="realworld_image_eef_pos")
def main(args):
    """ Main function. Just a wrapper around `deploy` for configs.

    Args:
        args: Command line arguments. See `configs/dbc/realworld_image_eef_pos.yaml` for an example.
    """
    deploy(args)

# main entry point
if __name__ == "__main__":
    main()