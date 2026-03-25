from typing import Dict, Tuple, Union, Callable
import copy
import torch
import torch.nn as nn
import torchvision

from cleandiffuser.utils.crop_randomizer import CropRandomizer
from cleandiffuser.nn_condition import BaseNNCondition

from source.utils import replace_submodules, get_vit_backbone


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.

    Modulates a feature tensor x using scale (gamma) and shift (beta)
    parameters predicted from a conditioning input.
    """
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, feature_dim * 2)

    def forward(self, x, cond):
        """
        Args:
            x: Feature tensor (B, D) to be modulated.
            cond: Conditioning tensor (B, cond_dim).
        Returns:
            Modulated feature tensor (B, D).
        """
        gamma_beta = self.fc(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1 + gamma) * x + beta


class MultiImageObsConditionFilm(BaseNNCondition):
    """
    Multi-modal observation condition encoder using FiLM conditioning.

    Uses the primary (RGB) vision features as the main representation and
    modulates them with tactile features via Feature-wise Linear Modulation (FiLM).
    Low-dim features are concatenated after modulation.

    Args:
        condition: Dictionary of observations with keys mapping to tensors
                  - RGB/tactile images: (B, C, H, W) or (B, seq_len, C, H, W)
                  - Low-dim features: (B, D) or (B, seq_len, D)
        mask: Optional mask tensor (B, *mask_shape) or None

    Returns:
        condition: Encoded feature tensor (B, *cond_out_shape)
    """
    def __init__(self,
            shape_meta: dict,
            rgb_model_name: str, # resnet18, resnet34, resnet50, vit_large_patch14_reg4_dinov2, vit_small_patch14_reg4_dinov2
            tactile_model_name: str = None, # resnet18, vit_large_patch14_reg4_dinov2
            emb_dim: int = 256,
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            # use_seq: B, seq_len, C, H, W or B, C, H, W
            use_seq=False,
            # if True: (bs, seq_len, embed_dim)
            keep_horizon_dims=False
        ):
        super().__init__()
        rgb_keys = list()
        tactile_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # backbones
        rgb_model = get_vit_backbone(rgb_model_name)
        if tactile_model_name is not None:
            tactile_model = get_vit_backbone(tactile_model_name)
        else:
            tactile_model = rgb_model  # use same architecture as vision for tactile

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'image_rgb' or type == 'image_tactile':
                if type == 'image_rgb':
                    rgb_keys.append(key)
                    base_model = rgb_model
                else:
                    tactile_keys.append(key)
                    base_model = tactile_model

                # configure model for this key
                this_model = None
                if isinstance(base_model, dict):
                    this_model = base_model[key]
                else:
                    assert isinstance(base_model, nn.Module)
                    this_model = copy.deepcopy(base_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16,
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h, w)
                    )
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        tactile_keys = sorted(tactile_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.rgb_keys = rgb_keys
        self.tactile_keys = tactile_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims

        # compute feature dimensions via a dummy forward pass
        rgb_feat_dim, tactile_feat_dim, low_dim_total = self._compute_feature_dims()

        # FiLM layers: one per RGB key, conditioned on concatenated tactile features
        self.film_layers = nn.ModuleDict()
        if tactile_feat_dim > 0:
            for key in self.rgb_keys:
                self.film_layers[key] = FiLMLayer(cond_dim=tactile_feat_dim, feature_dim=rgb_feat_dim // len(self.rgb_keys))

        # final MLP: modulated RGB features + low_dim -> emb_dim
        mlp_input_dim = rgb_feat_dim + low_dim_total
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

    def _encode_image(self, key, img):
        """Encode a single image through its transform and model."""
        img = self.key_transform_map[key](img)
        # check for input mismatch and resize if needed
        default_cfg = getattr(self.key_model_map[key], "default_cfg", None)
        if default_cfg is not None and (input_size := default_cfg.get("input_size")) is not None:
            if tuple(input_size) != tuple(img.shape[1:]):
                img = torchvision.transforms.Resize(size=input_size[1:])(img)
        return self.key_model_map[key](img)

    @torch.no_grad()
    def _compute_feature_dims(self):
        """Compute feature dimensions for RGB, tactile, and low-dim via dummy forward."""
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        rgb_feat_dim = 0
        tactile_feat_dim = 0
        low_dim_total = 0

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'low_dim':
                low_dim_total += shape[0] if len(shape) == 1 else int(torch.tensor(shape).prod().item())
            elif type in ('image_rgb', 'image_tactile'):
                dummy = torch.zeros((batch_size,) + shape, dtype=self.dtype, device=self.device)
                feat = self._encode_image(key, dummy)
                feat_d = feat.shape[-1]
                if type == 'image_rgb':
                    rgb_feat_dim += feat_d
                else:
                    tactile_feat_dim += feat_d

        return rgb_feat_dim, tactile_feat_dim, low_dim_total

    def multi_image_forward(self, obs_dict):
        batch_size = None
        rgb_features = list()
        tactile_features = list()
        low_dim_features = list()

        if self.use_seq:
            for k in obs_dict.keys():
                obs_dict[k] = obs_dict[k].flatten(end_dim=1)

        # extract RGB features
        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == self.key_shape_map[key]
            feature = self._encode_image(key, img)
            rgb_features.append((key, feature))

        # extract tactile features
        for key in self.tactile_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == self.key_shape_map[key]
            feature = self._encode_image(key, img)
            tactile_features.append(feature)

        # FiLM: modulate RGB features with tactile conditioning
        modulated = list()
        if tactile_features:
            tactile_cond = torch.cat(tactile_features, dim=-1)  # (B, tactile_feat_dim)
            for key, rgb_feat in rgb_features:
                modulated.append(self.film_layers[key](rgb_feat, tactile_cond))
        else:
            # no tactile input, pass RGB features through unmodulated
            modulated = [feat for _, feat in rgb_features]

        # process low-dim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            low_dim_features.append(data)

        # concatenate modulated RGB + low-dim
        all_features = modulated + low_dim_features
        return torch.cat(all_features, dim=-1)

    def forward(self, obs_dict, mask=None):
        ori_batch_size, ori_seq_len = self.get_batch_size(obs_dict)
        features = self.multi_image_forward(obs_dict)
        result = self.mlp(features)
        if self.use_seq:
            if self.keep_horizon_dims:
                result = result.reshape(ori_batch_size, ori_seq_len, -1)
            else:
                result = result.reshape(ori_batch_size, -1)
        return result

    def get_batch_size(self, obs_dict):
        any_key = next(iter(obs_dict))
        any_tensor = obs_dict[any_key]
        return any_tensor.size(0), any_tensor.size(1)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

