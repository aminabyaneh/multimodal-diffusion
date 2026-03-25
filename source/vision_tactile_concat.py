from typing import Dict, Tuple, Union, Callable
import copy
import torch
import torch.nn as nn
import torchvision


from cleandiffuser.utils.crop_randomizer import CropRandomizer
from cleandiffuser.nn_condition import BaseNNCondition

from source.utils import replace_submodules, get_vit_backbone


class MultiImageObsConditionConcat(BaseNNCondition):
    """
    Multi-modal observation condition encoder that concatenates vision and tactile features.

    Processes RGB images and tactile data through separate encoders, then concatenates
    the resulting features for downstream conditioning.

    Args:
        condition: Dictionary of observations with keys mapping to tensors
                  - RGB/tactile images: (B, C, H, W) or (B, seq_len, C, H, W)
                  - Low-dim features: (B, D) or (B, seq_len, D)
        mask: Optional mask tensor (B, *mask_shape) or None

    Returns:
        condition: Concatenated feature tensor (B, *cond_out_shape)
    """
    def __init__(self,
            shape_meta: dict,
            rgb_model_name: str, # resnet18, resnet34, resnet50, vit_large_patch14_reg4_dinov2, vit_small_patch14_reg4_dinov2
            tactile_model_name: str = None, # resnet18, vit_large_patch14_reg4_dinov2, T3, Sparsh
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
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # rgb_model
        rgb_model = get_vit_backbone(rgb_model_name)
        if tactile_model_name is not None:
            tactile_model = get_vit_backbone(tactile_model_name)
        else:
            tactile_model = rgb_model # use same architecture as vision for tactile

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape

            if type == 'image_rgb' or type == 'image_tactile': # treat tactile as image input
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if isinstance(rgb_model, dict):
                    # have provided model for each key
                    this_model = rgb_model[key]
                else:
                    assert isinstance(rgb_model, nn.Module)
                    # have a copy of the rgb model
                    this_model = copy.deepcopy(rgb_model)

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
                # If resize_shape provided use that, otherwise try to infer from model
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
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims

        self.mlp = nn.Sequential(
            nn.Linear(self.output_shape(), emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

    def multi_image_forward(self, obs_dict):
        batch_size = None
        features = list()

        if self.use_seq:
            # input: (bs, horizon, c, h, w)
            for k in obs_dict.keys():
                obs_dict[k] = obs_dict[k].flatten(end_dim=1)

        # process rgb input
        # run each rgb obs to independent models
        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == self.key_shape_map[key]
            img = self.key_transform_map[key](img)

            # check for input mismatch
            if (input_size := getattr(self.key_model_map[key], "default_cfg", None)["input_size"]) != img.shape[1:]:
                img = torchvision.transforms.Resize(size=input_size[2:])(img)

            feature = self.key_model_map[key](img)
            features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        features = torch.cat(features, dim=-1)
        return features

    def forward(self, obs_dict, mask=None):
        ori_batch_size, ori_seq_len = self.get_batch_size(obs_dict)
        features = self.multi_image_forward(obs_dict)
        # linear embedding
        result = self.mlp(features)
        if self.use_seq:
            if self.keep_horizon_dims:
                result = result.reshape(ori_batch_size, ori_seq_len, -1)
            else:
                result = result.reshape(ori_batch_size, -1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            if self.use_seq:
                prefix = (batch_size, 1)
            else:
                prefix = (batch_size,)
            this_obs = torch.zeros(
                prefix + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.multi_image_forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape[0]

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

