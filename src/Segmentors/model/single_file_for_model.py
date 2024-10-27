from collections import OrderedDict
from functools import lru_cache, partial
from numbers import Number

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.efficientnet_blocks import ConvBnAct, InvertedResidual

# from utils.ops import cus_sample

_DEBUG = True


def structure_loss(logits, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def cus_sample(
    feat: torch.Tensor,
    scale_factor=None,
    size=None,
    *,
    interpolation="bilinear",
    align_corners=False,
) -> torch.Tensor:
    """
    :param feat: 输入特征
    :param scale: scale
    :param size: size
    :param interpolation:
    :param align_corners: 具体差异可见https://www.yuque.com/lart/idh721/ugwn46
    :return: the resized tensor
    """
    interp_cfg = {}
    if size and not scale_factor:
        if isinstance(size, Number):
            size = (size, size)
        assert isinstance(size, (list, tuple)) and len(size) == 2
        size = [int(x) for x in size]
        if size == list(feat.shape[2:]):
            return feat
        interp_cfg["size"] = size
    elif scale_factor and not size:
        assert isinstance(scale_factor, (int, float))
        if scale_factor == 1:
            return feat
        recompute_scale_factor = None
        if isinstance(scale_factor, float):
            recompute_scale_factor = False
        interp_cfg["scale_factor"] = scale_factor
        interp_cfg["recompute_scale_factor"] = recompute_scale_factor
    else:
        raise NotImplementedError("only one of size or scale_factor should be defined")

    if interpolation == "nearest":
        if align_corners is False:
            align_corners = None
        assert align_corners is None, ("align_corners option can only be set with the interpolating modes: "
                                       "linear | bilinear | bicubic | trilinear, so we will set it to None")
    try:
        result = F.interpolate(feat, mode=interpolation, align_corners=align_corners, **interp_cfg)
    except NotImplementedError as e:
        print(f"shape: {feat.shape}\n"
              f"size={size}\n"
              f"scale_factor={scale_factor}\n"
              f"interpolation={interpolation}\n"
              f"align_corners={align_corners}")
        raise e
    except Exception as e:
        raise e
    return result


class ConvMlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = nn.ReLU(True)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act2 = nn.ReLU(True)
        self.fc3 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


@lru_cache()
def get_unfold_params_v2(width, kernel_size=8, num_kernels=3):
    # 确保kernel_size和num_kernels尽可能不变，因为这两者之间关联计算量
    if width <= kernel_size:
        kernel_size = width
        stride = 1
        dilation = 1
        num_kernels = 1
    elif width <= kernel_size + num_kernels:
        stride = 1
        dilation = 1
        num_kernels = width - kernel_size + 1
    else:

        def _get_stride_and_dilation(nk):
            d = 1
            while True:
                k = (kernel_size - 1) * d + 1
                if (width - k) % (nk - 1) == 0:
                    s = (width - k) // (nk - 1)
                    return s, d
                else:
                    d += 1
                if d >= kernel_size:
                    return None, d

        while True:
            stride, dilation = _get_stride_and_dilation(num_kernels)
            if stride and stride < ((kernel_size - 1) * dilation + 1):
                break
            num_kernels += 1

    params = dict(kernel_size=kernel_size, stride=stride, dilation=dilation)

    if _DEBUG:
        print(dict(width=width, num_kernels=num_kernels, **params))
    if not (stride >= 1 or width == stride * (num_kernels - 1) + (kernel_size - 1) * dilation + 1):
        raise ValueError(f"valid params does not exist for {dict(width=width, num_kernels=num_kernels, **params)}")
    return params, width, num_kernels


def reshape_tensor(x, bs, nh, pw, npw, pattern, unfold_params):
    x = F.unfold(x, **unfold_params)
    x = rearrange(x, pattern, b=bs, nh=nh, ph=pw, pw=pw, nph=npw, npw=npw)
    return x


class SpatialAttention(nn.Module):

    def __init__(self, dim, p=4, nh=2, nk=4):
        super().__init__()
        self.nh = nh
        self.p = p
        self.nk = nk
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None):
        if kv is None:
            kv = q
        N, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)

        # window-based attention
        unfold_params, _, npw = get_unfold_params_v2(width=W, kernel_size=self.p, num_kernels=self.nk)
        reshape_func = partial(reshape_tensor,
                               bs=N,
                               nh=self.nh,
                               pw=self.p,
                               npw=npw,
                               pattern="b (nh hd ph pw) (nph npw) -> b nh (nph npw) hd (ph pw)",
                               unfold_params=unfold_params)
        q, k, v = list(map(reshape_func, (q, k, v)))

        qk = torch.einsum("bnsdx, bnsdy -> bnsxy", q * q.shape[-2]**(-0.5), k)
        qk = qk.softmax(dim=-1)
        qkv = torch.einsum("bnsxy, bnsdy -> bnsdx", qk, v)

        qkv = rearrange(qkv, "b nh (nph npw) hd (ph pw) -> b (nh hd ph pw) (nph npw)", ph=self.p, pw=self.p, npw=npw)
        qkv = F.fold(qkv, output_size=(H, W), **unfold_params)  # b,nhhd,h,w
        return self.proj(qkv)


class SpatialCrossAttention(nn.Module):

    def __init__(self, i_dim, k_dim, v_dim, o_dim, p, nh=2, nk=4):
        super().__init__()
        self.p = p
        self.nh = nh
        self.nk = nk
        self.to_k = nn.Conv2d(2 * i_dim, 2 * k_dim, 1, groups=2)
        self.to_v = nn.Conv2d(2 * i_dim, 2 * v_dim, 1, groups=2)
        self.proj = nn.Conv2d(2 * v_dim, o_dim, 1)

    def forward(self, x1, x2):
        """B,C,H,W -> B,C,H,W"""
        assert x1.shape == x2.shape
        x1x2 = torch.cat([x1, x2], dim=1)
        k1, k2 = self.to_k(x1x2).chunk(2, dim=1)
        v1, v2 = self.to_v(x1x2).chunk(2, dim=1)
        N, C, H, W = k1.shape

        # window-based attention
        unfold_params, _, npw = get_unfold_params_v2(width=W, kernel_size=self.p, num_kernels=self.nk)
        reshape_func = partial(reshape_tensor,
                               bs=N,
                               nh=self.nh,
                               pw=self.p,
                               npw=npw,
                               pattern="b (nh hd ph pw) (nph npw) -> b nh (nph npw) hd (ph pw)",
                               unfold_params=unfold_params)
        k1, v1, k2, v2 = list(map(reshape_func, (k1, v1, k2, v2)))

        k1k2 = torch.einsum("bnsdx, bnsdy -> bnsxy", k1 * (k1.shape[-2]**(-0.5)), k2)
        a1 = k1k2.softmax(dim=-2)
        o1 = torch.einsum("bnsxy, bnsdx -> bnsdy", a1, v1)
        a2 = k1k2.softmax(dim=-1)
        o2 = torch.einsum("bnsxy, bnsdy -> bnsdx", a2, v2)

        o1 = rearrange(o1, "b nh (nph npw) hd (ph pw) -> b (nh hd ph pw) (nph npw)", ph=self.p, pw=self.p, npw=npw)
        o1 = F.fold(o1, output_size=(H, W), **unfold_params)  # b, hdnh,h,w
        o2 = rearrange(o2, "b nh (nph npw) hd (ph pw) -> b (nh hd ph pw) (nph npw)", ph=self.p, pw=self.p, npw=npw)
        o2 = F.fold(o2, output_size=(H, W), **unfold_params)  # b, hdnh,h,w

        o = torch.cat([o1, o2], dim=1)
        o = self.proj(o)
        return o


class ChannelAttention(nn.Module):

    def __init__(self, dim, nh):
        super().__init__()
        self.nh = nh
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q, kv=None):
        """B,C,H,W"""
        if kv is None:
            kv = q
        B, C, H, W = q.shape

        q = self.to_q(q)
        k, v = torch.chunk(self.to_kv(kv), 2, dim=1)
        q = q.reshape(B, self.nh, C // self.nh, H * W)
        k = k.reshape(B, self.nh, C // self.nh, H * W)
        v = v.reshape(B, self.nh, C // self.nh, H * W)

        q = q * (q.shape[-1]**(-0.5))
        qk = q @ k.transpose(-2, -1)
        qk = qk.softmax(dim=-1)
        qkv = qk @ v

        qkv = qkv.reshape(B, C, H, W)
        x = self.proj(qkv)
        return x


class WeightedDualDimSA(nn.Module):

    def __init__(self, dim, p, nh, ffn_expand=1, nk=4):
        super().__init__()
        # shared pre-norm
        self.norm1 = nn.BatchNorm2d(dim)
        # spatial attention
        self.alpha = nn.Parameter(data=torch.zeros(1))
        self.sa = SpatialAttention(dim, p=p, nh=nh, nk=nk)
        # channel attention
        self.beta = nn.Parameter(data=torch.zeros(1))
        self.ca = ChannelAttention(dim, nh=nh)
        # shared ffn
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = ConvMlp(in_features=dim, hidden_features=int(dim * ffn_expand), out_features=dim)

    def forward(self, x):
        normed_x = self.norm1(x)
        x = x + self.alpha.sigmoid() * self.sa(normed_x) + self.beta.sigmoid() * self.ca(normed_x)
        x = x + self.ffn(self.norm2(x))
        return x


class DualCA(nn.Module):

    def __init__(self, dim, p, nh, ffn_expand=1, nk=4):
        super().__init__()
        self.p = p

        self.conv_rgbd = nn.Sequential(nn.Conv2d(2 * dim, dim, 1), nn.ReLU(True),
                                       nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))
        self.rgb_norm1 = nn.BatchNorm2d(dim)
        self.depth_norm1 = nn.BatchNorm2d(dim)
        self.rgbd_s_ca = SpatialCrossAttention(i_dim=dim, k_dim=dim, v_dim=dim // 2, o_dim=dim, p=p, nh=nh, nk=nk)

        self.rgbd_norm1 = nn.BatchNorm2d(dim)
        self.rgbd_ffn1 = ConvMlp(in_features=dim, hidden_features=int(dim * ffn_expand), out_features=dim)

    def forward(self, rgb, depth):
        """B,C,H,W"""
        conv_rgbd = self.conv_rgbd(torch.cat([rgb, depth], dim=1))
        rgbd = conv_rgbd + self.rgbd_s_ca(self.rgb_norm1(rgb), self.depth_norm1(depth))
        rgbd = rgbd + self.rgbd_ffn1(self.rgbd_norm1(rgbd))
        return rgbd


class MSPatchLayer(nn.Module):

    def __init__(self, embed_dim, p, nh, n_rgb_sa, n_depth_sa, n_rgbd_sa, ffn_expand, nk):
        super().__init__()
        self.p = p
        self.rgb_cnn_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.depth_cnn_proj = nn.Conv2d(embed_dim, embed_dim, 1)

        self.rgb_selfblk = nn.Sequential(
            *[WeightedDualDimSA(embed_dim, nh=nh, p=p, nk=nk, ffn_expand=ffn_expand) for _ in range(n_rgb_sa)])
        self.depth_selfblk = nn.Sequential(
            *[WeightedDualDimSA(embed_dim, nh=nh, p=p, nk=nk, ffn_expand=ffn_expand) for _ in range(n_depth_sa)])

        self.crossblk = DualCA(embed_dim, nh=nh, p=p, nk=nk, ffn_expand=ffn_expand)
        self.rgbd_selfblk = nn.Sequential(
            *[WeightedDualDimSA(embed_dim, nh=nh, p=p, nk=nk, ffn_expand=ffn_expand) for _ in range(n_rgbd_sa)])

    def forward(self, rgb, depth, top_rgbd=None):
        """输入均为NCHW"""
        rgb = self.rgb_cnn_proj(rgb)
        depth = self.depth_cnn_proj(depth)

        # intra-modal
        rgb = self.rgb_selfblk(rgb)
        depth = self.depth_selfblk(depth)
        # inter-modal
        rgbd = self.crossblk(rgb, depth)
        # cross-sa
        rgbd = self.rgbd_selfblk(rgbd + top_rgbd)
        return rgbd


class MyASPP(nn.Module):
    #  this module is proposed in deeplabv3 and we use it in all of our baselines
    def __init__(self, in_dim, out_dim):
        super(MyASPP, self).__init__()
        self.conv1 = ConvBnAct(in_dim, out_dim, 1)
        self.conv2 = InvertedResidual(in_dim, out_dim, dilation=2)
        self.conv3 = InvertedResidual(in_dim, out_dim, dilation=5)
        self.conv4 = InvertedResidual(in_dim, out_dim, dilation=7)
        self.conv5 = ConvBnAct(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBnAct(5 * out_dim, out_dim, 3)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), size=x.size()[2:]))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels_list, hidden_channels=None, out_channels=None):
        super(FeaturePyramidNetwork, self).__init__()
        out_channels = out_channels or min(in_channels_list)
        hidden_channels = hidden_channels or out_channels

        self.top_inner_block = MyASPP(in_channels_list[-1], hidden_channels)

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            assert in_channels > 0
            self.inner_blocks.append(InvertedResidual(in_channels, hidden_channels))
            self.layer_blocks.append(InvertedResidual(hidden_channels, hidden_channels))
            self.out_blocks.append(InvertedResidual(hidden_channels, out_channels))

    def forward(self, x):
        # B, H, W, C = x[...].shape
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        top_inner = self.top_inner_block(x[-1])

        inner_lateral = self.inner_blocks[-1](x[-1])
        last_inner = self.layer_blocks[-1](inner_lateral + top_inner)

        results = []
        results.append(self.out_blocks[-1](last_inner))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            last_inner = self.layer_blocks[idx](inner_lateral + cus_sample(last_inner, scale_factor=2))
            results.insert(0, self.out_blocks[idx](last_inner))

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out, top_inner


class R50D_LPro_CFFN_BN_SP_LA_P7777_CA_NK2468_R05_H2_N221_C64_E2_SACAV4_Lite(nn.Module):

    def __init__(self,
                 ps=(7, 7, 7, 7),
                 nks=(2, 4, 6, 8),
                 nh=2,
                 n_rgb_sa=2,
                 n_depth_sa=2,
                 n_rgbd_sa=1,
                 ffn_expand=2,
                 embed_dim=64):
        super().__init__()
        self.encoder_rgb = timm.create_model(model_name="resnet50d",
                                             pretrained=True,
                                             features_only=True,
                                             out_indices=range(1, 5))
        self.encoder_depth = timm.create_model(model_name="resnet50d",
                                               pretrained=True,
                                               features_only=True,
                                               out_indices=range(1, 5))

        self.rgb_trans = FeaturePyramidNetwork(in_channels_list=(256, 512, 1024, 2048), out_channels=embed_dim)
        self.depth_trans = FeaturePyramidNetwork(in_channels_list=(256, 512, 1024, 2048), out_channels=embed_dim)
        self.top_cm_block = DualCA(embed_dim, nh=nh, p=ps[0], nk=nks[0], ffn_expand=ffn_expand)

        self.decs = nn.ModuleList([
            MSPatchLayer(
                embed_dim=embed_dim,
                p=p,
                nh=nh,
                nk=nk,
                n_rgb_sa=n_rgb_sa,
                n_depth_sa=n_depth_sa,
                n_rgbd_sa=n_rgbd_sa,
                ffn_expand=ffn_expand,
            ) for layer_idx, (p, nk) in enumerate(zip(ps, nks))
        ])
        self.decs.append(InvertedResidual(embed_dim, embed_dim))
        self.decs.append(InvertedResidual(embed_dim, 32))
        self.seg_head_0 = nn.Conv2d(32, 1, 1)

    def forward(self, images, depths):
        rgb_feats = self.encoder_rgb(images)
        depth_feats = self.encoder_depth(depths.repeat(1, 3, 1, 1))
        rgb_feats, rgb_aspp = self.rgb_trans({k: v for k, v in enumerate(rgb_feats)})
        depth_feats, depth_aspp = self.depth_trans({k: v for k, v in enumerate(depth_feats)})

        x = self.top_cm_block(rgb_aspp, depth_aspp)
        x = self.decs[0](rgb=rgb_feats[3], depth=depth_feats[3], top_rgbd=x)
        x = self.decs[1](rgb=rgb_feats[2], depth=depth_feats[2], top_rgbd=cus_sample(x, scale_factor=2))
        x = self.decs[2](rgb=rgb_feats[1], depth=depth_feats[1], top_rgbd=cus_sample(x, scale_factor=2))
        x = self.decs[3](rgb=rgb_feats[0], depth=depth_feats[0], top_rgbd=cus_sample(x, scale_factor=2))
        x = self.decs[4](cus_sample(x, scale_factor=2))
        x = self.decs[5](cus_sample(x, scale_factor=2))
        logits_0 = self.seg_head_0(x)
        return logits_0

