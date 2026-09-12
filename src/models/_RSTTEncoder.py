import einops
import torch
import torch.nn as nn

from ._RSTT_layers import EncoderLayer, InputProj, trunc_normal_
from ._utils import BaseBackbone


class RSTT_Downsample(nn.Module):
    """
    Uses a valid convolution to reduce each spatial dimension by two pixels.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
    ):
        super().__init__()
        # Kernel 1x3x3 ensures we don't mix Time yet, just Space.
        self.conv = nn.Conv3d(in_chans, out_chans, kernel_size=(1, 3, 3), bias=False)
        self.norm = nn.LayerNorm(out_chans)

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # -> (B, C, T, H, W) for Conv3d
        x = self.conv(x)

        # Norm expects (..., C), so we shuffle
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # -> (B, T, H_new, W_new, C_out)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # -> (B, C_out, T, H_new, W_new)

        return x.permute(0, 2, 1, 3, 4).contiguous()  # Back to (B, T, C, H, W)


class RSTTEncoder(BaseBackbone):
    def __init__(
        self,
        num_classes: int | None = 12,
        in_chans: int = 24,
        embed_dim: int = 96,
        num_frames: int = 12,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()

        # Depth Config: 9x9(96) -> 7x7(128) -> 5x5(192) -> 3x3(256)
        self.dims = [96, 128, 192, 256]
        self.num_classes = num_classes
        self.n_features = self.dims[-1]

        self.norm = nn.LayerNorm([num_frames, in_chans])

        self.input_proj = InputProj(
            in_channels=in_chans,
            embed_dim=embed_dim,
            kernel_size=3,
            stride=1,
            act_layer=nn.LeakyReLU,
        )

        # Stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 12)]
        # --- ENCODER LAYERS ---
        self.enc1 = EncoderLayer(
            dim=self.dims[0],
            num_heads=4,
            window_size=(3, 3),
            num_frames=num_frames,
            depth=2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[:2],
            norm_layer=norm_layer,
        )
        self.down1 = RSTT_Downsample(self.dims[0], self.dims[1])

        self.enc2 = EncoderLayer(
            dim=self.dims[1],
            num_heads=4,
            window_size=(7, 7),
            num_frames=num_frames,
            depth=2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[2:4],
            norm_layer=norm_layer,
        )
        self.down2 = RSTT_Downsample(self.dims[1], self.dims[2])

        self.enc3 = EncoderLayer(
            dim=self.dims[2],
            num_heads=8,
            window_size=(5, 5),
            num_frames=num_frames,
            depth=4,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[4:8],
            norm_layer=norm_layer,
        )
        self.down3 = RSTT_Downsample(self.dims[2], self.dims[3])

        # --- BOTTLENECK (3x3) ---
        # Window size matches feature map size (3,3) -> Global Attention
        self.bottleneck = EncoderLayer(
            dim=self.dims[3],
            num_heads=8,
            window_size=(3, 3),
            num_frames=num_frames,
            depth=4,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=enc_dpr[8:],
            norm_layer=norm_layer,
        )

        # --- HEAD ---
        if self.num_classes is not None:
            self.head = nn.Conv2d(
                self.n_features,
                self.num_classes,
                kernel_size=1,
            )

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.tempmaxpool = nn.MaxPool1d(num_frames)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.size(0)
        x = x.permute(0, 3, 4, 2, 1).contiguous()  # B, H, W, T, C
        # Statistics are computed over (time, band) for every spatial location.
        x = self.norm(x)
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        x = self.input_proj(x)

        skip1 = self.enc1(x)  # 9x9
        x = self.down1(skip1)

        skip2 = self.enc2(x)  # 7x7
        x = self.down2(skip2)

        skip3 = self.enc3(x)  # 5x5
        x = self.down3(skip3)

        x = self.bottleneck(x)  # 3x3

        x = einops.rearrange(x, "b t c h w -> (b h w) c t").contiguous()
        x = self.tempmaxpool(x).squeeze(-1)
        x = einops.rearrange(x, "(b h w) c -> b c h w", b=B, h=3, w=3).contiguous()

        if self.num_classes is None:
            return x
        logits = self.head(x)
        return logits
