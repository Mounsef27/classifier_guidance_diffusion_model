from torch import nn
from torch.nn import functional as F
from blocks import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CLS_Classifier(nn.Module):
    def __init__(
        self,
        source_channel, # 3
        unet_base_channel, # 128
        num_norm_groups, # 32
        head_dim, # 64
    ):
        super().__init__()

        #
        # For timestep embedding
        #
        self.pos_enc = CLS_PositionalEncoding(
            base_dim=unet_base_channel,
            hidden_dim=unet_base_channel*4,
            output_dim=unet_base_channel*4,
        )

        #
        # For U-Net style downsampling
        # (see 02-ddpm.ipynb)
        #
        self.down_conv = nn.Conv2d(
            source_channel,
            unet_base_channel,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.top_to_down = nn.ModuleList([
            # 1st layer
            CLS_ResnetBlock(
                in_channel=unet_base_channel,
                out_channel=unet_base_channel,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            CLS_ResnetBlock(
                in_channel=unet_base_channel,
                out_channel=unet_base_channel,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            CLS_ResnetBlock(        # downsampling
                in_channel=unet_base_channel,
                out_channel=unet_base_channel,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
                down=True,
            ),
            # 2nd layer
            CLS_ResnetAndAttention(
                in_channel=unet_base_channel,
                out_channel=unet_base_channel*2,
                num_heads=(unet_base_channel*2)//head_dim,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            CLS_ResnetAndAttention(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_heads=(unet_base_channel*2)//head_dim,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            CLS_ResnetBlock(        # downsampling
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
                down=True,
            ),
            # 3rd layer
            CLS_ResnetAndAttention(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_heads=(unet_base_channel*2)//head_dim,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            CLS_ResnetAndAttention(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_heads=(unet_base_channel*2)//head_dim,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            CLS_ResnetBlock(        # downsampling
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
                down=True,
            ),
            # 4th layer
            CLS_ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            CLS_ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
        ])
        self.middle = nn.ModuleList([
            CLS_ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
            CLS_AttentionBlock(
                channel=unet_base_channel*2,
                num_heads=(unet_base_channel*2)//head_dim,
                num_norm_groups=num_norm_groups,
            ),
            CLS_ResnetBlock(
                in_channel=unet_base_channel*2,
                out_channel=unet_base_channel*2,
                num_norm_groups=num_norm_groups,
                timestep_embedding_dim=unet_base_channel*4,
            ),
        ])

        #
        # For classification head
        #
        self.norm = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=unet_base_channel*2,
        )
        self.pool2d = CLS_AttentionPool2d(
            in_resolution=(4,4),
            channel=unet_base_channel*2,
            num_heads=(unet_base_channel*2)//head_dim,
        )
        self.classify = nn.Linear(unet_base_channel*2, num_classes)

    def forward(self, x, t):
        """
        Parameters
        ----------
        x : torch.tensor((batch_size, in_channel, width, height), dtype=float)
            Gaussian-noised images
        t : torch.tensor((batch_size), dtype=int)
            timestep
        """

        # generate time embedding
        time_embs = self.pos_enc(t)

        # apply U-Net style top-to-down
        out = self.down_conv(x)
        for block in self.top_to_down:
            out = block(out, time_embs)
        for block in self.middle:
            if isinstance(block, CLS_ResnetBlock):
                out = block(out, time_embs)
            elif isinstance(block, CLS_AttentionBlock):
                out = block(out)
            else:
                raise Exception("Unknown block")

        # apply classification head
        out = self.norm(out)
        out = F.silu(out)
        out = self.pool2d(out)
        out = self.classify(out)

        return out
    
