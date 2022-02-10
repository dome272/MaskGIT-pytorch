import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        attn_resolutions = [16]
        ch_mult = [128, 128, 256, 256, 512]
        num_resolutions = len(ch_mult)
        block_in = ch_mult[num_resolutions-1]
        curr_res = 256 // 2**(num_resolutions-1)

        layers = [nn.Conv2d(args.latent_dim, block_in, kernel_size=3, stride=1, padding=1),
                  ResidualBlock(block_in, block_in),
                  NonLocalBlock(block_in),
                  ResidualBlock(block_in, block_in)
                  ]

        for i in reversed(range(num_resolutions)):
            block_out = ch_mult[i]
            for i_block in range(3):
                layers.append(ResidualBlock(block_in, block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    layers.append(NonLocalBlock(block_in))
            if i != 0:
                layers.append(UpSampleBlock(block_in))
                curr_res = curr_res * 2

        layers.append(GroupNorm(block_in))
        # layers.append(Swish())
        layers.append(nn.Conv2d(block_in, args.image_channels, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
