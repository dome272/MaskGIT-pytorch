import torch.nn as nn
import functools


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator (https://arxiv.org/pdf/1611.07004.pdf)
    """

    def __init__(self, args, num_filters_last=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding_size = 1
        sequence = [nn.Conv2d(args.image_channels, num_filters_last, kernel_size, stride=2, padding=padding_size),
                    nn.LeakyReLU(0.2)]
        num_filters_mult = 1
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            sequence += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, kernel_size,
                          2 if i < n_layers else 1, padding_size, bias=use_bias),
                norm_layer(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size, 1, padding_size)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
