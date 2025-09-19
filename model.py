import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):

        x = self.down(x)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):

        x = self.up(x)

        return x


class ASMFunc(nn.Module):
    def __init__(self, in_channels):
        super(ASMFunc, self).__init__()

        self.norm = LayerNorm2d(in_channels)  # ln > bn
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convc = nn.Conv2d(in_channels, in_channels, kernel_size=1)


        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(self.norm(x))

        a = self.convc(self.gap(x))
        t = self.mlp(x)

        x = (x - a) * t + a
        x = self.out_conv(x)

        return x


class MultiScaleModule(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleModule, self).__init__()

        self.norm = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode="reflect"),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode="reflect"),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode="reflect"),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, padding_mode="reflect"),
        )

        self.a = nn.Parameter(torch.ones(1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)
        self.c = nn.Parameter(torch.ones(1), requires_grad=True)
        self.d = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):

        res = x

        x = self.norm(x)
        x2 = F.interpolate(x, scale_factor=0.75, mode="bilinear")
        x3 = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        x4 = F.interpolate(x, scale_factor=0.25, mode="bilinear")

        x = self.conv1(x)
        x2 = F.interpolate(self.conv2(x2), scale_factor=1.334, mode="bilinear")
        x3 = F.interpolate(self.conv2(x3), scale_factor=2, mode="bilinear")
        x4 = F.interpolate(self.conv2(x4), scale_factor=4, mode="bilinear")

        x = x * self.a + x2 * self.b + x3 * self.c + x4 * self.d

        return x + res


class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()

        self.msm = MultiScaleModule(in_channels)
        self.asm = ASMFunc(in_channels)

    def forward(self, x):

        res = x
        x = self.msm(x)
        x = self.asm(x)

        return x + res


class Generator(nn.Module):
    def __init__(self, base_channels=32):
        super(Generator, self).__init__()

        self.inconv = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.inconv_2 = nn.Conv2d(3, base_channels * 2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.inconv_3 = nn.Conv2d(3, base_channels * 4, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

        self.outconv1 = nn.Conv2d(base_channels * 4, 3, kernel_size=1)
        self.outconv2 = nn.Conv2d(base_channels * 2, 3, kernel_size=1)
        self.outconv = nn.Conv2d(base_channels, 3, kernel_size=1)

        self.down_1 = Down(base_channels, base_channels * 2)
        self.down_2 = Down(base_channels * 2, base_channels * 4)
        self.up_1 = Up(base_channels * 4, base_channels * 2)
        self.up_2 = Up(base_channels * 2, base_channels)

        self.encoder_1 = Block(base_channels)
        self.encoder_2 = Block(base_channels * 2)
        self.encoder_3 = Block(base_channels * 4)

        self.decoder_1 = Block(base_channels * 4)
        self.decoder_2 = Block(base_channels * 2)
        self.decoder_3 = Block(base_channels)

    def forward(self, input):

        input_2 = F.interpolate(input, scale_factor=0.5)
        input_4 = F.interpolate(input, scale_factor=0.25)
        x_2 = self.inconv_2(input_2)
        x_4 = self.inconv_3(input_4)

        x = self.inconv(input)           # 32 * hw
        x = self.encoder_1(x)
        skip1 = x

        x = self.down_1(x) + x_2        # 64 * hw/2
        x = self.encoder_2(x)
        skip2 = x

        x = self.down_2(x) + x_4        # 128 * hw/4
        x = self.encoder_3(x)

        x = self.decoder_1(x)
        out_4 = self.outconv1(x)
        x = self.up_1(x) + skip2         # 64 * hw/2

        x = self.decoder_2(x)
        out_2 = self.outconv2(x)
        x = self.up_2(x) + skip1         # 32 * hw

        x = self.decoder_3(x)
        out = self.outconv(x)

        return [input_4 + out_4, input_2 + out_2, input + out]


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Generator().to(device)

    # 参数量
    params = sum([param.numel() for param in model.parameters()])
    params_m = params / 1000000
    print(f'params:{params_m:.2f}M')

    # 计算量
    input_tensor = torch.randn(1, 3, 256, 256).to(device)  # 假设输入图像的尺寸为 (1, 3, 256, 256)
    flops, _ = profile(model, inputs=(input_tensor,))
    flops_g = flops / 1000000000
    print(f'FLOPs:{flops_g:.2f}G')