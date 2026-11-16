# Ultralytics YOLO, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, RepConv

__all__ = ('RCSPELAN', 'CSPFQIM_dynamic', 'HCFM', 'CFRM')

class PSPModule(nn.Module):


    def __init__(self, grids=(1, 2, 3, 6), channels=256):
        super(PSPModule, self).__init__()

        self.grids = grids
        self.channels = channels

    def forward(self, feats):

        b, c , h , w = feats.size()
        ar = w / h

        return torch.cat([
            F.adaptive_avg_pool2d(feats, (self.grids[0], max(1, round(ar * self.grids[0])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[1], max(1, round(ar * self.grids[1])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[2], max(1, round(ar * self.grids[2])))).view(b, self.channels, -1),
            F.adaptive_avg_pool2d(feats, (self.grids[3], max(1, round(ar * self.grids[3])))).view(b, self.channels, -1)
        ], dim=2)

class LocalAttenModule(nn.Module):
    def __init__(self, in_channels=256,inter_channels=32):
        super(LocalAttenModule, self).__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, inter_channels,1),
            nn.Conv2d(inter_channels, in_channels, kernel_size=3, padding=1, bias=False))

        self.tanh_spatial = nn.Tanh()
        self.conv[1].weight.data.zero_()
        self.keras_init_weight()
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)

                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        res1 = x
        res2 = x

        x = self.conv(x)
        x_mask = self.tanh_spatial(x)

        res1 = res1 * x_mask

        return res1 + res2

class HCFM(nn.Module):
    def __init__(self, in_channels=512, grids=(6, 3, 2, 1)):
        super(HCFM, self).__init__()
        self.grids = grids
        inter_channels = in_channels // 2
        self.inter_channels = inter_channels

        self.reduce_channel = Conv(in_channels, inter_channels, 3)
        self.query_conv = nn.Conv2d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=inter_channels, out_channels=32, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=inter_channels, out_channels=self.inter_channels, kernel_size=1)
        self.key_channels = 32

        self.value_psp = PSPModule(grids, inter_channels)
        self.key_psp = PSPModule(grids, inter_channels)

        self.softmax = nn.Softmax(dim=-1)

        self.local_attention = LocalAttenModule(inter_channels,inter_channels//8)
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)

                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):

        x = self.reduce_channel(x)

        m_batchsize,_,h,w = x.size()

        query = self.query_conv(x).view(m_batchsize,32,-1).permute(0,2,1)

        key = self.key_conv(self.key_psp(x))

        sim_map = torch.matmul(query,key)

        sim_map = self.softmax(sim_map)

        value = self.value_conv(self.value_psp(x))


        context = torch.bmm(value,sim_map.permute(0,2,1))


        context = context.view(m_batchsize,self.inter_channels,h,w)

        context = self.local_attention(context)

        out = x + context

        return out

class CFRM(nn.Module):
    def __init__(self, inc):
        super(CFRM, self).__init__()
        hidc = inc[0]

        self.groups = 2
        self.conv_8 = Conv(inc[0], hidc, 3)
        self.conv_32 = Conv(inc[1], hidc, 3)

        self.conv_offset = nn.Sequential(
            Conv(hidc * 2, 64),
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )

        self.keras_init_weight()
        self.conv_offset[1].weight.data.zero_()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        cp, sp = x
        n, _, out_h, out_w = cp.size()


        sp = self.conv_32(sp)
        sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True)

        cp = self.conv_8(cp)

        conv_results = self.conv_offset(torch.cat([cp, sp], 1))

        sp = sp.reshape(n*self.groups,-1,out_h,out_w)
        cp = cp.reshape(n*self.groups,-1,out_h,out_w)

        offset_l = conv_results[:, 0:self.groups*2, :, :].reshape(n*self.groups,-1,out_h,out_w)
        offset_h = conv_results[:, self.groups*2:self.groups*4, :, :].reshape(n*self.groups,-1,out_h,out_w)


        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n*self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        cp = F.grid_sample(cp, grid_l , align_corners=True)
        sp = F.grid_sample(sp, grid_h , align_corners=True)

        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        att = 1 + torch.tanh(conv_results[:, self.groups*4:, :, :])
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]

        return sp

class RCSPELAN(nn.Module):
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(RCSPELAN, self).__init__()

        self.c = int(c2 * e)
        self.mid = int(self.c * scale)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)

        self.cv3 = RepConv(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 1)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.extend(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)


        self.alpha = nn.Parameter(torch.full((dim, 1, 1), 1e-3))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta

class FQIM(nn.Module):

    def __init__(self, dim, stage=3):
        super().__init__()
        if type(stage) is not int or stage not in (1, 2, 3, 4):
            raise ValueError(
                'FQIM stage must be 1 (DFS+RCC), 2 (+FGM), '
                f'3 (+FGM+QFI), or 4 (+QFI without FGM), but got {stage!r}'
            )
        self.stage = stage

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )

        self.out_conv = nn.Conv2d(dim, dim, 1)

        if self.stage in (3, 4):

            self.q_conv = nn.Conv2d(dim, dim, 1)
            self.attn_conv = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.Sigmoid()
            )


        self.conv = nn.Conv2d(dim, dim, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)


        self.fac_conv = nn.Conv2d(dim, dim, 1)
        self.fac_pool = nn.AdaptiveAvgPool2d(1)

        if self.stage in (2, 3):
            self.fgm = FGM(dim)

        self.act = nn.ReLU()

    def forward(self, x):


        stage = getattr(self, 'stage', 3)
        local = self.in_conv(x)


        frequency_weight = self.fac_conv(self.fac_pool(local))

        frequency = torch.fft.fft2(local, norm='backward')
        frequency = frequency_weight * frequency
        frequency = torch.fft.ifft2(frequency, dim=(-2, -1), norm='backward').abs()


        channel_weight = self.conv(self.pool(frequency))
        frequency_residual = channel_weight * frequency


        if stage in (2, 3):
            frequency_residual = self.fgm(frequency_residual)

        out = x + frequency_residual


        if stage in (3, 4):
            query = self.q_conv(local)
            query_residual = self.attn_conv(query * frequency_residual) * local
            out = out + query_residual
        out = self.act(out)
        return self.out_conv(out)

class CSPFQIM_dynamic(nn.Module):
    """Staged CSPFQIM whose DFS weights are dynamically predicted from the input."""

    def __init__(self, dim, e=0.25, stage=3):
        super().__init__()
        self.e = e
        self.branch_channels = int(dim * e)
        if not 0 < self.branch_channels < dim:
            raise ValueError(f'CSPFQIM_dynamic requires 0 < int(dim * e) < dim, but got dim={dim}, e={e}')

        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = FQIM(self.branch_channels, stage=stage)

    def forward(self, x):
        x = self.cv1(x)
        fqim_branch, identity = torch.split(
            x, [self.branch_channels, x.size(1) - self.branch_channels], dim=1
        )
        return self.cv2(torch.cat((self.m(fqim_branch), identity), dim=1))
