import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torchvision import transforms
from MyDataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import models
import math
from timm.models.layers import DropPath
from GPEC import ERPCoordLatitudeWeight

# ===========================================================
# LayerNorm2d
# ===========================================================
class LayerNorm2d(nn.Module):
    """LayerNorm over channel dim for NCHW."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

# ===========================================================
# SEBlock 
# ===========================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(B, C)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.sig(s).view(B, C, 1, 1)
        return x * s

class ScaleAwareModule(nn.Module):
    """
    多尺度扩张卷积 + SE + 1x1 映射 + 像素级 softmax 权重融合
    输入: (B,C,H,W)
    输出: fused (B,C,H,W)
    """
    def __init__(self, channels, dilation_rates=(1, 3, 5), se_reduction=16):
        super().__init__()
        self.channels = channels
        self.dil_rates = dilation_rates

        # 为每个分支构建 dilated conv -> BN -> ReLU -> SE -> 1x1 映射
        self.branches = nn.ModuleList()
        for d in dilation_rates:
            branch = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                SEBlock(channels, reduction=se_reduction),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # 映射回 channels
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

        # 通过 1x1 生成 3 个 per-pixel attention logits -> softmax
        self.attn_conv = nn.Conv2d(channels, len(dilation_rates), kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B,C,H,W)
        outs = []
        for br in self.branches:
            outs.append(br(x))  # 每个 (B,C,H,W)
        # 逐像素求和（先求和再产生注意力）
        sum_feats = outs[0]
        for o in outs[1:]:
            sum_feats = sum_feats + o  # (B,C,H,W)

        logits = self.attn_conv(sum_feats)  # (B,3,H,W)
        attn = F.softmax(logits, dim=1)     # (B,3,H,W), 在 branch dim 上 softmax

        # 将 attn 分配到每个分支并加权融合
        fused = 0
        for i, o in enumerate(outs):
            a = attn[:, i:i+1, :, :]   # (B,1,H,W)
            fused = fused + o * a      # 广播到通道维后加权
        return fused, outs, attn

class MQCA(nn.Module):
    """
 
    输入: (B,C,H,W)
    输出: (B, out_channels, H, W)
    """
    def __init__(self, in_channels, out_channels=None, se_reduction=16, dilation_rates=(1,3,5)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

       
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

       
        self.scale_module = ScaleAwareModule(in_channels, dilation_rates=dilation_rates, se_reduction=se_reduction)

       
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=True),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 1x1 分支
        local = self.conv1x1(x)          # (B,C,H,W)

        # scale-aware module -> fused scale feature
        scale_fused, branches, attn = self.scale_module(x)  # (B,C,H,W)

        # global branch
        gap = F.adaptive_avg_pool2d(x, 1).view(B, C)         # (B,C)
        g = self.global_fc(gap)                             # (B,C)
        g = g.view(B, C, 1, 1).expand(-1, -1, H, W)         # (B,C,H,W)

        # concat and project
        cat = torch.cat([scale_fused, local, g], dim=1)     # (B,3C,H,W)
        out = self.out_conv(cat)                            # (B, out_channels, H, W)
        return out

# ===========================================================
# 大核+小核融合模块
# ===========================================================
class LargeSmallKernelFusion(nn.Module):
    """
    使用大核卷积捕获全局上下文，小核卷积保留局部信息，
    并通过门控机制动态融合两者。
    """
    def __init__(self, dim, large_kernel=9, small_kernel=3, reduction=4):
        super().__init__()
        pad_l = large_kernel // 2
        pad_s = small_kernel // 2

        # 大核分支（深度可分离卷积）
        self.large_branch = nn.Sequential(
            nn.Conv2d(dim, dim, large_kernel, padding=pad_l, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False)
        )

        # 小核分支（轻量卷积）
        self.small_branch = nn.Sequential(
            nn.Conv2d(dim, dim, small_kernel, padding=pad_s, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False)
        )

        # 门控模块：学习性融合权重
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        xl = self.large_branch(x)
        xs = self.small_branch(x)
        g = self.gate(x)
        out = g * xl + (1 - g) * xs
        return out


class CGDC(nn.Module):
    """
    大核 + 小核混合特征融合块（单向 cross 形式）
    """
    def __init__(self, dim, large_kernel=9, small_kernel=3, reduction=4, use_norm=True):
        super().__init__()
        self.norm_x = LayerNorm2d(dim) if use_norm else nn.Identity()
        self.norm_y = LayerNorm2d(dim) if use_norm else nn.Identity()

        self.fusion_x = LargeSmallKernelFusion(dim, large_kernel, small_kernel, reduction)
        self.fusion_y = LargeSmallKernelFusion(dim, large_kernel, small_kernel, reduction)

        # 融合后 1×1 卷积整合
        self.merge = nn.Conv2d(dim * 2, dim, 1, bias=False)

    def forward(self, x, y):
        x_ = self.fusion_x(self.norm_x(x))
        y_ = self.fusion_y(self.norm_y(y))
        fused = self.merge(torch.cat([x_ + y_, x], dim=1))
        return fused

class ACDC(nn.Module):
    def __init__(self, in_channels=32, kernel_sizes=(3, 5, 7, 9), normalize_kernel=True, fusion='learnable'):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.normalize_kernel = normalize_kernel
        self.fusion = fusion

        self.kernel_generators = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, k * k, kernel_size=1)
            )
            for k in kernel_sizes
        ])

        if fusion == 'learnable':
            self.fusion_weights = nn.Parameter(torch.ones(len(kernel_sizes)))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_mean = x.mean(dim=1, keepdim=True)
        att_maps = []

        for idx, k in enumerate(self.kernel_sizes):
            kernels = self.kernel_generators[idx](x).view(B, k * k)
            if self.normalize_kernel:
                kernels = F.softmax(kernels, dim=1)
            kernels = kernels.view(B, 1, k, k)

            x_mean_reshape = x_mean.view(1, B, H, W)
            att = F.conv2d(x_mean_reshape, weight=kernels, padding=k // 2, groups=B)
            att = att.view(B, 1, H, W)
            att_maps.append(att)

        att_stack = torch.stack(att_maps, dim=0)
        if self.fusion == 'sum':
            att = att_stack.sum(dim=0)
        elif self.fusion == 'avg':
            att = att_stack.mean(dim=0)
        elif self.fusion == 'learnable':
            weights = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1, 1, 1)
            att = (att_stack * weights).sum(dim=0)
        else:
            raise ValueError(f"Invalid fusion mode: {self.fusion}")

        att = self.sigmoid(att)
        return x * att

class LGDC(nn.Module):
    def __init__(self):
        super(LGDC, self).__init__()
        # Swin Transformer backbone
        backbone = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.layer_1 = self.backbone[0][0:3]    # 阶段1
        self.layer_2 = self.backbone[0][3:5]    # 阶段2  
        self.layer_3 = self.backbone[0][5:7]    # 阶段3
        self.layer_4 = self.backbone[0][7]      # 阶段4
        self.enhence = ERPCoordLatitudeWeight(
        channels=256,
        reduction=4,
        pole_min_weight=0.35,
        init_gamma=0.1,
        use_coord_embed=True
    )
        self.mqca_modules = nn.ModuleList([
            MQCA(in_channels=192, out_channels=192, se_reduction=16, dilation_rates=(1, 3, 5)),
            MQCA(in_channels=384, out_channels=384, se_reduction=16, dilation_rates=(1, 3, 5)),
            MQCA(in_channels=768, out_channels=768, se_reduction=16, dilation_rates=(1, 3, 5)),
            MQCA(in_channels=768, out_channels=768, se_reduction=16, dilation_rates=(1, 3, 5))
        ])
        
        self.mqca_adjust_layers = nn.ModuleList([
            nn.Conv2d(192, 256, kernel_size=1),
            nn.Conv2d(384, 256, kernel_size=1),
            nn.Conv2d(768, 256, kernel_size=1),
            nn.Conv2d(768, 256, kernel_size=1)
        ])
        
       
        self.acdc_modules = nn.ModuleList([
            ACDC(in_channels=192, kernel_sizes=(3, 5, 7, 9), fusion='learnable'),
            ACDC(in_channels=384, kernel_sizes=(3, 5, 7, 9), fusion='learnable'),
            ACDC(in_channels=768, kernel_sizes=(3, 5, 7, 9), fusion='learnable'),
            ACDC(in_channels=768, kernel_sizes=(3, 5, 7, 9), fusion='learnable')
        ])
        
     
        self.acdc_adjust_layers = nn.ModuleList([
            nn.Conv2d(192, 256, kernel_size=1),
            nn.Conv2d(384, 256, kernel_size=1),
            nn.Conv2d(768, 256, kernel_size=1),
            nn.Conv2d(768, 256, kernel_size=1)
        ])
        
    
        self.cgdc_fusion_modules = nn.ModuleList([
            CGDC(
                dim=256, 
                large_kernel=9, 
                small_kernel=3, 
                reduction=4, 
                use_norm=True
            ) for _ in range(4)
        ])
        
        # 上采样层 - 统一特征图尺寸到最大尺寸 (128x128)
        self.upsample_layers = nn.ModuleList([
            nn.Identity(),  # 阶段1: 128x128 保持不变
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),  # 阶段2: 64x64 → 128x128
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),  # 阶段3: 32x32 → 128x128
            nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),  # 阶段4: 32x32 → 128x128
        ])
        
        # 自适应池化
        self.adp = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        
        # 冻结前两层
        self.freeze_layers([self.layer_1, self.layer_2])
        
    def freeze_layers(self, layers):
        """冻结指定层"""
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
                
    def reshape_to_bchw(self, x):
        """将Swin的输出 [B, H, W, C] 转换为 [B, C, H, W]"""
        return rearrange(x, 'b h w c -> b c h w')
    
    def feature_forward(self, x):
        """特征前向传播"""
        # 获取四个阶段的特征
        layer1_out = self.layer_1(x)  # [B, 128, 128, 192]
        layer2_out = self.layer_2(layer1_out)  # [B, 64, 64, 384]  
        layer3_out = self.layer_3(layer2_out)  # [B, 32, 32, 768]
        layer4_out = self.layer_4(layer3_out)  # [B, 32, 32, 768]
        
        stage_outputs = [layer1_out, layer2_out, layer3_out, layer4_out]
        mqca_features = []
        acdc_features = []
        
        # 并行处理每个阶段的特征
        for i, stage_out in enumerate(stage_outputs):
          
            stage_bchw = self.reshape_to_bchw(stage_out).contiguous()
            
  
            mqca_out = self.mqca_modules[i](stage_bchw)  # [B, C, H, W]，保持输入维度
            mqca_out_adjusted = self.mqca_adjust_layers[i](mqca_out)  # [B, 256, H, W]
            mqca_features.append(mqca_out_adjusted)
            
    
            acdc_out = self.acdc_modules[i](stage_bchw)
            acdc_out_adjusted = self.acdc_adjust_layers[i](acdc_out)  # [B, 256, H, W]
            acdc_features.append(acdc_out_adjusted)
        
  
        cgdc_outputs = []
        for i in range(4):
            cgdc_fused = self.cgdc_fusion_modules[i](mqca_features[i], acdc_features[i])  # [B, 256, H, W]
            # 上采样到统一尺寸 (128x128)
            cgdc_fused_upsampled = self.upsample_layers[i](cgdc_fused)
            cgdc_outputs.append(cgdc_fused_upsampled)
        
        # 将四个融合输出相加 (现在都是 [B, 256, 128, 128])
        final_feature = torch.zeros_like(cgdc_outputs[0])
        for cgdc_out in cgdc_outputs:
            final_feature += cgdc_out
            
            
        # EMC
        final_feature = self.enhence(final_feature)
        # 全局平均池化并展平
        pooled = self.adp(final_feature)
        flattened = pooled.view(pooled.size(0), -1)
        
        return flattened
    
    def forward(self, x):
        feature = self.feature_forward(x)
        out = self.fc1(feature)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out).squeeze(1)
        return out


if __name__ == "__main__":
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    
   
    net = LGDC().to(device=device)
    
    # 打印模型参数
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"可训练参数数量: {total_params:,}")
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    test_dataset = MyDataset('/mnt/10T/liziyi/LargeKernel/JUFE-10K/resized_dis_1024',
                           '/mnt/10T/liziyi/LargeKernel/JUFE-10K/JUFE-10k_mos.csv',
                           mode='test', transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )
    
    # 测试
    for imgs, mos in test_loader:
        imgs = imgs.to(device=device)
        out = net(imgs)
        print(f"模型输出: {out}")
        print(f"真实MOS: {mos}")
        break