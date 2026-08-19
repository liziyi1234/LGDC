import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.GroupNorm(1, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ERPCoordLatitudeWeight(nn.Module):


    def __init__(
        self,
        channels,
        reduction=4,
        pole_min_weight=0.35,
        init_gamma=0.1,
        use_coord_embed=True
    ):
        super().__init__()

        self.channels = channels
        self.pole_min_weight = pole_min_weight
        self.use_coord_embed = use_coord_embed

        hidden_channels = max(channels // reduction, 32)

        # Coordinate embedding:
        # input coordinate maps:
        # sin(latitude), cos(latitude), sin(longitude), cos(longitude), latitude_weight, polar_weight
        if use_coord_embed:
            self.coord_embed = nn.Sequential(
                nn.Conv2d(6, hidden_channels, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
            )

        # Channel attention with latitude-weighted global pooling
        self.channel_attn = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention using feature statistics + ERP coordinates
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(6, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            ConvGNAct(channels * 2, channels, kernel_size=1),
            ConvGNAct(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )

        # Learnable latitude-prior strength
        # sigmoid(prior_strength) controls how strongly the latitude prior is used.
        self.prior_strength = nn.Parameter(torch.tensor(0.0))

        # Residual scale
        self.gamma = nn.Parameter(torch.tensor(init_gamma))

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        assert C == self.channels, \
            f"Input channel {C} does not match module channel {self.channels}"

        # Build ERP coordinate maps
        coord_maps, lat_weight = self._build_erp_coord_maps(
            H, W, device=x.device, dtype=x.dtype
        )

        # coord_maps: [1, 6, H, W]
        # lat_weight: [1, 1, H, W]

        coord_maps_b = coord_maps.expand(B, -1, -1, -1)
        lat_weight_b = lat_weight.expand(B, -1, -1, -1)

        # -------------------------
        # 1. Latitude prior weighting
        # -------------------------
        prior_strength = torch.sigmoid(self.prior_strength)

        # lat_weight is normalized around mean=1.
        # Equator > 1, poles < 1.
        geo_weight = 1.0 + prior_strength * (lat_weight_b - 1.0)

        x_geo = x * geo_weight

        # -------------------------
        # 2. Channel attention
        # -------------------------
        # Latitude-weighted global pooling
        denom = lat_weight_b.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        pooled = (x * lat_weight_b).sum(dim=(2, 3), keepdim=True) / denom

        ch_attn = self.channel_attn(pooled)
        x_geo = x_geo * ch_attn

        # -------------------------
        # 3. Spatial attention
        # -------------------------
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)

        # Use feature statistics + four coordinate maps
        # [avg, max, sin_lat, cos_lat, sin_lon, cos_lon]
        spatial_input = torch.cat(
            [
                avg_map,
                max_map,
                coord_maps_b[:, 0:1, :, :],
                coord_maps_b[:, 1:2, :, :],
                coord_maps_b[:, 2:3, :, :],
                coord_maps_b[:, 3:4, :, :]
            ],
            dim=1
        )

        sp_attn = self.spatial_attn(spatial_input)

        # Avoid fully suppressing any region
        x_geo = x_geo * (1.0 + sp_attn)

        # -------------------------
        # 4. Coordinate embedding
        # -------------------------
        if self.use_coord_embed:
            coord_feat = self.coord_embed(coord_maps)
            x_geo = x_geo + coord_feat

        # -------------------------
        # 5. Residual fusion
        # -------------------------
        fused = self.fusion(torch.cat([x, x_geo], dim=1))

        out = x + self.gamma * fused

        return out

    def _build_erp_coord_maps(self, H, W, device, dtype):
        """
        Build ERP coordinate maps.

        latitude:
            top    -> +pi/2
            middle -> 0
            bottom -> -pi/2

        longitude:
            left  -> -pi
            right -> +pi

        Returns:
            coord_maps: [1, 6, H, W]
            lat_weight: [1, 1, H, W]
        """

        # latitude coordinate
        y = torch.arange(H, device=device, dtype=dtype) + 0.5
        latitude = (0.5 - y / H) * math.pi

        sin_lat = torch.sin(latitude)
        cos_lat = torch.cos(latitude).clamp(min=0.0, max=1.0)

        # longitude coordinate
        x = torch.arange(W, device=device, dtype=dtype) + 0.5
        longitude = (x / W - 0.5) * 2.0 * math.pi

        sin_lon = torch.sin(longitude)
        cos_lon = torch.cos(longitude)

        sin_lat = sin_lat.view(1, 1, H, 1).expand(1, 1, H, W)
        cos_lat = cos_lat.view(1, 1, H, 1).expand(1, 1, H, W)

        sin_lon = sin_lon.view(1, 1, 1, W).expand(1, 1, H, W)
        cos_lon = cos_lon.view(1, 1, 1, W).expand(1, 1, H, W)

        # Latitude prior:
        # Equator high, poles low.
        lat_weight = self.pole_min_weight + (1.0 - self.pole_min_weight) * cos_lat

        # Normalize to keep feature magnitude stable.
        lat_weight = lat_weight / lat_weight.mean().clamp_min(1e-6)

        # Polar weight:
        # Larger near poles, used only as coordinate information.
        polar_weight = 1.0 - cos_lat

        coord_maps = torch.cat(
            [
                sin_lat,
                cos_lat,
                sin_lon,
                cos_lon,
                lat_weight,
                polar_weight
            ],
            dim=1
        )

        return coord_maps, lat_weight