"""METEOR-M2 MSU-MR 画像欠損補完（8ch Single-Pass / 11ch Temporal Prev-Guided）推論モジュール。"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
    sobel,
    zoom,
)
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import h5py
except ImportError:  # pragma: no cover - h5py が無い場合は npz キャッシュのみで動作
    h5py = None  # type: ignore[assignment]


SCHEMA_VERSION = 1

# ひまわり8号キャッシュが利用できない場合の METEOR 晴天基準統計 (Ch1, Ch2, Ch4)
FALLBACK_METEOR_LAND_MEAN = (0.186, 0.242, 0.238)
FALLBACK_METEOR_LAND_STD = (0.028, 0.045, 0.040)
FALLBACK_METEOR_SEA_MEAN = (0.162, 0.125, 0.265)
FALLBACK_METEOR_SEA_STD = (0.022, 0.025, 0.038)


class FastResSEBlock(nn.Module):
    """Residual Squeeze-and-Excitation 畳み込みブロック。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = nn.Sequential(
            nn.Conv2d(out_ch, max(out_ch // 8, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_ch // 8, 8), out_ch, 1),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.act(self.proj(x) + h * self.se(h.mean(dim=(2, 3), keepdim=True)))


class PyramidContextBottleneck(nn.Module):
    """マルチスケール文脈集約ボトルネック。"""

    def __init__(self, ch: int) -> None:
        super().__init__()
        q = ch // 4
        self.b1 = nn.Sequential(nn.Conv2d(ch, q, 1, bias=False), nn.BatchNorm2d(q), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(ch, q, 3, padding=1, bias=False), nn.BatchNorm2d(q), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(ch, q, 1, bias=False), nn.BatchNorm2d(q), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.Conv2d(ch, q, 1, bias=False), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        p1, p2 = self.b1(x), self.b2(x)
        p3 = F.interpolate(self.b3(x), size=(h, w), mode="nearest")
        p4 = self.b4(x.mean(dim=(2, 3), keepdim=True)).expand(-1, -1, h, w)
        return self.act(x + self.fuse(torch.cat([p1, p2, p3, p4], dim=1)))


class PriorGateFusion(nn.Module):
    """空間事前知識（海岸線・前イメージ）をゲート融合するデコーダヘッド。"""

    def __init__(self, feat_ch: int, prior_ch: int = 2) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch + prior_ch, feat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch),
        )
        self.spatial_attn = nn.Sequential(nn.Conv2d(feat_ch + prior_ch, feat_ch, 1), nn.Sigmoid())
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([feat, prior], dim=1)
        return self.act(feat + self.gate(cat) * self.spatial_attn(cat))


class CoastlineGuidedSharpResSEUNet(nn.Module):
    """8ch 単体パス補完 ResSE-UNet ([corr3, obs3, land1, coast1] -> 3ch)。"""

    def __init__(self, base: int = 48) -> None:
        super().__init__()
        self.e1 = FastResSEBlock(8, base)
        self.e2 = FastResSEBlock(base, base * 2)
        self.b = FastResSEBlock(base * 2, base * 4)
        self.pyr = PyramidContextBottleneck(base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = FastResSEBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = FastResSEBlock(base * 2, base)
        self.prior_gate = PriorGateFusion(base, prior_ch=2)
        self.out = nn.Conv2d(base, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, corr: torch.Tensor, obs_mask: torch.Tensor, prior_2ch: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(torch.cat([corr, obs_mask, prior_2ch], dim=1))
        e2 = self.e2(self.pool(e1))
        b = self.pyr(self.b(self.pool(e2)))
        d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        pred = torch.sigmoid(self.out(self.prior_gate(d1, prior_2ch)))
        return corr * obs_mask + pred * (1.0 - obs_mask)


class PrevImageGuidedResSEUNet(nn.Module):
    """11ch 前イメージガイド ResSE-UNet ([corr3, obs3, prev3, land1, coast1] -> 3ch + Temporal Trust Gate)。"""

    def __init__(self, base: int = 48) -> None:
        super().__init__()
        self.e1 = FastResSEBlock(11, base)
        self.e2 = FastResSEBlock(base, base * 2)
        self.b = FastResSEBlock(base * 2, base * 4)
        self.pyr = PyramidContextBottleneck(base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = FastResSEBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = FastResSEBlock(base * 2, base)
        self.prior_gate = PriorGateFusion(base, prior_ch=5)
        self.out = nn.Conv2d(base, 4, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(
        self,
        corr: torch.Tensor,
        obs_mask: torch.Tensor,
        prev_3ch: torch.Tensor,
        prior_2ch: torch.Tensor,
    ) -> torch.Tensor:
        guide5 = torch.cat([prev_3ch, prior_2ch], dim=1)
        e1 = self.e1(torch.cat([corr, obs_mask, guide5], dim=1))
        e2 = self.e2(self.pool(e1))
        b = self.pyr(self.b(self.pool(e2)))
        d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        raw = self.out(self.prior_gate(d1, guide5))
        pred_net = torch.sigmoid(raw[:, :3])
        alpha_prev = torch.sigmoid(raw[:, 3:4]) * (1.0 - obs_mask.mean(dim=1, keepdim=True))
        fused = (1.0 - alpha_prev) * pred_net + alpha_prev * prev_3ch
        return corr * obs_mask + fused * (1.0 - obs_mask)


def select_torch_device() -> torch.device:
    """利用可能な計算デバイス（MPS > CPU）を返す。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_meteor_pass_bands(pass_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """パスディレクトリから MSU-MR-1/2/4 のバンド画像と有効画素マスクを読み込む。"""
    pdir = Path(pass_dir)
    band_files = ["MSU-MR-1.png", "MSU-MR-2.png", "MSU-MR-4.png"]
    ref_shape: tuple[int, int] | None = None
    loaded: dict[int, np.ndarray] = {}
    avail: list[str] = []
    for i, fname in enumerate(band_files):
        fpath = pdir / fname
        if fpath.exists():
            arr = np.asarray(Image.open(fpath).convert("L"), dtype=np.float32) / 255.0
            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                arr = np.asarray(
                    Image.fromarray((arr * 255.0).astype(np.uint8)).resize((ref_shape[1], ref_shape[0]), Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
            loaded[i] = arr
            avail.append(fname[7])

    if ref_shape is None:
        raise FileNotFoundError(f"MSU-MR バンド画像が見つかりません: {pdir}")

    h, w = ref_shape
    arr_hw3 = np.zeros((h, w, 3), dtype=np.float32)
    mask_hw3 = np.zeros((h, w, 3), dtype=np.float32)
    for i, a in loaded.items():
        arr_hw3[..., i] = a
        mask_hw3[..., i] = (a > 0.02).astype(np.float32)
    return arr_hw3, mask_hw3, "".join(avail)


def compute_input_stats(mask_hw3: np.ndarray, avail: str) -> dict[str, Any]:
    """観測マスクから全生存率・部分生存率・欠損率・最大連続欠損行数を算出する。"""
    h, w, _ = mask_hw3.shape
    all_v = mask_hw3.min(axis=2) > 0.5
    any_v = mask_hw3.max(axis=2) > 0.5
    part_v = any_v & (~all_v)
    none_v = ~any_v
    dead_rows = any_v.mean(axis=1) < 0.05
    max_gap, cur = 0, 0
    for is_dead in dead_rows:
        cur = cur + 1 if is_dead else 0
        if cur > max_gap:
            max_gap = cur

    return {
        "height": int(h),
        "width": int(w),
        "available_channels": avail,
        "all_valid_pct": round(float(all_v.mean() * 100.0), 1),
        "partial_valid_pct": round(float(part_v.mean() * 100.0), 1),
        "missing_pct": round(float(none_v.mean() * 100.0), 1),
        "any_valid_pct": round(float(any_v.mean() * 100.0), 1),
        "max_gap_px": int(max_gap),
    }


def classify_pass_tier(input_stats: dict[str, Any]) -> tuple[str, str]:
    """Notebook 10 のトリアージ基準に従ってパスの復元適性 Tier を判定する。"""
    any_pct = float(input_stats["any_valid_pct"])
    max_gap = int(input_stats["max_gap_px"])
    avail = str(input_stats["available_channels"])
    if any_pct < 15.0:
        return "Tier C", "極限欠損限界"
    if max_gap <= 16 and len(avail) == 3:
        return "Tier A", "高精細相互補完"
    return "Tier B", "Prior長大帯/欠落Ch合成"


def build_tone_features(arr_hw3: np.ndarray) -> np.ndarray:
    """SatDump 風の多項式トーン較正特徴量（10 次元）を生成する。"""
    c1 = np.clip(arr_hw3[..., 0], 0.0, 1.0)
    c2 = np.clip(arr_hw3[..., 1], 0.0, 1.0)
    c4 = np.clip(arr_hw3[..., 2], 0.0, 1.0)
    return np.stack(
        [
            np.ones_like(c1),
            c1,
            c2,
            c4,
            np.sqrt(c1),
            np.sqrt(c2),
            np.sqrt(c4),
            c1**2,
            c2**2,
            c4**2,
        ],
        axis=-1,
    )


def render_satdump_av221(
    arr_hw3: np.ndarray,
    W_av221: np.ndarray,
    valid_mask_hw1: np.ndarray | None = None,
) -> np.ndarray:
    """3 バンド配列を AVHRR 221 False Color (RGB [0, 1]) へ変換する。"""
    out = np.clip(build_tone_features(arr_hw3) @ W_av221, 0.0, 1.0).astype(np.float32)
    return out * valid_mask_hw1 if valid_mask_hw1 is not None else out


def render_satdump_msu124(
    arr_hw3: np.ndarray,
    W_msu124: np.ndarray | None,
    valid_mask_hw1: np.ndarray | None = None,
) -> np.ndarray:
    """3 バンド配列を MSU-MR 124 False Color (RGB [0, 1]) へ変換する。"""
    out = np.clip(build_tone_features(arr_hw3) @ W_msu124, 0.0, 1.0).astype(np.float32) if W_msu124 is not None else arr_hw3
    return out * valid_mask_hw1 if valid_mask_hw1 is not None else out


def render_inpaint_mask_overlay(av221_inpainted: np.ndarray, mask_hw3: np.ndarray) -> np.ndarray:
    """実測領域・部分救済領域(#D4A039)・全欠損AI補完領域(#E07A38)の可視化オーバーレイを生成する。"""
    out = np.clip(av221_inpainted, 0.0, 1.0).astype(np.float32).copy()
    all_v = mask_hw3.min(axis=2) > 0.5
    any_v = mask_hw3.max(axis=2) > 0.5
    part_v = any_v & (~all_v)
    miss_v = ~any_v

    amber = np.array([0.831, 0.627, 0.224], dtype=np.float32)  # #D4A039
    orange = np.array([0.878, 0.478, 0.220], dtype=np.float32)  # #E07A38

    if part_v.any():
        out[part_v] = out[part_v] * 0.70 + amber * 0.30
    if miss_v.any():
        out[miss_v] = (out[miss_v] * 0.80) * 0.64 + orange * 0.36
        boundary = binary_dilation(miss_v, iterations=1) & (~miss_v)
        if boundary.any():
            out[boundary] = out[boundary] * 0.15 + orange * 0.85
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def save_rgb_png(arr_hw3: np.ndarray, path: Path) -> None:
    """float32 RGB [0, 1] 配列を uint8 PNG ファイルとしてアトミック保存する。"""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = np.clip(arr_hw3 * 255.0, 0.0, 255.0).round().astype(np.uint8)
    tmp = dest.with_suffix(".tmp.png")
    Image.fromarray(img_u8, mode="RGB").save(tmp, format="PNG", optimize=True)
    tmp.replace(dest)


def make_coast_features(land_bin_hw: np.ndarray, sigma_edge: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2 値陸地マスクから連続陸地確率・沿岸ガウス重み・符号付き距離場を算出する。"""
    b = land_bin_hw > 0.5
    coast_px = (b & (~binary_erosion(b))) | ((~b) & binary_dilation(b))
    dist_to_coast = distance_transform_edt(~coast_px).astype(np.float32)
    coast_edge = np.exp(-(dist_to_coast**2) / (2.0 * (sigma_edge**2))).astype(np.float32)
    d_in = distance_transform_edt(b).astype(np.float32)
    d_out = distance_transform_edt(~b).astype(np.float32)
    signed_dist = d_in - d_out
    land_soft = np.clip(0.5 + 0.5 * np.tanh(signed_dist / 0.8), 0.0, 1.0).astype(np.float32)
    return land_soft, coast_edge, signed_dist


def _load_himawari_scenes(himawari_npz_path: Path) -> list[np.ndarray]:
    if not himawari_npz_path.exists():
        raise FileNotFoundError(f"ひまわりキャッシュが見つかりません: {himawari_npz_path}")
    cached = np.load(himawari_npz_path)
    scenes = [cached[k] for k in sorted(cached.files)]
    hf_hw_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--PubAccount--Himawari-Japen"
    if h5py is not None and hf_hw_dir.exists():
        nc5_list = sorted(hf_hw_dir.rglob("*20180101*.nc"))
        if nc5_list:
            with h5py.File(nc5_list[0], "r") as f:
                def _rs(k: str) -> np.ndarray:
                    ds = f[k]
                    raw = ds[::2, ::2]
                    sc = float(np.atleast_1d(ds.attrs.get("scale_factor", 1.0))[0])
                    off = float(np.atleast_1d(ds.attrs.get("add_offset", 0.0))[0])
                    return np.where(raw > -30000, raw.astype(np.float32) * sc + off, np.nan)

                a3 = np.nan_to_num(_rs("albedo_03"), nan=0.0)
                a4 = np.nan_to_num(_rs("albedo_04"), nan=0.0)
                t7 = _rs("tbb_07")
            scenes.append(
                np.stack(
                    [
                        np.power(np.clip(a3, 0, 1), 1 / 2.2),
                        np.power(np.clip(a4, 0, 1), 1 / 2.2),
                        np.nan_to_num(np.clip((310 - t7) / 70, 0, 1), nan=0.0),
                    ],
                    axis=-1,
                ).astype(np.float32)
            )
    return scenes


def build_or_load_cal_base_atlas(
    clean_land_hw: np.ndarray,
    cache_path: Path,
    himawari_npz_path: Path,
    meteor_dir: Path,
) -> np.ndarray:
    """METEOR 空間へ放射量較正した雲除去地表ベースアトラスを構築またはキャッシュ読込する。"""
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path)
        if "cal_base_hw" in data and data["cal_base_hw"].shape[:2] == clean_land_hw.shape:
            return data["cal_base_hw"].astype(np.float32)

    scenes = _load_himawari_scenes(himawari_npz_path)
    scenes_stack = np.stack(scenes, axis=0)
    cloud_score = scenes_stack[..., 0] + 0.6 * scenes_stack[..., 2] - 0.5 * (scenes_stack[..., 1] - scenes_stack[..., 0]) * clean_land_hw[None]
    best_t = np.argmin(cloud_score, axis=0)
    yy_h, xx_h = np.meshgrid(np.arange(clean_land_hw.shape[0]), np.arange(clean_land_hw.shape[1]), indexing="ij")
    raw_base_hw = scenes_stack[best_t, yy_h, xx_h].astype(np.float32)
    raw_base_hw[:5, :] = raw_base_hw[5:6, :]

    # METEOR-M2 3_20261026_003009 の晴天陸・海統計が利用可能なら実測から算出、無ければ既定統計を使用
    ref_dir = Path(meteor_dir) / "METEOR-M2 3_20261026_003009"
    if (ref_dir / "MSU-MR-1.png").exists() and (ref_dir / "MSU-MR-2.png").exists() and (ref_dir / "MSU-MR-4.png").exists():
        arr_1026, msk_1026, _ = load_meteor_pass_bands(ref_dir)
        m_clr = (msk_1026.min(axis=2) > 0.5) & (arr_1026[..., 0] < 0.22)
        m_land = m_clr & ((arr_1026[..., 1] - arr_1026[..., 0]) > 0.018)
        m_sea = m_clr & ((arr_1026[..., 1] - arr_1026[..., 0]) <= 0.018)
        meteor_stats = [
            (
                float(arr_1026[..., ci][m_land].mean()),
                float(arr_1026[..., ci][m_land].std()),
                float(arr_1026[..., ci][m_sea].mean()),
                float(arr_1026[..., ci][m_sea].std()),
            )
            for ci in range(3)
        ]
    else:
        meteor_stats = [
            (
                FALLBACK_METEOR_LAND_MEAN[ci],
                FALLBACK_METEOR_LAND_STD[ci],
                FALLBACK_METEOR_SEA_MEAN[ci],
                FALLBACK_METEOR_SEA_STD[ci],
            )
            for ci in range(3)
        ]

    land_b, sea_b = clean_land_hw > 0.5, clean_land_hw <= 0.5
    w_land_soft_cal = gaussian_filter(clean_land_hw, sigma=0.8)
    cal_base_hw = np.zeros_like(raw_base_hw)
    for ci in range(3):
        vals_l = raw_base_hw[..., ci][land_b]
        vals_s = raw_base_hw[..., ci][sea_b]
        clr_l = vals_l[vals_l < 0.55]
        clr_s = vals_s[vals_s < 0.42]
        mu_hl, sd_hl = float(clr_l.mean()), max(float(clr_l.std()), 1e-4)
        mu_hs, sd_hs = float(clr_s.mean()), max(float(clr_s.std()), 1e-4)
        mu_ml, sd_ml, mu_ms, sd_ms = meteor_stats[ci]
        l_map = (np.clip(raw_base_hw[..., ci], 0.05, 0.70) - mu_hl) * (sd_ml / sd_hl) + mu_ml
        s_map = (np.clip(raw_base_hw[..., ci], 0.03, 0.45) - mu_hs) * (sd_ms / sd_hs) + mu_ms
        cal_base_hw[..., ci] = np.clip(w_land_soft_cal * l_map + (1.0 - w_land_soft_cal) * s_map, 0.03, 0.92)

    cal_base_hw = cal_base_hw.astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, cal_base_hw=cal_base_hw)
    tmp.replace(cache_path)
    return cal_base_hw


def _eval_orbital_warp(p: Any, py: np.ndarray, px: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    cy, cx, ayy, ayx, axy, axx, qy, qx = p
    u, v = px / (width * 0.5), py / 500.0
    hy = cy + ayy * py + ayx * px + qy * (u**2) * 30.0
    hx = cx + axy * py + axx * px + qx * px * (u**2) - 40.0 * v * u
    return hy, hx


def _parse_utc_minutes(pass_id: str) -> float | None:
    m = re.search(r"_(\d{2})(\d{2})(\d{2})$", pass_id)
    if not m:
        return None
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return hh * 60.0 + mm + ss / 60.0


def infer_internal_spectral_prior(arr_hw3: np.ndarray, mask_hw3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """東アジア視野外・夜間赤外専用パス向けの自己整合スペクトル Prior を推定する。"""
    has_12 = (mask_hw3[..., 0] > 0.5) & (mask_hw3[..., 1] > 0.5)
    h, w = mask_hw3.shape[:2]
    vis1_mean = float(arr_hw3[..., 0][has_12].mean()) if has_12.sum() >= 200 else 0.0
    vis2_mean = float(arr_hw3[..., 1][has_12].mean()) if has_12.sum() >= 200 else 0.0
    # 夜間等で可視/近赤外信号がほぼゼロの場合は中立 Prior を返す
    if has_12.sum() < 200 or (vis1_mean < 0.065 and vis2_mean < 0.065):
        lp = np.full((h, w), 0.5, dtype=np.float32)
        ce = np.zeros((h, w), dtype=np.float32)
        sd = np.zeros((h, w), dtype=np.float32)
        return lp, ce, sd

    obs_land = np.clip((arr_hw3[..., 1] - arr_hw3[..., 0] - 0.015) / 0.045, 0.0, 1.0)
    prior = obs_land.copy()
    for x in range(w):
        vy = has_12[:, x]
        if vy.any() and (~vy).any():
            prior[~vy, x] = np.interp(np.where(~vy)[0], np.where(vy)[0], obs_land[vy, x])
        elif not vy.any():
            prior[:, x] = 0.5
    prior_smooth = gaussian_filter(prior, sigma=(8.0, 1.5)).astype(np.float32)
    b = (prior_smooth > 0.45).astype(np.float32)
    _, _, sd_s = make_coast_features(b, sigma_edge=2.0)
    edge_soft = np.clip(np.hypot(sobel(prior_smooth, axis=0), sobel(prior_smooth, axis=1)) * 1.4, 0.0, 0.85).astype(np.float32)
    return prior_smooth, edge_soft, sd_s


def _is_night_ir_only_pass(arr_hw3: np.ndarray, mask_hw3: np.ndarray) -> bool:
    """可視(Ch1)・近赤外(Ch2)が夜間等で極低照度かつ赤外(Ch4)主体のパスか判定する。"""
    m1 = mask_hw3[..., 0] > 0.5
    m2 = mask_hw3[..., 1] > 0.5
    m4 = mask_hw3[..., 2] > 0.5
    v1_mean = float(arr_hw3[..., 0][m1].mean()) if m1.sum() > 200 else 0.0
    v2_mean = float(arr_hw3[..., 1][m2].mean()) if m2.sum() > 200 else 0.0
    return (m1.sum() <= 200 or v1_mean < 0.065) and (m2.sum() <= 200 or v2_mean < 0.065) and (m4.sum() > 200)


def register_pass_to_east_asia(
    pass_id: str,
    arr_hw3: np.ndarray,
    mask_hw3: np.ndarray,
    clean_land_hw: np.ndarray,
    hw_smooth: np.ndarray,
    cal_base_hw: np.ndarray,
    is_golden: bool = False,
) -> dict[str, Any]:
    """任意パスを極東アトラス座標へ位置合わせし、登録モード・座標格子・Prior を返す。"""
    h, w, _ = arr_hw3.shape
    if is_golden or "GOLDEN" in pass_id.upper():
        lp, ce, sd = infer_internal_spectral_prior(arr_hw3, mask_hw3)
        return {
            "mode": "outside_fov",
            "hy": None,
            "hx": None,
            "land_bin": (lp > 0.45).astype(np.float32),
            "land_soft": lp,
            "coast_edge": ce,
            "signed_dist": sd,
            "clear_agree_pct": 0.0,
        }

    c1, c2 = arr_hw3[..., 0], arr_hw3[..., 1]
    v12 = (mask_hw3[..., 0] > 0.5) & (mask_hw3[..., 1] > 0.5)
    clear = v12 & (c1 < 0.25)

    # 夜間（可視光ゼロ・赤外主体）は昼間ベースの前イメージを誤適用しないよう Single-Pass へ
    if _is_night_ir_only_pass(arr_hw3, mask_hw3):
        lp, ce, sd = infer_internal_spectral_prior(arr_hw3, mask_hw3)
        return {
            "mode": "internal_night_or_sparse",
            "hy": None,
            "hx": None,
            "land_bin": (lp > 0.45).astype(np.float32),
            "land_soft": lp,
            "coast_edge": ce,
            "signed_dist": sd,
            "clear_agree_pct": 0.0,
        }

    # Case A: 十分な晴天海岸ピクセルがある場合 (Stage 1 Powell + Stage 2 TPS)
    if clear.sum() >= 2000:
        obs_cont = np.clip((c2 - c1 - 0.008) / 0.028, 0.0, 1.0).astype(np.float32)
        obs_bin = ((c2 - c1) > 0.018).astype(np.float32)
        ogx = sobel(gaussian_filter(obs_cont, sigma=1.2), axis=1)
        ogy = sobel(gaussian_filter(obs_cont, sigma=1.2), axis=0)
        obs_edge = clear & (np.hypot(ogx, ogy) > 0.35) & binary_closing(v12, structure=np.ones((3, 3)))
        px_w = np.where(binary_dilation(obs_edge, iterations=6), 4.0, 1.0).astype(np.float32)

        ds = 6 if h >= 120 else 3
        ym, xm = np.meshgrid(np.arange(0, h, ds, dtype=np.float32), np.arange(0, w, ds, dtype=np.float32), indexing="ij")
        clr_s = clear[::ds, ::ds]
        py, px = ym[clr_s] - h * 0.5, xm[clr_s] - w * 0.5
        ob, pw = obs_bin[::ds, ::ds][clr_s], px_w[::ds, ::ds][clr_s]
        ob_b = ob > 0.5
        ob_c = ob - (ob * pw).sum() / pw.sum()

        best_s, best_p = -1e9, (300.0, 700.0, 0.46, -0.02, -0.18, 0.72, 0.0, -0.10)
        for ayy in [0.46, -0.46]:
            axx_sign = 1.0 if ayy > 0 else -1.0
            for axy in [-0.28, -0.16, 0.0, 0.14]:
                for ayx in [-0.03, 0.02]:
                    for axx_mag in [0.68, 0.84]:
                        axx = axx_sign * axx_mag
                        for cy in np.linspace(140, 960, 8):
                            for cx in np.linspace(350, 1100, 7):
                                p = (cy, cx, ayy, ayx, axy, axx, 0.0, -0.10)
                                hy, hx = _eval_orbital_warp(p, py, px, w)
                                if ((hy >= 2) & (hy < 1299) & (hx >= 2) & (hx < 1348)).mean() < 0.72:
                                    continue
                                wv = map_coordinates(hw_smooth, [np.clip(hy, 0, 1300), np.clip(hx, 0, 1349)], order=1, mode="nearest")
                                wb = wv > 0.5
                                iou_l = float((wb & ob_b).sum()) / float((wb | ob_b).sum() + 1e-6)
                                iou_s = float(((~wb) & (~ob_b)).sum()) / float(((~wb) | (~ob_b)).sum() + 1e-6)
                                wc = wv - (wv * pw).sum() / pw.sum()
                                w_corr = float((wc * ob_c * pw).sum() / (np.sqrt((wc**2 * pw).sum() * (ob_c**2 * pw).sum()) + 1e-6))
                                sc = 0.5 * w_corr + 0.5 * 0.5 * (iou_l + iou_s)
                                if sc > best_s:
                                    best_s, best_p = sc, p

        ds2 = 4 if h >= 120 else 2
        ym2, xm2 = np.meshgrid(np.arange(0, h, ds2, dtype=np.float32), np.arange(0, w, ds2, dtype=np.float32), indexing="ij")
        clr2 = clear[::ds2, ::ds2]
        py2, px2 = ym2[clr2] - h * 0.5, xm2[clr2] - w * 0.5
        ob2, ob2_b, pw2 = obs_bin[::ds2, ::ds2][clr2], obs_bin[::ds2, ::ds2][clr2] > 0.5, px_w[::ds2, ::ds2][clr2]
        sign_y = 1.0 if best_p[2] > 0 else -1.0

        def _obj(p: np.ndarray) -> float:
            cy, cx, ayy, ayx, axy, axx, qy, qx = p
            if not (0.34 <= abs(ayy) <= 0.62 and 0.45 <= abs(axx) <= 1.15 and abs(ayx) <= 0.15 and abs(axy) <= 0.45):
                return 10.0
            if np.sign(ayy) != sign_y or np.sign(axx) != sign_y or abs(qy) > 0.8 or abs(qx) > 0.55:
                return 10.0
            hy, hx = _eval_orbital_warp(p, py2, px2, w)
            inb = ((hy >= 0) & (hy < 1300) & (hx >= 0) & (hx < 1349)).mean()
            if inb < 0.72:
                return 5.0 - float(inb)
            wv = map_coordinates(hw_smooth, [np.clip(hy, 0, 1300), np.clip(hx, 0, 1349)], order=1, mode="nearest")
            w_acc = float((((wv > 0.5) == ob2_b) * pw2).sum() / pw2.sum())
            w_bce = float(((wv - ob2) ** 2 * pw2).sum() / pw2.sum())
            iou_l = float(((wv > 0.5) & ob2_b).sum()) / float(((wv > 0.5) | ob2_b).sum() + 1e-6)
            return (1.0 - w_acc) + 0.5 * w_bce + 0.25 * (1.0 - iou_l)

        p_opt = minimize(_obj, np.array(best_p, dtype=np.float64), method="Powell", options={"maxiter": 450, "ftol": 1e-4}).x
        yy_s, xx_s = np.meshgrid(np.arange(0, h, 4, dtype=np.float32), np.arange(0, w, 4, dtype=np.float32), indexing="ij")
        hy0, hx0 = _eval_orbital_warp(p_opt, yy_s - h * 0.5, xx_s - w * 0.5, w)
        hy0_full = zoom(hy0, (h / yy_s.shape[0], w / yy_s.shape[1]), order=1)
        hx0_full = zoom(hx0, (h / yy_s.shape[0], w / yy_s.shape[1]), order=1)
        warped0 = (map_coordinates(clean_land_hw, [hy0_full, hx0_full], order=1, mode="nearest") > 0.5).astype(np.float32)

        iou_land = float(((warped0[clear] > 0.5) & (obs_bin[clear] > 0.5)).sum()) / float(((warped0[clear] > 0.5) | (obs_bin[clear] > 0.5)).sum() + 1e-6)
        if iou_land < 0.45:
            lp, ce, sd = infer_internal_spectral_prior(arr_hw3, mask_hw3)
            return {
                "mode": "outside_fov",
                "hy": None,
                "hx": None,
                "land_bin": (lp > 0.45).astype(np.float32),
                "land_soft": lp,
                "coast_edge": ce,
                "signed_dist": sd,
                "clear_agree_pct": round(float(((lp > 0.45) == (obs_bin > 0.5))[clear].mean() * 100.0), 1),
            }

        hy_fin, hx_fin = hy0_full, hx0_full
        if h >= 120:
            edge_zone = binary_dilation(np.hypot(sobel(warped0, axis=0), sobel(warped0, axis=1)) > 0.3, iterations=10)
            src_pts, res_pts, sh_cnt = [], [], 0
            for ay in np.linspace(0, h - 1, 7):
                for ax in [0.0, w * 0.25, w - 1.0]:
                    src_pts.append([ay, ax])
                    res_pts.append([0.0, 0.0])
            for cy_w in range(20, h - 20, 32):
                for cx_w in range(180, w - 120, 48):
                    y0, y1 = max(0, cy_w - 22), min(h, cy_w + 22)
                    x0, x1 = max(0, cx_w - 40), min(w, cx_w + 40)
                    c_sub = clear[y0:y1, x0:x1]
                    if c_sub.mean() < 0.35 or (edge_zone[y0:y1, x0:x1] & c_sub).sum() < 15:
                        continue
                    if not (0.12 <= obs_bin[y0:y1, x0:x1][c_sub].mean() <= 0.88):
                        continue
                    py_l, px_l = np.where(c_sub)
                    hy_b = hy0_full[py_l + y0, px_l + x0]
                    hx_b = hx0_full[py_l + y0, px_l + x0]
                    ov_l = obs_cont[py_l + y0, px_l + x0]
                    ob_l = obs_bin[py_l + y0, px_l + x0] > 0.5
                    w0 = map_coordinates(hw_smooth, [np.clip(hy_b, 0, 1300), np.clip(hx_b, 0, 1349)], order=1, mode="nearest")
                    s0 = float(((w0 > 0.5) == ob_l).mean()) - 0.30 * float(np.abs(w0 - ov_l).mean())
                    best_d, best_sc = (0.0, 0.0), s0
                    for dhy in range(-5, 6, 1):
                        for dhx in range(-8, 9, 2):
                            wv = map_coordinates(hw_smooth, [np.clip(hy_b + dhy, 0, 1300), np.clip(hx_b + dhx, 0, 1349)], order=1, mode="nearest")
                            sc = float(((wv > 0.5) == ob_l).mean()) - 0.30 * float(np.abs(wv - ov_l).mean()) - 0.002 * (dhy**2 + dhx**2)
                            if sc > best_sc:
                                best_sc, best_d = sc, (float(dhy), float(dhx))
                    if best_sc >= s0 + 0.012 and best_sc > 0.72:
                        src_pts.append([float(cy_w), float(cx_w)])
                        res_pts.append([best_d[0], best_d[1]])
                        sh_cnt += 1
            if sh_cnt >= 3:
                rbf = RBFInterpolator(np.array(src_pts, dtype=np.float32), np.array(res_pts, dtype=np.float32), kernel="thin_plate_spline", smoothing=35.0)
                d_sub = rbf(np.stack([yy_s.ravel(), xx_s.ravel()], axis=-1)).reshape(yy_s.shape[0], yy_s.shape[1], 2)
                hy_fin = hy0_full + zoom(np.clip(d_sub[..., 0], -6.0, 6.0), (h / yy_s.shape[0], w / yy_s.shape[1]), order=1)
                hx_fin = hx0_full + zoom(np.clip(d_sub[..., 1], -8.0, 8.0), (h / yy_s.shape[0], w / yy_s.shape[1]), order=1)

        w_fin = (map_coordinates(clean_land_hw, [hy_fin, hx_fin], order=1, mode="nearest") > 0.5).astype(np.float32)
        land_s, coast_e, signed_d = make_coast_features(w_fin, sigma_edge=2.0)
        agree = float(((w_fin > 0.5) == (obs_bin > 0.5))[clear].mean() * 100.0)
        return {
            "mode": "clear_orbital_tps",
            "hy": hy_fin.astype(np.float32),
            "hx": hx_fin.astype(np.float32),
            "land_bin": w_fin,
            "land_soft": land_s,
            "coast_edge": coast_e,
            "signed_dist": signed_d,
            "clear_agree_pct": round(agree, 1),
        }

    # Case B: 高欠損・短尺・単一バンド欠落パス -> UTC受信時刻＋生存輝度相関
    utc_min = _parse_utc_minutes(pass_id) or 30.0
    cx_nom = float(np.clip(726.0 - (utc_min - 30.15) * 4.5, 420.0, 1020.0))
    yy_f, xx_f = np.meshgrid(np.arange(0, h, dtype=np.float32), np.arange(0, w, dtype=np.float32), indexing="ij")
    any_m = mask_hw3.max(axis=2) > 0.5
    best_corr, best_p = -1e9, (460.0, cx_nom, 0.46, -0.02, -0.18, 0.72, 0.0, -0.10)

    if any_m.sum() >= 200:
        avail_chs = [i for i in range(3) if mask_hw3[..., i].sum() > 200]
        ds = 2 if h < 120 else 4
        ym, xm = np.meshgrid(np.arange(0, h, ds, dtype=np.float32), np.arange(0, w, ds, dtype=np.float32), indexing="ij")
        am_s = any_m[::ds, ::ds]
        py, px = ym[am_s] - h * 0.5, xm[am_s] - w * 0.5
        for cy in np.linspace(260, 820, 9):
            for cx in np.linspace(max(320, cx_nom - 160), min(1100, cx_nom + 160), 7):
                p = (cy, cx, 0.46, -0.02, -0.18, 0.74, 0.0, -0.10)
                hy, hx = _eval_orbital_warp(p, py, px, w)
                if ((hy >= 5) & (hy < 1295) & (hx >= 5) & (hx < 1345)).mean() < 0.70:
                    continue
                l1_err = 0.0
                for ci in avail_chs:
                    cm_s = mask_hw3[::ds, ::ds, ci][am_s] > 0.5
                    if not cm_s.any():
                        continue
                    atlas_v = map_coordinates(cal_base_hw[..., ci], [np.clip(hy[cm_s], 0, 1300), np.clip(hx[cm_s], 0, 1349)], order=1, mode="nearest")
                    obs_v = arr_hw3[::ds, ::ds, ci][am_s][cm_s]
                    l1_err += float(np.abs(atlas_v - np.clip(obs_v, 0, 0.35)).mean())
                sc = -l1_err
                if sc > best_corr:
                    best_corr, best_p = sc, p

    hy_fin, hx_fin = _eval_orbital_warp(best_p, yy_f - h * 0.5, xx_f - w * 0.5, w)
    w_fin = (map_coordinates(clean_land_hw, [hy_fin, hx_fin], order=1, mode="nearest") > 0.5).astype(np.float32)
    land_s, coast_e, signed_d = make_coast_features(w_fin, sigma_edge=2.0)
    return {
        "mode": "utc_track_scan_match",
        "hy": hy_fin.astype(np.float32),
        "hx": hx_fin.astype(np.float32),
        "land_bin": w_fin,
        "land_soft": land_s,
        "coast_edge": coast_e,
        "signed_dist": signed_d,
        "clear_agree_pct": 0.0,
    }


def project_meteor_to_atlas(
    arr_hw3: np.ndarray,
    mask_hw3: np.ndarray,
    hy: np.ndarray,
    hx: np.ndarray,
    atlas_shape: tuple[int, int] = (1301, 1351),
) -> tuple[np.ndarray, np.ndarray]:
    """晴天 METEOR 観測画素を極東アトラス座標空間へ逆投影し、投影画像と信頼度重みを返す。"""
    ah, aw = atlas_shape
    all_v = mask_hw3.min(axis=2) > 0.5
    clr = all_v & (arr_hw3[..., 0] < 0.25)
    acc = np.zeros((ah, aw, 3), dtype=np.float32)
    cnt = np.zeros((ah, aw), dtype=np.float32)
    ys_c, xs_c = np.where(clr)
    hy_c = np.rint(hy[ys_c, xs_c]).astype(np.int32)
    hx_c = np.rint(hx[ys_c, xs_c]).astype(np.int32)
    ok = (hy_c >= 0) & (hy_c < ah) & (hx_c >= 0) & (hx_c < aw)
    np.add.at(acc, (hy_c[ok], hx_c[ok]), arr_hw3[ys_c[ok], xs_c[ok]])
    np.add.at(cnt, (hy_c[ok], hx_c[ok]), 1.0)
    w_blur = gaussian_filter((cnt > 0).astype(np.float32), sigma=1.2)
    a_blur = gaussian_filter(acc, sigma=(1.2, 1.2, 0.0))
    cov_blur = gaussian_filter(cnt, sigma=1.2)[..., None]
    proj_img = np.where(cov_blur > 0.08, a_blur / np.maximum(cov_blur, 1e-4), 0.0)
    proj_w = np.clip(cov_blur[..., 0] * 1.5, 0.0, 0.55) * (w_blur > 0.15)
    return proj_img.astype(np.float32), proj_w.astype(np.float32)


def build_leave_one_out_prev_atlas(
    target_pass_id: str,
    cal_base_hw: np.ndarray,
    proj_pool: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[str]]:
    """ターゲット自身を除外した過去 METEOR 晴天実測と較正ベースを融合した前イメージアトラスを返す。"""
    ah, aw = cal_base_hw.shape[:2]
    acc_w = np.zeros((ah, aw, 1), dtype=np.float32)
    acc_v = np.zeros((ah, aw, 3), dtype=np.float32)
    used_sources: list[str] = []
    for src_pid, p_img, p_w in proj_pool:
        if src_pid != target_pass_id:
            acc_v += p_img * p_w[..., None]
            acc_w += p_w[..., None]
            used_sources.append(src_pid)
    valid_acc = acc_w > 0.05
    merged_prev = np.where(valid_acc, acc_v / np.maximum(acc_w, 1e-4), cal_base_hw)
    blend_alpha = np.clip(acc_w, 0.0, 0.50)
    atlas = np.clip((1.0 - blend_alpha) * cal_base_hw + blend_alpha * merged_prev, 0.0, 1.0).astype(np.float32)
    return atlas, used_sources


def _predict_tta(
    net: nn.Module,
    corr: torch.Tensor,
    omask: torch.Tensor,
    prior2: torch.Tensor,
    prev3: torch.Tensor | None = None,
    tta_4way: bool = False,
) -> torch.Tensor:
    n = corr.size(0)
    if prev3 is not None:
        if tta_4way:
            o0, o1, o2, o3 = net(
                torch.cat([corr, corr.flip(3), corr.flip(2), corr.flip((2, 3))], dim=0),
                torch.cat([omask, omask.flip(3), omask.flip(2), omask.flip((2, 3))], dim=0),
                torch.cat([prev3, prev3.flip(3), prev3.flip(2), prev3.flip((2, 3))], dim=0),
                torch.cat([prior2, prior2.flip(3), prior2.flip(2), prior2.flip((2, 3))], dim=0),
            ).split(n, dim=0)
            merged = 0.25 * (o0 + o1.flip(3) + o2.flip(2) + o3.flip((2, 3)))
        else:
            o0, o1 = net(
                torch.cat([corr, corr.flip(3)], dim=0),
                torch.cat([omask, omask.flip(3)], dim=0),
                torch.cat([prev3, prev3.flip(3)], dim=0),
                torch.cat([prior2, prior2.flip(3)], dim=0),
            ).split(n, dim=0)
            merged = 0.5 * (o0 + o1.flip(3))
    else:
        if tta_4way:
            o0, o1, o2, o3 = net(
                torch.cat([corr, corr.flip(3), corr.flip(2), corr.flip((2, 3))], dim=0),
                torch.cat([omask, omask.flip(3), omask.flip(2), omask.flip((2, 3))], dim=0),
                torch.cat([prior2, prior2.flip(3), prior2.flip(2), prior2.flip((2, 3))], dim=0),
            ).split(n, dim=0)
            merged = 0.25 * (o0 + o1.flip(3) + o2.flip(2) + o3.flip((2, 3)))
        else:
            o0, o1 = net(
                torch.cat([corr, corr.flip(3)], dim=0),
                torch.cat([omask, omask.flip(3)], dim=0),
                torch.cat([prior2, prior2.flip(3)], dim=0),
            ).split(n, dim=0)
            merged = 0.5 * (o0 + o1.flip(3))
    return corr * omask + merged * (1.0 - omask)


def _inpaint_tiled(
    net: nn.Module,
    img_hw3: np.ndarray,
    mask_hw3: np.ndarray,
    prior_hw2: np.ndarray,
    device: torch.device,
    prev_hw3: np.ndarray | None = None,
    target_2d: np.ndarray | None = None,
    patch_size: int = 64,
    stride: int = 24,
    batch_size: int = 64,
    tta_4way: bool = False,
) -> np.ndarray:
    h, w, c = img_hw3.shape
    ys = list(range(0, max(h - patch_size + 1, 1), stride))
    if not ys or ys[-1] != h - patch_size:
        ys.append(max(h - patch_size, 0))
    xs = list(range(0, max(w - patch_size + 1, 1), stride))
    if not xs or xs[-1] != w - patch_size:
        xs.append(max(w - patch_size, 0))
    coords = [(y, x) for y in ys for x in xs if target_2d is None or target_2d[y:y + patch_size, x:x + patch_size].any()]
    if not coords:
        return img_hw3.copy()

    win_1d = np.hanning(patch_size + 4)[2:-2].astype(np.float32)
    win_2d = np.maximum(np.outer(win_1d, win_1d), 0.05)
    acc, wsum = np.zeros((h, w, c), dtype=np.float32), np.zeros((h, w, 1), dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            bc = coords[i:i + batch_size]
            bi = torch.from_numpy(np.stack([img_hw3[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1) for y, x in bc])).to(device)
            bm = torch.from_numpy(np.stack([mask_hw3[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1) for y, x in bc])).to(device)
            bp = torch.from_numpy(np.stack([prior_hw2[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1) for y, x in bc])).to(device)
            bv = (
                torch.from_numpy(np.stack([prev_hw3[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1) for y, x in bc])).to(device)
                if prev_hw3 is not None
                else None
            )
            pred = _predict_tta(net, bi, bm, bp, prev3=bv, tta_4way=tta_4way).cpu().numpy()
            for idx, (y, x) in enumerate(bc):
                acc[y:y + patch_size, x:x + patch_size] += pred[idx].transpose(1, 2, 0) * win_2d[..., None]
                wsum[y:y + patch_size, x:x + patch_size] += win_2d[..., None]
    return img_hw3 * mask_hw3 + np.clip(acc / np.maximum(wsum, 1e-6), 0.0, 1.0) * (1.0 - mask_hw3)


def _inpaint_wide_2x_8ch(
    net: nn.Module,
    img_hw3: np.ndarray,
    mask_hw3: np.ndarray,
    prior_hw2: np.ndarray,
    target_2d: np.ndarray,
    device: torch.device,
    stride: int = 24,
) -> np.ndarray:
    h, w, _ = img_hw3.shape
    if h < 128 or w < 128:
        return img_hw3.copy()
    t_img = torch.from_numpy(img_hw3.transpose(2, 0, 1)).unsqueeze(0)
    t_msk = torch.from_numpy(mask_hw3.transpose(2, 0, 1)).unsqueeze(0)
    den = F.avg_pool2d(t_msk, 2)
    num = F.avg_pool2d(t_img * t_msk, 2)
    m2 = (den >= 0.25).float()
    c2 = torch.where(m2 > 0.5, num / den.clamp_min(1e-6), torch.zeros_like(num))
    pr2 = (
        torch.cat(
            [
                F.avg_pool2d(torch.from_numpy(prior_hw2[..., 0])[None, None], 2),
                F.max_pool2d(torch.from_numpy(prior_hw2[..., 1])[None, None], 2),
            ],
            dim=1,
        )
        .squeeze(0)
        .numpy()
        .transpose(1, 2, 0)
    )
    t_sub = F.max_pool2d(torch.from_numpy(target_2d.astype(np.float32))[None, None], 2).squeeze().numpy() > 0.5
    pred2 = _inpaint_tiled(
        net,
        c2.squeeze(0).numpy().transpose(1, 2, 0),
        m2.squeeze(0).numpy().transpose(1, 2, 0),
        pr2,
        device=device,
        target_2d=t_sub,
        patch_size=64,
        stride=stride,
        batch_size=64,
        tta_4way=False,
    )
    up = F.interpolate(torch.from_numpy(pred2.transpose(2, 0, 1)).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
    return np.clip(up.squeeze(0).numpy().transpose(1, 2, 0), 0.0, 1.0)


def _infuse_contextual_microtexture(
    img_hw3: np.ndarray,
    orig_valid_2d: np.ndarray,
    prior_hw2: np.ndarray,
    dist_from_real: np.ndarray,
) -> np.ndarray:
    h, w, _ = img_hw3.shape
    if (~orig_valid_2d).sum() < 500 or h < 48:
        return img_hw3
    low_src = gaussian_filter(img_hw3, sigma=(1.35, 1.35, 0.0))
    hf_src = np.clip(img_hw3 - low_src, -0.095, 0.095).astype(np.float32)
    valid_interior = binary_erosion(orig_valid_2d, iterations=2)

    tile = 16
    nh, nw = h // tile, w // tile
    src_blocks, src_feats = [], []
    for by in range(nh):
        y0, y1 = by * tile, (by + 1) * tile
        for bx in range(nw):
            x0, x1 = bx * tile, (bx + 1) * tile
            if valid_interior[y0:y1, x0:x1].mean() >= 0.85:
                blk_img, blk_hf = img_hw3[y0:y1, x0:x1], hf_src[y0:y1, x0:x1]
                if blk_hf[..., :2].std() < 0.004:
                    continue
                lp_m = float(prior_hw2[y0:y1, x0:x1, 0].mean())
                c1_m, c2_m, c4_m = float(blk_img[..., 0].mean()), float(blk_img[..., 1].mean()), float(blk_img[..., 2].mean())
                edge_m = float(prior_hw2[y0:y1, x0:x1, 1].mean())
                src_blocks.append(blk_hf)
                src_feats.append([lp_m * 1.6, c1_m * 1.4, c2_m * 1.4, c4_m * 1.0, (c2_m - c1_m) * 3.0, -edge_m * 0.8, (y0 / h) * 0.28, (x0 / w) * 0.28])

    if len(src_blocks) < 8:
        return img_hw3

    src_blocks_arr = np.stack(src_blocks, axis=0)
    hole_2d = ~orig_valid_2d
    q_coords, q_feats = [], []
    for y0 in range(0, h - tile + 1, 8):
        for x0 in range(0, w - tile + 1, 8):
            if not hole_2d[y0:y0 + tile, x0:x0 + tile].any():
                continue
            blk = img_hw3[y0:y0 + tile, x0:x0 + tile]
            lp_m = float(prior_hw2[y0:y0 + tile, x0:x0 + tile, 0].mean())
            c1_m, c2_m, c4_m = float(blk[..., 0].mean()), float(blk[..., 1].mean()), float(blk[..., 2].mean())
            edge_m = float(prior_hw2[y0:y0 + tile, x0:x0 + tile, 1].mean())
            q_coords.append((y0, x0))
            q_feats.append([lp_m * 1.6, c1_m * 1.4, c2_m * 1.4, c4_m * 1.0, (c2_m - c1_m) * 3.0, -edge_m * 0.8, (y0 / h) * 0.28, (x0 / w) * 0.28])

    if not q_coords:
        return img_hw3

    t_sf = torch.from_numpy(np.array(src_feats, dtype=np.float32))
    t_qf = torch.from_numpy(np.array(q_feats, dtype=np.float32))
    _, topk_idx = torch.topk(torch.cdist(t_qf, t_sf), k=min(4, len(src_blocks)), dim=1, largest=False)
    topk_np = topk_idx.numpy()

    win_1d = np.hanning(tile + 2)[1:-1].astype(np.float32)
    win_2d = np.outer(win_1d, win_1d)[..., None]
    tex_acc, tex_wsum = np.zeros_like(img_hw3), np.zeros((h, w, 1), dtype=np.float32)
    rng = np.random.default_rng(2026)
    for qi, (y0, x0) in enumerate(q_coords):
        pick = src_blocks_arr[topk_np[qi, int(rng.integers(0, topk_np.shape[1]))]]
        if rng.random() > 0.5:
            pick = pick[:, ::-1]
        if rng.random() > 0.5:
            pick = pick[::-1, :]
        tex_acc[y0:y0 + tile, x0:x0 + tile] += pick * win_2d
        tex_wsum[y0:y0 + tile, x0:x0 + tile] += win_2d

    tex_field = tex_acc / np.maximum(tex_wsum, 1e-6)
    depth_gate = np.clip((dist_from_real - 2.0) / 14.0, 0.0, 1.0)
    coast_protect = 1.0 - 0.55 * prior_hw2[..., 1]
    infuse_w = (depth_gate * coast_protect * (~orig_valid_2d).astype(np.float32))[..., None] * 0.72
    return np.clip(img_hw3 + infuse_w * tex_field, 0.0, 1.0).astype(np.float32)


def restore_single_pass_8ch(
    net_8ch: CoastlineGuidedSharpResSEUNet,
    arr_hw3: np.ndarray,
    mask_hw3: np.ndarray,
    prior_hw2: np.ndarray,
    signed_dist: np.ndarray,
    device: torch.device,
    has_geo_prior: bool = True,
    high_quality: bool = False,
) -> np.ndarray:
    """8ch Single-Pass モデルによる階層的欠損補完と微細質感再構成。"""
    h0 = arr_hw3.shape[0]
    pad_top, pad_bot = 0, 0
    if h0 < 64:
        pad_top = (64 - h0) // 2
        pad_bot = 64 - h0 - pad_top
        a_p = np.pad(arr_hw3, ((pad_top, pad_bot), (0, 0), (0, 0)), mode="edge")
        m_p = np.pad(mask_hw3, ((pad_top, pad_bot), (0, 0), (0, 0)), mode="edge")
        pr_p = np.pad(prior_hw2, ((pad_top, pad_bot), (0, 0), (0, 0)), mode="edge")
        sd_p = np.pad(signed_dist, ((pad_top, pad_bot), (0, 0)), mode="edge")
    else:
        a_p, m_p, pr_p, sd_p = arr_hw3, mask_hw3, prior_hw2, signed_dist

    hp = a_p.shape[0]
    any_valid = m_p.max(axis=2) > 0.5
    all_valid = m_p.min(axis=2) > 0.5
    p1_stride = 22 if high_quality else 28
    j_stride = 16 if high_quality else 26

    rescue_target = any_valid & (~all_valid)
    step0_pred = _inpaint_tiled(net_8ch, a_p, m_p, pr_p, device=device, target_2d=rescue_target, stride=24, tta_4way=high_quality)
    step0_img = np.where(any_valid[..., None], a_p * m_p + step0_pred * (1.0 - m_p), 0.0).astype(np.float32)

    if (~any_valid).sum() == 0:
        out_raw = step0_img[pad_top:pad_top + h0] if pad_top or pad_bot else step0_img
        return np.clip(out_raw, 0.0, 1.0).astype(np.float32)

    cur_img, cur_val = step0_img.copy(), any_valid.copy()
    cur_msk = np.repeat(cur_val[..., None].astype(np.float32), 3, axis=2)
    dist_from_real = distance_transform_edt(~any_valid)
    n_steps = int(np.clip(np.ceil(float(dist_from_real.max()) / 11.0), 1, 7))
    s_fine = np.ones((2 * 11 + 1, 1), dtype=bool)
    for st in range(1, n_steps + 1):
        nxt = binary_dilation(cur_val, structure=s_fine)
        if st == n_steps:
            nxt[:] = True
        fr = nxt & (~cur_val)
        if fr.any():
            p_fine = _inpaint_tiled(net_8ch, cur_img, cur_msk, pr_p, device=device, target_2d=fr, stride=p1_stride, tta_4way=(high_quality and st == n_steps))
            if hp >= 128:
                p_wide = _inpaint_wide_2x_8ch(net_8ch, cur_img, cur_msk, pr_p, target_2d=fr, device=device, stride=24)
                low_f = gaussian_filter(p_fine, sigma=(2.5, 2.5, 0.0))
                low_w = gaussian_filter(p_wide, sigma=(2.5, 2.5, 0.0))
                fused = np.clip(0.84 * low_f + 0.16 * low_w + (p_fine - low_f), 0.0, 1.0)
            else:
                fused = p_fine
            cur_img = np.where(fr[..., None], fused, cur_img).astype(np.float32)
            cur_val = nxt
            cur_msk = np.repeat(cur_val[..., None].astype(np.float32), 3, axis=2)

    core_mask = dist_from_real > 20.0
    if core_mask.any():
        anchor_mask = np.repeat((~core_mask)[..., None].astype(np.float32), 3, axis=2)
        joint_pred = _inpaint_tiled(net_8ch, cur_img * anchor_mask, anchor_mask, pr_p, device=device, target_2d=core_mask, stride=j_stride, tta_4way=high_quality)
        w_core = gaussian_filter(core_mask.astype(np.float32), sigma=(3.5, 1.0))[..., None]
        cur_img = np.where((~any_valid)[..., None], w_core * joint_pred + (1.0 - w_core) * cur_img, step0_img).astype(np.float32)

    hole_inner = np.clip(gaussian_filter((~any_valid).astype(np.float32), sigma=2.0) * 2.0 - 1.0, 0.0, 1.0) * (~any_valid).astype(np.float32)
    hole_feather = hole_inner[..., None]
    v_high = cur_img - gaussian_filter(cur_img, sigma=(2.2, 0.0, 0.0))
    destreaked = np.clip(cur_img - hole_feather * 0.75 * gaussian_filter(v_high, sigma=(0.0, 20.0, 0.0)), 0.0, 1.0)
    detail_x = np.clip(destreaked - gaussian_filter(destreaked, sigma=(0.7, 2.2, 0.0)), -0.08, 0.08)
    sharpened = np.clip(destreaked + hole_feather * 0.65 * detail_x, 0.0, 1.0)

    if has_geo_prior:
        local_vis = gaussian_filter(sharpened[..., 0], sigma=1.5)
        clear_gate = np.clip(1.0 - (local_vis - 0.17) / 0.12, 0.0, 1.0)
        coastal_band = np.exp(-(sd_p**2) / (2.0 * (4.5**2)))
        land_reg = (sd_p > 0.0).astype(np.float32)
        sea_reg = 1.0 - land_reg
        w_land_zone = hole_inner * coastal_band * clear_gate
        mean_land = gaussian_filter(sharpened * land_reg[..., None], sigma=(5.0, 5.0, 0.0)) / np.maximum(gaussian_filter(land_reg, sigma=5.0)[..., None], 1e-4)
        mean_sea = gaussian_filter(sharpened * sea_reg[..., None], sigma=(5.0, 5.0, 0.0)) / np.maximum(gaussian_filter(sea_reg, sigma=5.0)[..., None], 1e-4)
        sharp_target = np.where(land_reg[..., None] > 0.5, mean_land, mean_sea)
        guided_shift = np.clip(sharp_target - gaussian_filter(sharpened, sigma=(3.5, 3.5, 0.0)), -0.075, 0.075)
        sharpened = np.clip(sharpened + (w_land_zone[..., None] * 0.55) * guided_shift, 0.0, 1.0).astype(np.float32)

    final_p = _infuse_contextual_microtexture(sharpened, any_valid, pr_p, dist_from_real)
    out = final_p[pad_top:pad_top + h0] if pad_top or pad_bot else final_p
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def restore_temporal_prev_11ch(
    net_11ch: PrevImageGuidedResSEUNet,
    arr_hw3: np.ndarray,
    mask_hw3: np.ndarray,
    prior_hw2: np.ndarray,
    prev_hw3: np.ndarray,
    device: torch.device,
    fast_mode: bool = True,
) -> np.ndarray:
    """11ch 時系列前イメージガイドモデルによる階層的欠損補完。"""
    h0 = arr_hw3.shape[0]
    pad_t, pad_b = 0, 0
    if h0 < 64:
        pad_t = (64 - h0) // 2
        pad_b = 64 - h0 - pad_t
        ap = np.pad(arr_hw3, ((pad_t, pad_b), (0, 0), (0, 0)), mode="edge")
        mp = np.pad(mask_hw3, ((pad_t, pad_b), (0, 0), (0, 0)), mode="edge")
        prp = np.pad(prior_hw2, ((pad_t, pad_b), (0, 0), (0, 0)), mode="edge")
        pvp = np.pad(prev_hw3, ((pad_t, pad_b), (0, 0), (0, 0)), mode="edge")
    else:
        ap, mp, prp, pvp = arr_hw3, mask_hw3, prior_hw2, prev_hw3

    any_v = mp.max(axis=2) > 0.5
    all_v = mp.min(axis=2) > 0.5

    step0_pred = _inpaint_tiled(net_11ch, ap, mp, prp, device=device, prev_hw3=pvp, target_2d=any_v & (~all_v), stride=24, tta_4way=not fast_mode)
    step0_img = np.where(any_v[..., None], ap * mp + step0_pred * (1.0 - mp), 0.0).astype(np.float32)

    if (~any_v).sum() == 0:
        out = step0_img[pad_t:pad_t + h0] if pad_t or pad_b else step0_img
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    cur_img, cur_val = step0_img.copy(), any_v.copy()
    cur_msk = np.repeat(cur_val[..., None].astype(np.float32), 3, axis=2)

    dist_r = distance_transform_edt(~any_v)
    n_steps = int(np.clip(np.ceil(float(dist_r.max()) / 12.0), 1, 7))
    s_fine = np.ones((2 * 12 + 1, 1), dtype=bool)
    p1_stride = 28 if fast_mode else 22
    for st in range(1, n_steps + 1):
        nxt = binary_dilation(cur_val, structure=s_fine)
        if st == n_steps:
            nxt[:] = True
        fr = nxt & (~cur_val)
        if fr.any():
            pf = _inpaint_tiled(
                net_11ch,
                cur_img,
                cur_msk,
                prp,
                device=device,
                prev_hw3=pvp,
                target_2d=fr,
                stride=p1_stride,
                tta_4way=(not fast_mode and st == n_steps),
            )
            cur_img = np.where(fr[..., None], pf, cur_img).astype(np.float32)
            cur_val = nxt
            cur_msk = np.repeat(cur_val[..., None].astype(np.float32), 3, axis=2)

    core = dist_r > 24.0
    if core.any():
        am = np.repeat((~core)[..., None].astype(np.float32), 3, axis=2)
        jp = _inpaint_tiled(
            net_11ch,
            cur_img * am,
            am,
            prp,
            device=device,
            prev_hw3=pvp,
            target_2d=core,
            stride=26 if fast_mode else 18,
            tta_4way=not fast_mode,
        )
        wc = gaussian_filter(core.astype(np.float32), sigma=(3.5, 1.0))[..., None]
        cur_img = np.where((~any_v)[..., None], wc * jp + (1.0 - wc) * cur_img, step0_img).astype(np.float32)

    hole_inner = np.clip(gaussian_filter((~any_v).astype(np.float32), sigma=2.0) * 2.0 - 1.0, 0.0, 1.0) * (~any_v).astype(np.float32)
    vh = cur_img - gaussian_filter(cur_img, sigma=(2.2, 0.0, 0.0))
    destr = np.clip(cur_img - hole_inner[..., None] * 0.70 * gaussian_filter(vh, sigma=(0.0, 20.0, 0.0)), 0.0, 1.0)
    out = destr[pad_t:pad_t + h0] if pad_t or pad_b else destr
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_seam_jump_l1(restored_hw3: np.ndarray, mask_hw3: np.ndarray) -> float:
    """実測走査線と補完領域の境界における垂直方向の輝度段差（Jump L1）を計測する。"""
    any_v = mask_hw3.max(axis=2) > 0.5
    row_v = any_v.mean(axis=1) > 0.25
    trans_idx = [y for y in range(restored_hw3.shape[0] - 1) if row_v[y] != row_v[y + 1]]
    if not trans_idx:
        return 0.0
    return round(float(np.mean([np.abs(restored_hw3[y + 1] - restored_hw3[y]).mean() for y in trans_idx])), 4)


class MeteorInpaintingEngine:
    """nb09 (8ch) および nb11 (11ch) バンドルを統合し、任意パスの補完推論を実行するエンジン。"""

    def __init__(
        self,
        model_dir: Path,
        data_dir: Path,
        device: torch.device | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = device or select_torch_device()

        self.nb09_path = self.model_dir / "nb09_coastline_patchgan_bundle.pt"
        self.nb11_path = self.model_dir / "nb11_temporal_prev_bundle.pt"
        if not self.nb09_path.exists():
            raise FileNotFoundError(f"nb09 バンドルが見つかりません: {self.nb09_path}")
        if not self.nb11_path.exists():
            raise FileNotFoundError(f"nb11 バンドルが見つかりません: {self.nb11_path}")

        bundle09 = torch.load(self.nb09_path, map_location="cpu", weights_only=False)
        bundle11 = torch.load(self.nb11_path, map_location="cpu", weights_only=False)

        self.nb09_hash = str(bundle09.get("config_hash", ""))
        self.nb11_hash = str(bundle11.get("config_hash", ""))

        self.clean_land_hw = bundle09["clean_land_hw"].astype(np.float32)
        self.hw_smooth = gaussian_filter(self.clean_land_hw, sigma=1.2).astype(np.float32)
        self.W_av221 = bundle09["W_av221"].astype(np.float32)
        self.W_msu124 = bundle09["W_msu124"].astype(np.float32) if bundle09.get("W_msu124") is not None else None

        self.single_net = CoastlineGuidedSharpResSEUNet(base=48).to(self.device)
        self.single_net.load_state_dict(bundle09["state_dict"], strict=True)
        self.single_net.eval()

        self.prev_net = PrevImageGuidedResSEUNet(base=48).to(self.device)
        self.prev_net.load_state_dict(bundle11["state_dict"], strict=True)
        self.prev_net.eval()

        self.cal_base_hw = build_or_load_cal_base_atlas(
            clean_land_hw=self.clean_land_hw,
            cache_path=self.model_dir / "east_asia_cal_base_atlas.npz",
            himawari_npz_path=self.data_dir / "himawari_multiband_cache.npz",
            meteor_dir=self.data_dir / "meteor",
        )
        self._proj_pool: list[tuple[str, np.ndarray, np.ndarray]] | None = None

    def get_expected_hashes(self) -> dict[str, str]:
        """モデルモード別の期待 config_hash マップを返す。"""
        return {
            "temporal_prev_11ch": self.nb11_hash,
            "single_pass_8ch": self.nb09_hash,
        }

    def _ensure_proj_pool(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """基準となる過去 METEOR 長尺パスの逆投影アトラスプールを構築（キャッシュ）する。"""
        if self._proj_pool is not None:
            return self._proj_pool

        meteor_dir = self.data_dir / "meteor"
        pool: list[tuple[str, np.ndarray, np.ndarray]] = []
        if not meteor_dir.exists():
            self._proj_pool = pool
            return pool

        candidates = sorted(
            [
                d
                for d in meteor_dir.iterdir()
                if d.is_dir()
                and "GOLDEN" not in d.name.upper()
                and (d / "MSU-MR-1.png").exists()
                and (d / "MSU-MR-2.png").exists()
                and (d / "MSU-MR-4.png").exists()
            ],
            key=lambda p: p.name,
            reverse=True,
        )
        for pdir in candidates:
            if len(pool) >= 6:
                break
            try:
                arr, msk, _ = load_meteor_pass_bands(pdir)
            except FileNotFoundError:
                continue
            if arr.shape[0] < 120:
                continue
            if _is_night_ir_only_pass(arr, msk):
                continue
            reg = register_pass_to_east_asia(
                pdir.name,
                arr,
                msk,
                self.clean_land_hw,
                self.hw_smooth,
                self.cal_base_hw,
                is_golden=False,
            )
            if reg["mode"] == "clear_orbital_tps" and reg["clear_agree_pct"] >= 85.0 and reg["hy"] is not None:
                p_img, p_w = project_meteor_to_atlas(arr, msk, reg["hy"], reg["hx"], self.cal_base_hw.shape[:2])
                pool.append((pdir.name, p_img, p_w))

        self._proj_pool = pool
        return pool

    def inpaint_pass(
        self,
        pass_id: str,
        satellite: str,
        pass_dir: Path,
        mode: str = "auto",
        quality: str = "standard",
        is_golden: bool = False,
    ) -> dict[str, Any]:
        """1 パス分の補完推論を実行し、出力画像配列4枚とメタデータ辞書を返す。"""
        t0 = time.perf_counter()
        arr_hw3, mask_hw3, avail = load_meteor_pass_bands(pass_dir)
        input_stats = compute_input_stats(mask_hw3, avail)
        tier, tier_label = classify_pass_tier(input_stats)

        reg = register_pass_to_east_asia(
            pass_id=pass_id,
            arr_hw3=arr_hw3,
            mask_hw3=mask_hw3,
            clean_land_hw=self.clean_land_hw,
            hw_smooth=self.hw_smooth,
            cal_base_hw=self.cal_base_hw,
            is_golden=is_golden,
        )
        prior_hw2 = np.stack([reg["land_soft"], reg["coast_edge"]], axis=-1).astype(np.float32)

        can_use_temporal = reg["mode"] in ("clear_orbital_tps", "utc_track_scan_match") and reg["hy"] is not None
        if mode == "temporal" and not can_use_temporal:
            raise ValueError(f"パス {pass_id} は東アジア軌道登録不可 ({reg['mode']}) のため temporal モードを実行できません")

        use_temporal = (mode == "temporal") or (mode == "auto" and can_use_temporal)
        fast_mode = quality != "high"

        prev_sources: list[str] = []
        if use_temporal:
            proj_pool = self._ensure_proj_pool()
            atlas, prev_sources = build_leave_one_out_prev_atlas(pass_id, self.cal_base_hw, proj_pool)
            prev_3ch = np.stack(
                [map_coordinates(atlas[..., ci], [reg["hy"], reg["hx"]], order=1, mode="nearest") for ci in range(3)],
                axis=-1,
            ).astype(np.float32)
            restored = restore_temporal_prev_11ch(
                self.prev_net,
                arr_hw3,
                mask_hw3,
                prior_hw2,
                prev_3ch,
                device=self.device,
                fast_mode=fast_mode,
            )
            model_meta = {
                "name": "PrevImageGuidedResSEUNet",
                "mode": "temporal_prev_11ch",
                "quality": quality,
                "bundle_file": self.nb11_path.name,
                "config_hash": self.nb11_hash,
                "base_bundle_hash": self.nb09_hash,
            }
        else:
            has_geo = reg["mode"] == "clear_orbital_tps"
            restored = restore_single_pass_8ch(
                self.single_net,
                arr_hw3,
                mask_hw3,
                prior_hw2,
                reg["signed_dist"],
                device=self.device,
                has_geo_prior=has_geo,
                high_quality=(not fast_mode),
            )
            model_meta = {
                "name": "CoastlineGuidedSharpResSEUNet",
                "mode": "single_pass_8ch",
                "quality": quality,
                "bundle_file": self.nb09_path.name,
                "config_hash": self.nb09_hash,
                "base_bundle_hash": self.nb09_hash,
            }

        if self.device.type == "mps":
            torch.mps.synchronize()
        elapsed = round(time.perf_counter() - t0, 2)

        any_mask_hw1 = (mask_hw3.max(axis=2) > 0.5)[..., None].astype(np.float32)
        raw_avhrr_221 = render_satdump_av221(arr_hw3, self.W_av221, valid_mask_hw1=any_mask_hw1)
        avhrr_221 = render_satdump_av221(restored, self.W_av221)
        msu_mr_124 = render_satdump_msu124(restored, self.W_msu124)
        mask_overlay = render_inpaint_mask_overlay(avhrr_221, mask_hw3)
        seam_jump = compute_seam_jump_l1(restored, mask_hw3)

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "pass_id": pass_id,
            "satellite": satellite,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device": self.device.type,
            "model": model_meta,
            "input_stats": input_stats,
            "registration": {
                "mode": reg["mode"],
                "clear_agree_pct": reg["clear_agree_pct"],
                "used_prev_atlas": bool(use_temporal),
                "prev_source_passes": prev_sources,
            },
            "quality": {
                "tier": tier,
                "tier_label": tier_label,
                "seam_jump_l1": seam_jump,
                "processing_sec": elapsed,
                "note": "欠損領域は過去観測・較正地表アトラスと当日の生存走査線から推定された補完値です。",
            },
            "files": {
                "avhrr_221": "ml_inpainted/avhrr_221_inpainted.png",
                "msu_mr_124": "ml_inpainted/msu_mr_124_inpainted.png",
                "raw_avhrr_221": "ml_inpainted/raw_avhrr_221_aligned.png",
                "mask_overlay": "ml_inpainted/inpaint_mask_overlay.png",
            },
        }

        return {
            "images": {
                "avhrr_221_inpainted.png": avhrr_221,
                "msu_mr_124_inpainted.png": msu_mr_124,
                "raw_avhrr_221_aligned.png": raw_avhrr_221,
                "inpaint_mask_overlay.png": mask_overlay,
            },
            "metadata": metadata,
        }
