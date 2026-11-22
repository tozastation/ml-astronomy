"""METEOR-M2 補完パイプラインおよび Garage S3 クライアントの単体テスト。"""

from __future__ import annotations

import io
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from inpainting import (  # noqa: E402
    CoastlineGuidedSharpResSEUNet,
    PrevImageGuidedResSEUNet,
    compute_input_stats,
    render_inpaint_mask_overlay,
)
from run_inpaint_pipeline import evaluate_pass_inpaint_status  # noqa: E402
from s3_loader import GarageS3Client  # noqa: E402


class _DummyResponse:
    def __init__(self, body: bytes = b"") -> None:
        self._buf = io.BytesIO(body)

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> _DummyResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class TestGarageS3ClientWrite(unittest.TestCase):
    def setUp(self) -> None:
        self.client = GarageS3Client(
            endpoint_url="http://192.168.68.66:30900",
            bucket="satellite-recordings",
            access_key="AKTEST",
            secret_key="SKTEST",
        )

    def test_put_bytes_sigv4_and_space_encoding(self) -> None:
        captured: list = []

        def fake_urlopen(req, timeout=15):
            captured.append(req)
            return _DummyResponse(b"")

        key = "results/METEOR-M2 3/METEOR-M2 3_20261026_003009/ml_inpainted/avhrr_221_inpainted.png"
        payload = b"\x89PNG\r\n\x1a\n"
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.client.put_bytes(key, payload, content_type="image/png")

        self.assertEqual(len(captured), 1)
        req = captured[0]
        self.assertEqual(req.get_method(), "PUT")
        self.assertIn("METEOR-M2%203", req.full_url)
        self.assertEqual(req.data, payload)
        self.assertEqual(req.get_header("Content-type"), "image/png")
        self.assertEqual(req.get_header("Content-length"), str(len(payload)))
        self.assertIn("AWS4-HMAC-SHA256", req.get_header("Authorization"))
        self.assertEqual(len(req.get_header("X-amz-content-sha256")), 64)

    def test_object_exists_head_200_and_404(self) -> None:
        def ok_urlopen(req, timeout=15):
            self.assertEqual(req.get_method(), "HEAD")
            return _DummyResponse(b"")

        with patch("urllib.request.urlopen", side_effect=ok_urlopen):
            self.assertTrue(self.client.object_exists("results/METEOR-M2 3/p1/ml_inpainted/metadata.json"))

        err_404 = urllib.error.HTTPError(
            url="http://192.168.68.66:30900/satellite-recordings/missing",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=err_404):
            self.assertFalse(self.client.object_exists("results/METEOR-M2 3/p1/ml_inpainted/metadata.json"))
            self.assertIsNone(self.client.get_json_if_exists("results/METEOR-M2 3/p1/ml_inpainted/metadata.json"))


class TestInpaintingModelsAndHelpers(unittest.TestCase):
    def test_bundle_loading_and_forward_pass(self) -> None:
        models_dir = PROJECT_DIR / "models"
        nb09_path = models_dir / "nb09_coastline_patchgan_bundle.pt"
        nb11_path = models_dir / "nb11_temporal_prev_bundle.pt"
        self.assertTrue(nb09_path.exists(), f"Missing {nb09_path}")
        self.assertTrue(nb11_path.exists(), f"Missing {nb11_path}")

        b09 = torch.load(nb09_path, map_location="cpu", weights_only=False)
        b11 = torch.load(nb11_path, map_location="cpu", weights_only=False)

        net8 = CoastlineGuidedSharpResSEUNet(base=48)
        net8.load_state_dict(b09["state_dict"], strict=True)
        net8.eval()

        net11 = PrevImageGuidedResSEUNet(base=48)
        net11.load_state_dict(b11["state_dict"], strict=True)
        net11.eval()

        with torch.no_grad():
            corr = torch.rand(1, 3, 64, 64)
            om = torch.ones(1, 3, 64, 64)
            om[:, :, 20:30, :] = 0.0
            prior2 = torch.rand(1, 2, 64, 64)
            prev3 = torch.rand(1, 3, 64, 64)

            out8 = net8(corr * om, om, prior2)
            out11 = net11(corr * om, om, prev3, prior2)

        self.assertEqual(tuple(out8.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(out11.shape), (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(out8).all())
        self.assertTrue(torch.isfinite(out11).all())

    def test_compute_input_stats_and_mask_overlay(self) -> None:
        mask = np.ones((20, 10, 3), dtype=np.float32)
        mask[5:11, :, :] = 0.0  # 6行全欠損
        mask[12:14, :, 1] = 0.0  # 2行部分欠損
        stats = compute_input_stats(mask, "124")
        self.assertEqual(stats["height"], 20)
        self.assertEqual(stats["width"], 10)
        self.assertEqual(stats["max_gap_px"], 6)
        self.assertAlmostEqual(stats["all_valid_pct"], 60.0)
        self.assertAlmostEqual(stats["partial_valid_pct"], 10.0)
        self.assertAlmostEqual(stats["missing_pct"], 30.0)
        self.assertAlmostEqual(stats["any_valid_pct"], 70.0)

        base_img = np.full((20, 10, 3), 0.5, dtype=np.float32)
        overlay = render_inpaint_mask_overlay(base_img, mask)
        self.assertEqual(overlay.shape, (20, 10, 3))
        # 完全生存域はそのまま、欠損域はオレンジ方向に変化
        self.assertTrue(np.allclose(overlay[0, 0], [0.5, 0.5, 0.5]))
        self.assertGreater(overlay[8, 5, 0], overlay[8, 5, 2])

    def test_evaluate_pass_inpaint_status_skip_logic(self) -> None:
        files_full = {
            "ml_inpainted/avhrr_221_inpainted.png": {"key": "k1"},
            "ml_inpainted/msu_mr_124_inpainted.png": {"key": "k2"},
            "ml_inpainted/raw_avhrr_221_aligned.png": {"key": "k3"},
            "ml_inpainted/inpaint_mask_overlay.png": {"key": "k4"},
            "ml_inpainted/metadata.json": {"key": "k5"},
        }
        p_info = {"pass_id": "METEOR-M2 3_20261026_003009", "files": files_full}
        hashes = {"temporal_prev_11ch": "a84ae13587fa7d7a", "single_pass_8ch": "46546fee33518457"}

        meta_ok = {
            "schema_version": 1,
            "model": {"mode": "temporal_prev_11ch", "config_hash": "a84ae13587fa7d7a", "quality": "standard"},
        }
        label, skip = evaluate_pass_inpaint_status(p_info, meta_ok, hashes, "auto", "standard")
        self.assertTrue(skip)
        self.assertIn("補完済", label)

        # ハッシュ不一致なら再補完要
        meta_old = {
            "schema_version": 1,
            "model": {"mode": "temporal_prev_11ch", "config_hash": "oldhash", "quality": "standard"},
        }
        label2, skip2 = evaluate_pass_inpaint_status(p_info, meta_old, hashes, "auto", "standard")
        self.assertFalse(skip2)
        self.assertIn("モデル更新", label2)

        # 品質変更なら再補完要
        _, skip3 = evaluate_pass_inpaint_status(p_info, meta_ok, hashes, "auto", "high")
        self.assertFalse(skip3)

        # PNG が欠けていたらスキップしない
        p_missing_png = {"pass_id": "p", "files": {"ml_inpainted/metadata.json": {"key": "k5"}}}
        _, skip4 = evaluate_pass_inpaint_status(p_missing_png, meta_ok, hashes, "auto", "standard")
        self.assertFalse(skip4)


if __name__ == "__main__":
    unittest.main()
