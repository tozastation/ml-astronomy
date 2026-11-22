#!/usr/bin/env python3
"""Mac から手動実行し、METEOR-M2 画像の欠損補完と Garage S3 転送を行うパイプライン CLI。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inpainting import MeteorInpaintingEngine, SCHEMA_VERSION, save_rgb_png  # noqa: E402
from s3_loader import (  # noqa: E402
    DEFAULT_BUCKET,
    DEFAULT_ENDPOINT,
    DEFAULT_SSH_HOST,
    GarageS3Client,
    build_client,
    download_pass_bands,
    sync_meteor_dataset,
)

INPAINT_PNG_NAMES = (
    "avhrr_221_inpainted.png",
    "msu_mr_124_inpainted.png",
    "raw_avhrr_221_aligned.png",
    "inpaint_mask_overlay.png",
)
INPAINT_META_REL = "ml_inpainted/metadata.json"


def load_bundle_hashes(model_dir: Path) -> dict[str, str]:
    """モデル重みバンドルから config_hash のみを軽量に読み出す。"""
    nb09 = torch.load(Path(model_dir) / "nb09_coastline_patchgan_bundle.pt", map_location="cpu", weights_only=False)
    nb11 = torch.load(Path(model_dir) / "nb11_temporal_prev_bundle.pt", map_location="cpu", weights_only=False)
    return {
        "single_pass_8ch": str(nb09.get("config_hash", "")),
        "temporal_prev_11ch": str(nb11.get("config_hash", "")),
    }


def evaluate_pass_inpaint_status(
    pass_info: dict[str, Any],
    existing_meta: dict[str, Any] | None,
    expected_hashes: dict[str, str],
    requested_mode: str,
    requested_quality: str,
) -> tuple[str, bool]:
    """パスの S3 上補完ステータス文字列とスキップ可否 (should_skip) を判定する。"""
    files = pass_info.get("files") or {}
    has_all_pngs = all(f"ml_inpainted/{name}" in files for name in INPAINT_PNG_NAMES)
    if existing_meta is None:
        if has_all_pngs:
            return "不完全(要再補完)", False
        return "未補完", False

    if not has_all_pngs:
        return "不完全(PNG不足)", False

    if existing_meta.get("schema_version") != SCHEMA_VERSION:
        return "再補完要(スキーマ更新)", False

    model_info = existing_meta.get("model") or {}
    existing_mode = model_info.get("mode")
    existing_hash = str(model_info.get("config_hash", ""))
    existing_quality = model_info.get("quality")

    if requested_mode == "temporal" and existing_mode != "temporal_prev_11ch":
        return "再補完要(モード変更)", False
    if requested_mode == "single" and existing_mode != "single_pass_8ch":
        return "再補完要(モード変更)", False

    expected_hash = expected_hashes.get(str(existing_mode or ""))
    if not expected_hash or existing_hash != expected_hash:
        return "再補完要(モデル更新)", False

    if existing_quality != requested_quality:
        return "再補完要(品質変更)", False

    return f"補完済(一致:{existing_mode})", True


def _has_band_images(pass_info: dict[str, Any]) -> bool:
    files = pass_info.get("files") or {}
    return any(b in files for b in ("MSU-MR-1.png", "MSU-MR-2.png", "MSU-MR-4.png"))


def _fetch_existing_metadata_map(
    client: GarageS3Client | None,
    passes: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {p["pass_id"]: None for p in passes}
    if client is None:
        return result

    targets = [
        (p["pass_id"], p["files"][INPAINT_META_REL]["key"])
        for p in passes
        if INPAINT_META_REL in (p.get("files") or {})
    ]
    if not targets:
        return result

    def _get(item: tuple[str, str]) -> tuple[str, dict[str, Any] | None]:
        pid, key = item
        return pid, client.get_json_if_exists(key)

    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
        for pid, meta in pool.map(_get, targets):
            result[pid] = meta
    return result


def _discover_meteor_passes(
    data_dir: Path,
    refresh: bool,
    endpoint_url: str | None,
    bucket: str | None,
    ssh_host: str,
    no_sync: bool,
) -> tuple[list[dict[str, Any]], GarageS3Client | None]:
    meteor_dir = data_dir / "meteor"
    index_path = meteor_dir / "meteor_index.json"
    client: GarageS3Client | None = None

    if no_sync:
        if not index_path.exists():
            raise FileNotFoundError(f"--no-sync が指定されましたが {index_path} が存在しません")
        records = json.loads(index_path.read_text(encoding="utf-8"))
        return records, None

    client = build_client(endpoint_url=endpoint_url, bucket=bucket, ssh_host=ssh_host)
    if refresh or not index_path.exists():
        # 環境変数や上書きパラメータが反映されるよう、一覧取得前にエンドポイント等を同期
        import os

        if endpoint_url:
            os.environ["GARAGE_S3_ENDPOINT"] = endpoint_url
        if bucket:
            os.environ["GARAGE_BUCKET"] = bucket
        records = sync_meteor_dataset(meteor_dir, refresh=True)
    else:
        records = json.loads(index_path.read_text(encoding="utf-8"))
    return records, client


def _filter_target_passes(
    records: list[dict[str, Any]],
    pass_ids: list[str] | None,
    latest: int | None,
    exclude_golden: bool,
) -> list[dict[str, Any]]:
    eligible = [p for p in records if p.get("has_decoded_images") and _has_band_images(p)]
    if exclude_golden:
        eligible = [p for p in eligible if not p.get("is_golden")]

    if pass_ids:
        by_id = {p["pass_id"]: p for p in eligible}
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for pid in pass_ids:
            if pid in by_id:
                selected.append(by_id[pid])
            else:
                missing.append(pid)
        if missing:
            raise ValueError(f"指定された pass_id が対象（復号済み METEOR パス）に見つかりません: {', '.join(missing)}")
        return selected

    eligible_sorted = sorted(eligible, key=lambda x: x["pass_id"], reverse=True)
    if latest is not None and latest > 0:
        return eligible_sorted[:latest]
    return eligible_sorted


def _print_pass_list(
    targets: list[dict[str, Any]],
    meta_map: dict[str, dict[str, Any] | None],
    expected_hashes: dict[str, str],
    mode: str,
    quality: str,
) -> None:
    print(f"=== METEOR 補完対象パス一覧 (計 {len(targets)} 件) ===")
    print(f"{'Pass ID':30s} | {'Sat':11s} | {'Bands':5s} | {'Status'}")
    print("-" * 82)
    for p in targets:
        pid = p["pass_id"]
        files = p.get("files") or {}
        bands = "".join(b[-5] for b in ("MSU-MR-1.png", "MSU-MR-2.png", "MSU-MR-4.png") if b in files)
        status_label, _ = evaluate_pass_inpaint_status(
            p,
            meta_map.get(pid),
            expected_hashes,
            requested_mode=mode,
            requested_quality=quality,
        )
        print(f"{pid:30s} | {p['satellite']:11s} | {bands:5s} | {status_label}")


def _upload_outputs(
    client: GarageS3Client,
    satellite: str,
    pass_id: str,
    out_dir: Path,
) -> None:
    prefix = f"results/{satellite}/{pass_id}/ml_inpainted"
    # PNG 4 枚を先にすべてアップロードし、完了マーカーとして metadata.json を最後に PUT する
    for fname in INPAINT_PNG_NAMES:
        client.put_file(f"{prefix}/{fname}", out_dir / fname, content_type="image/png")
    client.put_file(f"{prefix}/metadata.json", out_dir / "metadata.json", content_type="application/json; charset=utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mac (MPS/CPU) で METEOR-M2 欠損補完を実行し Garage S3 へ転送するパイプライン",
    )
    parser.add_argument("--list", action="store_true", help="対象 METEOR パス一覧と S3 補完状態を表示して終了")
    parser.add_argument("--pass-id", action="append", dest="pass_ids", help="処理対象の pass_id（複数回指定可）")
    parser.add_argument("--latest", type=int, default=None, help="最新の復号済みパス N 件のみ処理")
    parser.add_argument("--mode", choices=["auto", "temporal", "single"], default="auto", help="推論モード (既定: auto)")
    parser.add_argument("--quality", choices=["standard", "high"], default="standard", help="推論品質 (既定: standard)")
    parser.add_argument("--dry-run", action="store_true", help="ローカル保存のみ行い Garage S3 へアップロードしない")
    parser.add_argument("--force", action="store_true", help="既に同一条件で補完済みでも再推論・再アップロードする")
    parser.add_argument("--no-sync", action="store_true", help="S3 接続を行わずローカルキャッシュのみで実行（--dry-run 推奨）")
    parser.add_argument("--exclude-golden", action="store_true", help="GOLDEN サンプルパスを対象から除外する")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data", help="ローカルデータディレクトリ")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "models", help="学習済みバンドルディレクトリ")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs", help="補完結果のローカル出力先")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"Garage S3 エンドポイント (既定: {DEFAULT_ENDPOINT})")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"Garage S3 バケット (既定: {DEFAULT_BUCKET})")
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST, help=f"認証情報取得先 SSH ホスト (既定: {DEFAULT_SSH_HOST})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_sync and not args.dry_run and not args.list:
        print("[エラー] --no-sync 時は Garage S3 へのアップロードはできません。--dry-run または --list を併用してください。", flush=True)
        return 2

    records, client = _discover_meteor_passes(
        data_dir=args.data_dir,
        refresh=not args.no_sync,
        endpoint_url=args.endpoint,
        bucket=args.bucket,
        ssh_host=args.ssh_host,
        no_sync=args.no_sync,
    )
    targets = _filter_target_passes(
        records,
        pass_ids=args.pass_ids,
        latest=args.latest,
        exclude_golden=args.exclude_golden,
    )

    expected_hashes = load_bundle_hashes(args.model_dir)
    meta_map = _fetch_existing_metadata_map(client, targets)

    if args.list:
        _print_pass_list(targets, meta_map, expected_hashes, args.mode, args.quality)
        return 0

    if not targets:
        print("処理対象の復号済み METEOR パスが見つかりませんでした。", flush=True)
        return 0

    run_queue: list[dict[str, Any]] = []
    skipped_count = 0
    for p in targets:
        pid = p["pass_id"]
        status_label, should_skip = evaluate_pass_inpaint_status(
            p,
            meta_map.get(pid),
            expected_hashes,
            requested_mode=args.mode,
            requested_quality=args.quality,
        )
        if should_skip and not args.force:
            print(f"[SKIP] {pid}: {status_label}", flush=True)
            skipped_count += 1
        else:
            run_queue.append(p)

    if not run_queue:
        print(f"すべての対象パス（{skipped_count} 件）が最新の条件ハッシュで補完済みです。", flush=True)
        return 0

    print(
        f"補完パイプライン開始: 実行 {len(run_queue)} 件 / スキップ {skipped_count} 件 "
        f"(mode={args.mode}, quality={args.quality}, dry_run={args.dry_run})",
        flush=True,
    )

    meteor_dir = args.data_dir / "meteor"
    # 過去パス投影プール用の基準パス群（非 GOLDEN・3 バンド揃い上位 6 件）もローカルキャッシュ確保
    if client is not None:
        pool_refs = [
            p
            for p in records
            if p.get("has_decoded_images")
            and not p.get("is_golden")
            and all(b in (p.get("files") or {}) for b in ("MSU-MR-1.png", "MSU-MR-2.png", "MSU-MR-4.png"))
        ]
        pool_refs = sorted(pool_refs, key=lambda x: x["pass_id"], reverse=True)[:6]
        for ref in pool_refs:
            download_pass_bands(client, ref, meteor_dir)

    engine = MeteorInpaintingEngine(model_dir=args.model_dir, data_dir=args.data_dir)
    print(f"推論デバイス: {engine.device} | nb11_hash={engine.nb11_hash} | nb09_hash={engine.nb09_hash}", flush=True)

    t_all = time.perf_counter()
    failures: list[tuple[str, str]] = []
    total = len(run_queue)
    for idx, p in enumerate(run_queue, start=1):
        pid = p["pass_id"]
        sat = p["satellite"]
        pct_before = ((idx - 1) / total) * 100.0
        elapsed_so_far = time.perf_counter() - t_all
        print(f"[{idx}/{total} ({pct_before:5.1f}%) | Elapsed {elapsed_so_far:5.1f}s] 開始: {pid}", flush=True)
        try:
            if client is not None:
                download_pass_bands(client, p, meteor_dir)

            res = engine.inpaint_pass(
                pass_id=pid,
                satellite=sat,
                pass_dir=meteor_dir / pid,
                mode=args.mode,
                quality=args.quality,
                is_golden=bool(p.get("is_golden")),
            )

            out_dir = args.output_dir / pid
            out_dir.mkdir(parents=True, exist_ok=True)
            for fname, arr in res["images"].items():
                save_rgb_png(arr, out_dir / fname)

            meta_path = out_dir / "metadata.json"
            tmp_meta = meta_path.with_suffix(".tmp.json")
            tmp_meta.write_text(json.dumps(res["metadata"], indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_meta.replace(meta_path)

            if not args.dry_run and client is not None:
                _upload_outputs(client, sat, pid, out_dir)
                upload_status = "S3転送済"
            else:
                upload_status = "ローカル保存のみ(dry-run)"

            elapsed = time.perf_counter() - t_all
            eta = (elapsed / idx) * (total - idx)
            pct_done = (idx / total) * 100.0
            meta = res["metadata"]
            print(
                f"[{idx}/{total} ({pct_done:5.1f}%) | Elapsed {elapsed:5.1f}s | ETA {eta:5.1f}s] 完了: {pid} | "
                f"model={meta['model']['mode']} | reg={meta['registration']['mode']} | "
                f"tier={meta['quality']['tier']} | jump={meta['quality']['seam_jump_l1']:.4f} | {upload_status}",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t_all
            failures.append((pid, str(exc)))
            print(f"[{idx}/{total} | Elapsed {elapsed:5.1f}s] 失敗: {pid} -> {exc}", flush=True)

    if failures:
        print(f"\n[エラー] {len(failures)} 件の処理に失敗しました:", flush=True)
        for pid, msg in failures:
            print(f"  - {pid}: {msg}", flush=True)
        return 1

    total_dt = time.perf_counter() - t_all
    print(f"\n全 {total} 件の補完処理が正常終了しました (合計 {total_dt:.1f}s)。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
