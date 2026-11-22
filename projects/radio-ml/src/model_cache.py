"""学習条件ハッシュ（config_hash）付きで学習済みモデルバンドルをキャッシュ管理するユーティリティ。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _canonicalize_value(val: Any) -> Any:
    """JSON 直列化時に型差異でハッシュがぶれないよう再帰的に標準 Python 型へ正規化する。"""
    if isinstance(val, dict):
        return {str(k): _canonicalize_value(v) for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))}
    if isinstance(val, (list, tuple)):
        return [_canonicalize_value(v) for v in val]
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, bool) or val is None or isinstance(val, (int, float, str)):
        return val
    if hasattr(val, "item"):
        return val.item()
    return str(val)


def compute_config_hash(train_config: dict[str, Any]) -> str:
    """学習条件辞書（アーキテクチャ・ハイパーパラメータ・損失重み・データ件数等）の決定論的 SHA-256 先頭 16 桁を返す。"""
    canonical = _canonicalize_value(train_config)
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def check_model_cache(
    bundle_path: Path | str,
    train_config: dict[str, Any],
    force_retrain: bool = False,
) -> tuple[bool, str, dict[str, Any] | None]:
    """学習済みキャッシュの存在と条件ハッシュ一致を検証する。

    Returns:
        (is_hit, config_hash, bundle): キャッシュが有効なら (True, config_hash, bundle)、
        未保存・条件不一致・force_retrain=True なら (False, config_hash, None)。
    """
    path = Path(bundle_path)
    config_hash = compute_config_hash(train_config)

    if force_retrain:
        print(
            f"[🔄 モデル新規学習] {path.name} (config_hash={config_hash}, FORCE_RETRAIN=True) -> 学習を実行しキャッシュへ保存します",
            flush=True,
        )
        return False, config_hash, None

    if not path.exists():
        print(
            f"[🔄 モデル新規学習] {path.name} (config_hash={config_hash}) -> 学習を実行しキャッシュへ保存します",
            flush=True,
        )
        return False, config_hash, None

    try:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(
            f"[🔄 モデル再学習] {path.name} の読込に失敗 ({exc}) -> 再学習してキャッシュを更新します",
            flush=True,
        )
        return False, config_hash, None

    old_hash = bundle.get("config_hash") if isinstance(bundle, dict) else None
    has_weights = isinstance(bundle, dict) and ("state_dict" in bundle)

    if has_weights and old_hash == config_hash:
        print(
            f"[💾 モデルキャッシュ HIT] {path.name} (config_hash={config_hash}) -> 条件未変更のため学習をスキップし保存済み重みを即時ロードします",
            flush=True,
        )
        return True, config_hash, bundle

    old_label = old_hash if old_hash is not None else "未設定(旧形式)"
    print(
        f"[🔄 モデル再学習] {path.name} の学習条件変更を検知 (旧={old_label} -> 新={config_hash}) -> 再学習してキャッシュを更新します",
        flush=True,
    )
    return False, config_hash, None


def save_model_cache(
    bundle_path: Path | str,
    train_config: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    """学習済み重み・学習曲線・較正行列とともに条件ハッシュを付与しアトミック保存する。"""
    path = Path(bundle_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_hash = compute_config_hash(train_config)

    to_save = dict(payload)
    to_save["config_hash"] = config_hash
    to_save["train_config"] = _canonicalize_value(train_config)

    tmp_path = path.with_suffix(".tmp.pt")
    torch.save(to_save, tmp_path)
    tmp_path.replace(path)

    size_mb = path.stat().st_size / (1024.0 * 1024.0)
    print(
        f"[💾 モデルキャッシュ保存完了] {path.name} ({size_mb:.2f} MB | config_hash={config_hash})",
        flush=True,
    )
    return config_hash
