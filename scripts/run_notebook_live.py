"""リアルタイム進捗・ETA を標準出力へストリーミングしながら Jupyter Notebook を実行する CLI。

nbconvert のサイレント実行とは異なり、Background Task Output 上で以下をリアルタイム可視化する:
- コードセル単位の進捗率 (%)・見出しヒント・経過時間・推定残り時間 (ETA)
- カーネル内の print(..., flush=True) 出力の即時中継
- plt.show() 等の PNG 図表生成通知
- 無音演算中の定期ハートビート表示 (デフォルト 15 秒間隔)
- セル完了ごとのインクリメンタル保存 (途中停止時も実行済みセルの出力を保持)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import re
import sys
import time
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.client import CellExecutionComplete


def _fmt_duration(seconds: float) -> str:
    """秒数を MM:SS または HH:MM:SS 形式に整形する。"""
    sec = max(0, int(round(seconds)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _extract_cell_label(nb: nbformat.NotebookNode, cell_index: int) -> str:
    """セル先頭のコメントまたは直前の Markdown 見出しから簡潔な説明ラベルを抽出する。"""
    code_lines = str(nb.cells[cell_index].get("source", "")).splitlines()
    skip_prefixes = (
        "# 日本語フォント設定",
        "%matplotlib",
        "import ",
        "from ",
        "try:",
        "except",
        "pass",
        "_available_fonts",
        "_jp_fonts",
        "plt.rcParams",
    )
    for raw in code_lines:
        stripped = raw.strip()
        if not stripped or any(stripped.startswith(p) for p in skip_prefixes):
            continue
        if stripped.startswith("#"):
            cleaned = re.sub(r"^#+\s*-*\s*|\s*-*$", "", stripped)
            if cleaned:
                return cleaned[:64]
        break

    # コード先頭に説明コメントがない場合は直前の Markdown 見出しを参照
    for prev_i in range(cell_index - 1, -1, -1):
        prev_cell = nb.cells[prev_i]
        if prev_cell.get("cell_type") == "markdown":
            for mline in str(prev_cell.get("source", "")).splitlines():
                ms = mline.strip()
                if ms.startswith("#"):
                    return re.sub(r"^#+\s*", "", ms)[:64]
            break
    return f"Code Cell #{cell_index}"


class LiveStreamingNotebookClient(NotebookClient):
    """セルの進捗・ETA・カーネル標準出力・図表出力を即時ストリーム表示する NotebookClient。"""

    def __init__(
        self,
        nb: nbformat.NotebookNode,
        output_path: Path,
        heartbeat_sec: float = 15.0,
        save_each_cell: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(nb, **kwargs)
        self.output_path = output_path
        self.heartbeat_sec = max(1.0, float(heartbeat_sec))
        self.save_each_cell = save_each_cell

        self._code_indices = [
            idx for idx, c in enumerate(nb.cells) if c.get("cell_type") == "code"
        ]
        self._code_pos_map = {idx: pos + 1 for pos, idx in enumerate(self._code_indices)}
        self._total_code = max(len(self._code_indices), 1)
        self._cell_labels = {
            idx: _extract_cell_label(nb, idx) for idx in self._code_indices
        }

        self._run_start: float = 0.0
        self._cell_start: float = 0.0
        self._last_emit: float = 0.0
        self._active_pos: int = 0
        self._stream_at_line_start: bool = True
        self._completed_count: int = 0

    def _emit_line(self, text: str) -> None:
        """進捗行を即座にフラッシュ出力する。"""
        if not self._stream_at_line_start:
            sys.stdout.write("\n")
            self._stream_at_line_start = True
        sys.stdout.write(text.rstrip("\n") + "\n")
        sys.stdout.flush()
        self._last_emit = time.monotonic()

    def _save_atomic(self) -> None:
        """実行中ノートブックを安全に書き出す。"""
        tmp_path = self.output_path.with_name(f".{self.output_path.name}.tmp")
        nbformat.write(self.nb, str(tmp_path))
        tmp_path.replace(self.output_path)

    async def _heartbeat_loop(self, pos: int) -> None:
        """カーネル出力が途絶えた間に生存・経過時刻のハートビートを表示する。"""
        while True:
            await asyncio.sleep(min(2.0, self.heartbeat_sec))
            now = time.monotonic()
            if now - self._last_emit >= self.heartbeat_sec:
                c_dt = _fmt_duration(now - self._cell_start)
                t_dt = _fmt_duration(now - self._run_start)
                self._emit_line(
                    f"[⏳ 実行中 Cell {pos}/{self._total_code} | セル経過 {c_dt} | 全体経過 {t_dt}] カーネル計算継続中..."
                )

    def process_message(
        self, msg: dict[str, Any], cell: nbformat.NotebookNode, cell_index: int
    ) -> nbformat.NotebookNode | None:
        out: nbformat.NotebookNode | None = None
        try:
            out = super().process_message(msg, cell, cell_index)
        except CellExecutionComplete:
            raise

        if out is None:
            return None

        pos = self._code_pos_map.get(cell_index, self._active_pos)
        otype = out.get("output_type")

        if otype == "stream":
            txt = "".join(out.get("text", []))
            if txt:
                prefix = f"[Cell {pos}/{self._total_code}] "
                for chunk in txt.splitlines(keepends=True):
                    if self._stream_at_line_start:
                        sys.stdout.write(prefix)
                    sys.stdout.write(chunk)
                    self._stream_at_line_start = chunk.endswith("\n")
                sys.stdout.flush()
                self._last_emit = time.monotonic()

        elif otype in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "image/png" in data:
                raw_b64 = data["image/png"]
                kb = len(base64.b64decode(raw_b64)) / 1024.0
                c_dt = _fmt_duration(time.monotonic() - self._cell_start)
                self._emit_line(
                    f"[📊 Cell {pos}/{self._total_code} | +{c_dt}] 図表生成完了 ({kb:.1f} KB)"
                )

        elif otype == "error":
            ename = out.get("ename", "Error")
            evalue = out.get("evalue", "")
            self._emit_line(f"[❌ Cell {pos}/{self._total_code} エラー] {ename}: {evalue}")

        return out

    async def async_execute_cell(
        self,
        cell: nbformat.NotebookNode,
        cell_index: int,
        execution_count: int | None = None,
        store_history: bool = True,
    ) -> nbformat.NotebookNode:
        if cell.get("cell_type") != "code":
            return await super().async_execute_cell(
                cell, cell_index, execution_count=execution_count, store_history=store_history
            )

        now = time.monotonic()
        if self._run_start == 0.0:
            self._run_start = now
        self._cell_start = now
        self._last_emit = now

        pos = self._code_pos_map.get(cell_index, 0)
        self._active_pos = pos
        pct_before = (self._completed_count / self._total_code) * 100.0
        elapsed = now - self._run_start
        eta_str = (
            _fmt_duration((elapsed / self._completed_count) * (self._total_code - self._completed_count))
            if self._completed_count > 0
            else "計測中"
        )
        label = self._cell_labels.get(cell_index, "")
        self._emit_line(
            f"[▶ Cell {pos}/{self._total_code} ({pct_before:4.1f}%) | 全体経過 {_fmt_duration(elapsed)} | 残り目安 {eta_str}] {label}"
        )

        hb_task = asyncio.create_task(self._heartbeat_loop(pos))
        try:
            result = await super().async_execute_cell(
                cell, cell_index, execution_count=execution_count, store_history=store_history
            )
        finally:
            hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb_task

        done_now = time.monotonic()
        self._completed_count += 1
        c_dt = done_now - self._cell_start
        tot_dt = done_now - self._run_start
        pct_after = (self._completed_count / self._total_code) * 100.0
        rem_cells = self._total_code - self._completed_count
        eta_after = (
            _fmt_duration((tot_dt / self._completed_count) * rem_cells)
            if rem_cells > 0
            else "00:00"
        )
        self._emit_line(
            f"[✅ Cell {pos}/{self._total_code} 完了 ({c_dt:.1f}s) | 進捗 {pct_after:5.1f}% | 全体 {_fmt_duration(tot_dt)} | 残り目安 {eta_after}]"
        )
        if self.save_each_cell:
            self._save_atomic()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupyter Notebook をリアルタイム進捗・ETA 表示付きで実行します（Background Task Output 可視化用）。"
    )
    parser.add_argument("notebook", type=Path, help="実行対象の .ipynb ファイルパス")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力先 .ipynb パス（省略時は入力ファイルを上書き更新）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="1 セルあたりのタイムアウト秒数 (デフォルト: 1200)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="python3",
        help="Jupyter カーネル名 (デフォルト: python3)",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=15.0,
        help="カーネル無出力時のハートビート出力間隔[秒] (デフォルト: 15.0)",
    )
    parser.add_argument(
        "--no-save-each-cell",
        action="store_true",
        help="セル完了ごとの中間保存を無効化（最終保存のみ実施）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nb_path = args.notebook.expanduser().resolve()
    if not nb_path.exists() or nb_path.suffix != ".ipynb":
        print(f"[エラー] ノートブックが見つからないか拡張子が不正です: {nb_path}", file=sys.stderr)
        return 1

    out_path = args.output.expanduser().resolve() if args.output else nb_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(str(nb_path), as_version=4)
    n_code = sum(1 for c in nb.cells if c.get("cell_type") == "code")
    print(
        f"[🚀 ノートブック実行開始] {nb_path.name} (全 {n_code} コードセル | timeout={args.timeout}s | kernel={args.kernel})",
        flush=True,
    )

    t0 = time.monotonic()
    client = LiveStreamingNotebookClient(
        nb,
        output_path=out_path,
        heartbeat_sec=args.heartbeat,
        save_each_cell=not args.no_save_each_cell,
        timeout=args.timeout,
        kernel_name=args.kernel,
    )
    try:
        client.execute()
    except Exception as exc:
        print(
            f"[❌ 実行失敗 | 経過 {_fmt_duration(time.monotonic() - t0)}] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    client._save_atomic()
    print(
        f"[🎉 完了] {out_path} ({out_path.stat().st_size / 1e6:.2f} MB | 総時間 {_fmt_duration(time.monotonic() - t0)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
