"""Garage S3 から radio-astronomy の観測データを取得・ローカルキャッシュするユーティリティ。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

S3_XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

DEFAULT_ENDPOINT = "http://192.168.68.66:30900"
DEFAULT_BUCKET = "satellite-recordings"
DEFAULT_SSH_HOST = "tozastation@192.168.68.66"


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    return _sign(k_service, "aws4_request")


class GarageS3Client:
    """Python 標準ライブラリのみで実装した最小限の Garage S3 (SigV4) クライアント。"""

    def __init__(
        self,
        endpoint_url: str = DEFAULT_ENDPOINT,
        bucket: str = DEFAULT_BUCKET,
        access_key: str = "",
        secret_key: str = "",
        region: str = "garage",
        timeout_sec: int = 15,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.timeout_sec = timeout_sec

    def _request(
        self,
        path: str,
        query_params: dict[str, str] | None = None,
        method: str = "GET",
        body: bytes = b"",
        content_type: str | None = None,
    ) -> bytes:
        parsed = urllib.parse.urlparse(self.endpoint_url)
        host = parsed.netloc
        canonical_uri = urllib.parse.quote(f"/{self.bucket}{path}", safe="/-_.~")

        params = dict(query_params or {})
        sorted_items = sorted(params.items())
        canonical_querystring = "&".join(
            f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
            for k, v in sorted_items
        )

        now = datetime.now(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(body).hexdigest()

        canonical_headers = (
            f"host:{host}\n"
            f"x-amz-content-sha256:{payload_hash}\n"
            f"x-amz-date:{amz_date}\n"
        )
        signed_headers = "host;x-amz-content-sha256;x-amz-date"
        canonical_request = (
            f"{method}\n{canonical_uri}\n{canonical_querystring}\n"
            f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
        )

        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = (
            f"{algorithm}\n{amz_date}\n{credential_scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )
        signing_key = _get_signature_key(self.secret_key, date_stamp, self.region, "s3")
        signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        authorization = (
            f"{algorithm} Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        url = f"{self.endpoint_url}{canonical_uri}"
        if canonical_querystring:
            url += f"?{canonical_querystring}"

        headers = {
            "x-amz-date": amz_date,
            "x-amz-content-sha256": payload_hash,
            "Authorization": authorization,
        }
        if method == "PUT":
            headers["Content-Type"] = content_type or "application/octet-stream"
            headers["Content-Length"] = str(len(body))
        elif content_type:
            headers["Content-Type"] = content_type

        req = urllib.request.Request(
            url,
            data=body if method == "PUT" else None,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            return resp.read()

    def list_objects(self, prefix: str) -> list[dict[str, Any]]:
        """指定プレフィックス配下のオブジェクト一覧を全件取得する。"""
        objects: list[dict[str, Any]] = []
        continuation_token: str | None = None

        while True:
            params: dict[str, str] = {"list-type": "2", "prefix": prefix}
            if continuation_token:
                params["continuation-token"] = continuation_token

            body = self._request("", params)
            root = ET.fromstring(body)
            for contents in root.findall("s3:Contents", S3_XML_NS):
                key_el = contents.find("s3:Key", S3_XML_NS)
                mod_el = contents.find("s3:LastModified", S3_XML_NS)
                size_el = contents.find("s3:Size", S3_XML_NS)
                if key_el is None or key_el.text is None:
                    continue
                objects.append(
                    {
                        "key": key_el.text,
                        "last_modified": mod_el.text if mod_el is not None else None,
                        "size": int(size_el.text) if size_el is not None and size_el.text else 0,
                    }
                )

            truncated_el = root.find("s3:IsTruncated", S3_XML_NS)
            is_truncated = truncated_el is not None and truncated_el.text == "true"
            if not is_truncated:
                break
            token_el = root.find("s3:NextContinuationToken", S3_XML_NS)
            if token_el is None or not token_el.text:
                break
            continuation_token = token_el.text

        return objects

    def get_bytes(self, key: str) -> bytes:
        return self._request("/" + key.lstrip("/"))

    def get_json(self, key: str) -> dict[str, Any]:
        return json.loads(self.get_bytes(key).decode("utf-8"))

    def get_json_if_exists(self, key: str) -> dict[str, Any] | None:
        """指定キーが存在すれば JSON を読み込み、404 なら None を返す。"""
        try:
            return self.get_json(key)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def object_exists(self, key: str) -> bool:
        """HEAD リクエストで S3 オブジェクトの存在を確認する。"""
        try:
            self._request("/" + key.lstrip("/"), method="HEAD")
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """バイト列を指定キーへ PUT アップロードする。"""
        self._request(
            "/" + key.lstrip("/"),
            method="PUT",
            body=data,
            content_type=content_type,
        )

    def put_file(self, key: str, src: Path, content_type: str | None = None) -> None:
        """ローカルファイルを指定キーへ PUT アップロードする。"""
        src_path = Path(src)
        ctype = content_type
        if ctype is None:
            suffix = src_path.suffix.lower()
            if suffix == ".png":
                ctype = "image/png"
            elif suffix == ".json":
                ctype = "application/json; charset=utf-8"
            else:
                ctype = "application/octet-stream"
        self.put_bytes(key, src_path.read_bytes(), content_type=ctype)

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        """辞書を整形 JSON として PUT アップロードする。"""
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.put_bytes(key, data, content_type="application/json; charset=utf-8")

    def download_if_missing(self, key: str, dest: Path) -> Path:
        """ローカルにファイルが存在しない場合のみ S3 からダウンロードする。"""
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self.get_bytes(key)
        dest.write_bytes(data)
        return dest


def resolve_credentials(ssh_host: str = DEFAULT_SSH_HOST) -> tuple[str, str]:
    """環境変数または SSH 経由の kubectl から Garage S3 認証情報を取得する。"""
    ak = os.environ.get("GARAGE_ACCESS_KEY", "")
    sk = os.environ.get("GARAGE_SECRET_KEY", "")
    if ak and sk:
        return ak, sk

    out = subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            ssh_host,
            "kubectl get secret garage-satellite-credentials -o jsonpath={.data}",
        ],
        text=True,
        timeout=10,
    )
    sec = json.loads(out)
    return (
        base64.b64decode(sec["access_key"]).decode("utf-8"),
        base64.b64decode(sec["secret_key"]).decode("utf-8"),
    )


def build_client(
    endpoint_url: str | None = None,
    bucket: str | None = None,
    ssh_host: str = DEFAULT_SSH_HOST,
) -> GarageS3Client:
    ak, sk = resolve_credentials(ssh_host=ssh_host)
    return GarageS3Client(
        endpoint_url=endpoint_url or os.environ.get("GARAGE_S3_ENDPOINT", DEFAULT_ENDPOINT),
        bucket=bucket or os.environ.get("GARAGE_BUCKET", DEFAULT_BUCKET),
        access_key=ak,
        secret_key=sk,
        region=os.environ.get("GARAGE_REGION", "garage"),
    )


def sync_meteor_dataset(
    data_dir: Path,
    max_failed_spectrograms: int = 30,
    max_workers: int = 8,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """METEOR-M2 パスのメタデータ・スペクトログラム・復号画像をローカルにキャッシュする。"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    index_path = data_dir / "meteor_index.json"

    if not refresh and index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))

    try:
        client = build_client()
        objs = client.list_objects("results/")
    except (urllib.error.URLError, OSError, subprocess.SubprocessError) as exc:
        if index_path.exists():
            print(f"[warn] Garage S3 へ接続できないためローカルキャッシュを使用します: {exc}")
            return json.loads(index_path.read_text(encoding="utf-8"))
        raise

    passes: dict[str, dict[str, Any]] = {}
    for o in objs:
        key = o["key"]
        if "METEOR" not in key.upper():
            continue
        parts = key.split("/")
        if len(parts) < 4:
            continue
        sat, pass_id = parts[1], parts[2]
        filename = "/".join(parts[3:])
        entry = passes.setdefault(
            pass_id,
            {
                "satellite": sat,
                "pass_id": pass_id,
                "prefix": f"results/{sat}/{pass_id}",
                "is_golden": "GOLDEN" in pass_id.upper(),
                "files": {},
            },
        )
        entry["files"][filename] = {"key": key, "size": o["size"], "last_modified": o["last_modified"]}

    # 各パスの summary.json を取得
    def _fetch_summary(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        pid, info = item
        local_summary = data_dir / pid / "summary.json"
        if "summary.json" in info["files"]:
            if not local_summary.exists():
                client.download_if_missing(info["files"]["summary.json"]["key"], local_summary)
            info["summary"] = json.loads(local_summary.read_text(encoding="utf-8"))
        else:
            info["summary"] = {}
        decoded_imgs = [
            f
            for f in info["files"]
            if f.endswith((".png", ".jpg"))
            and "spectrogram" not in f
            and not f.startswith("ml_inpainted/")
        ]
        info["has_decoded_images"] = len(decoded_imgs) > 0
        info["decoded_image_count"] = len(decoded_imgs)
        return pid, info

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_summary, kv) for kv in passes.items()]
        for fut in as_completed(futures):
            pid, info = fut.result()
            passes[pid] = info

    # ダウンロード対象ファイルの選定:
    # - 画像ありパス: spectrogram.png + 代表画像 (AVHRR 221 / MSU-MR-1 / MSU-MR-2)
    # - 画像なしパス: 最新 max_failed_spectrograms 件の spectrogram.png
    download_tasks: list[tuple[str, Path]] = []
    preferred_imgs = [
        "msu_mr_rgb_AVHRR_221_False_Color.png",
        "MSU-MR-1.png",
        "MSU-MR-2.png",
        "MSU-MR-4.png",
    ]
    failed_candidates = sorted(
        [p for p in passes.values() if not p["has_decoded_images"] and "spectrogram.png" in p["files"]],
        key=lambda x: x["pass_id"],
        reverse=True,
    )[:max_failed_spectrograms]
    failed_ids = {p["pass_id"] for p in failed_candidates}

    for pid, info in passes.items():
        if info["has_decoded_images"] or pid in failed_ids:
            if "spectrogram.png" in info["files"]:
                download_tasks.append(
                    (info["files"]["spectrogram.png"]["key"], data_dir / pid / "spectrogram.png")
                )
        if info["has_decoded_images"]:
            for img_name in preferred_imgs:
                if img_name in info["files"]:
                    download_tasks.append(
                        (info["files"][img_name]["key"], data_dir / pid / img_name)
                    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(lambda args: client.download_if_missing(*args), download_tasks))

    records = sorted(passes.values(), key=lambda x: x["pass_id"])
    index_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def download_pass_bands(
    client: GarageS3Client,
    pass_info: dict[str, Any],
    data_dir: Path,
) -> list[Path]:
    """指定パスの推論に必要な MSU-MR バンド画像および基準 RGB 画像をローカルへキャッシュする。"""
    data_dir = Path(data_dir)
    pid = pass_info["pass_id"]
    target_names = (
        "MSU-MR-1.png",
        "MSU-MR-2.png",
        "MSU-MR-4.png",
        "msu_mr_rgb_AVHRR_221_False_Color.png",
    )
    downloaded: list[Path] = []
    files = pass_info.get("files") or {}
    for fname in target_names:
        if fname in files:
            dest = data_dir / pid / fname
            client.download_if_missing(files[fname]["key"], dest)
            downloaded.append(dest)
    return downloaded
