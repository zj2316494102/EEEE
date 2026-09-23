from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, __version__ as huggingface_hub_version
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
MANIFEST_PATH = MODELS_ROOT / "model_manifest.json"


MODELS = [
    {
        "name": "text",
        "repo_id": "answerdotai/ModernBERT-base",
        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "directory": "answerdotai--ModernBERT-base",
        "allow_patterns": ["*.json", "*.txt", "model.safetensors", "README.md"],
    },
    {
        "name": "asr_timestamp",
        "repo_id": "openai/whisper-large-v3-turbo",
        "revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "directory": "openai--whisper-large-v3-turbo",
        "allow_patterns": ["*.json", "*.txt", "model.safetensors", "README.md"],
    },
    {
        "name": "audio",
        "repo_id": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "directory": "microsoft--wavlm-base-plus",
        "allow_patterns": ["*.json", "pytorch_model.bin", "README.md"],
    },
    {
        "name": "vision",
        "repo_id": "dima806/facial_emotions_image_detection",
        "revision": "747cf16692eea925b54b7b543cf436848128b68d",
        "directory": "dima806--facial_emotions_image_detection",
        "allow_patterns": [
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "README.md",
        ],
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_records(directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        records.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_manifest(entries: list[dict[str, object]]) -> None:
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "huggingface_hub_version": huggingface_hub_version,
        "models": entries,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    entries: list[dict[str, object]] = []

    print(f"模型根目录: {MODELS_ROOT}", flush=True)
    print(f"下载组件: huggingface_hub {huggingface_hub_version}", flush=True)

    for index, item in enumerate(MODELS, start=1):
        target = MODELS_ROOT / item["directory"]
        repo_id = item["repo_id"]
        revision = item["revision"]
        print(
            f"\n[{index}/{len(MODELS)}] 开始: {repo_id}\n"
            f"固定版本: {revision}\n目标目录: {target}",
            flush=True,
        )

        # Confirm that the pinned revision still exists before downloading.
        info = api.model_info(repo_id, revision=revision)
        if info.sha != revision:
            raise RuntimeError(
                f"仓库 {repo_id} 返回的 commit {info.sha} 与固定版本 {revision} 不一致"
            )

        started = time.time()
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target),
            allow_patterns=item["allow_patterns"],
            max_workers=4,
        )
        records = file_records(target)
        size_bytes = sum(int(record["size_bytes"]) for record in records)
        entry = {
            "name": item["name"],
            "repo_id": repo_id,
            "revision": revision,
            "local_dir": str(target),
            "files": records,
            "total_size_bytes": size_bytes,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        entries.append(entry)
        write_manifest(entries)
        print(
            f"[{index}/{len(MODELS)}] 完成: {repo_id} | "
            f"{size_bytes / 1024**3:.2f} GiB | "
            f"{entry['elapsed_seconds']} 秒",
            flush=True,
        )

    print(f"\n全部模型下载完成。版本清单: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
