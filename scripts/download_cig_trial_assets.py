"""Fetch selected public SD1.5/CLIP assets; never download entire model repositories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    selections = {
        "sd15": ("AI-ModelScope/stable-diffusion-v1-5", [
            "model_index.json", "scheduler/scheduler_config.json", "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors", "text_encoder/config.json",
            "text_encoder/model.safetensors", "tokenizer/merges.txt", "tokenizer/vocab.json",
            "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json"]),
        "clip_l14": ("AI-ModelScope/clip-vit-large-patch14", [
            "config.json", "preprocessor_config.json", "model.safetensors"]),
    }
    plan = []
    for folder, (repo, names) in selections.items():
        url = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
        with opener.open(url, timeout=60) as response:
            entries = {r["Path"]: r for r in json.load(response)["Data"]["Files"]}
        for name in names:
            row = entries[name]
            plan.append(dict(folder=folder, repository=repo, path=name, size=row["Size"],
                             sha256=row["Sha256"], revision=row["Revision"]))
    missing_bytes = sum(r["size"] for r in plan if not (args.output / r["folder"] / r["path"]).exists())
    if shutil.disk_usage(args.output).free - missing_bytes < 8 * 1024**3:
        raise RuntimeError("Insufficient disk for all selected assets plus 8 GiB reserve")
    (args.output / "download_plan.json").write_text(json.dumps(plan, indent=2))
    for row in plan:
        path = args.output / row["folder"] / row["path"]
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(4 * 1024**2):
                    digest.update(chunk)
            if path.stat().st_size != row["size"] or digest.hexdigest() != row["sha256"]:
                raise RuntimeError(f"Existing asset does not match source: {row['path']}")
            print("verified_existing", row["folder"], row["path"], flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        if partial.exists():
            raise RuntimeError(f"Inspect interrupted download before resuming: {partial}")
        query = urllib.parse.urlencode(dict(Revision=row["revision"], FilePath=row["path"]))
        url = f"https://modelscope.cn/api/v1/models/{row['repository']}/repo?{query}"
        digest, count = hashlib.sha256(), 0
        with opener.open(url, timeout=90) as response, partial.open("xb") as output:
            while chunk := response.read(4 * 1024**2):
                if shutil.disk_usage(args.output).free < 8 * 1024**3 + len(chunk):
                    raise RuntimeError("Disk reserve reached during asset download")
                count += len(chunk)
                if count > row["size"]:
                    raise RuntimeError("Download exceeds declared size")
                output.write(chunk)
                digest.update(chunk)
        if count != row["size"] or digest.hexdigest() != row["sha256"]:
            raise RuntimeError(f"Asset integrity check failed: {row['path']}")
        partial.replace(path)
        print("downloaded_verified", row["folder"], row["path"], count, flush=True)
    (args.output / "assets_verified.json").write_text(json.dumps(dict(status="verified", files=plan), indent=2))


if __name__ == "__main__":
    main()
