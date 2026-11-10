"""Restore one of the published FashionCIR source snapshots with hash checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


REPOSITORY = Path(__file__).resolve().parents[1]


def relative_path(value):
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts
            or any(part in (".", "..") or ":" in part or "\\" in part for part in path.parts)):
        raise ValueError(f"Unsafe path in the source manifest: {value}")
    return Path(*path.parts)


def source_bytes(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def restore(entries, scope, prefix, output):
    output = output.resolve()
    staged = {}
    for entry in entries:
        if entry["scope"] != scope or not entry["path"].startswith(prefix):
            continue
        source = REPOSITORY / relative_path(entry["published_path"])
        if not source.resolve().is_relative_to(REPOSITORY):
            raise ValueError(f"Published path leaves the repository: {source}")
        data = source_bytes(source)
        if hashlib.sha256(data).hexdigest() != entry["published_sha256"]:
            raise ValueError(f"Published source checksum differs: {source}")
        target = output / relative_path(entry["path"][len(prefix):])
        if target.is_symlink() or not target.resolve().is_relative_to(output):
            raise ValueError(f"Unsafe destination: {target}")
        if target.exists() and (not target.is_file() or source_bytes(target) != data):
            raise FileExistsError(f"Destination has different content: {target}")
        staged[target] = (data, entry["executable"])
    if not staged:
        raise ValueError("No files match this snapshot")
    for target, (data, executable) in staged.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)
            if executable and os.name == "posix":
                target.chmod(target.stat().st_mode | 0o111)
    return len(staged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--list", action="store_true")
    selector.add_argument("--run")
    selector.add_argument("--workspace-experiment")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.list and args.output is None:
        parser.error("--output is required when restoring a snapshot")
    manifest = REPOSITORY / "docs/server_source_manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))["entries"]
    if args.list:
        names = {
            f"run: {entry['path'].split('/', 1)[0]}" for entry in entries if entry["scope"] == "run"
        }
        names.update(
            f"workspace-experiment: {entry['path'].split('/')[1]}"
            for entry in entries if entry["scope"] == "workspace" and entry["path"].startswith("experiments/")
        )
        print("\n".join(sorted(names)))
        return
    if args.run:
        scope, prefix = "run", f"{args.run}/"
    else:
        scope, prefix = "workspace", f"experiments/{args.workspace_experiment}/"
    print(f"Verified and restored {restore(entries, scope, prefix, args.output)} source files.")


if __name__ == "__main__":
    main()
