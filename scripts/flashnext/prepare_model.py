# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a checkpoint view without modifying source weights or configuration."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.absolute()
    if destination.exists():
        parser.error("destination must not exist; source checkpoint remains untouched")
    from safetensors import safe_open

    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not all(
            isinstance(name, str) and isinstance(file, str) for name, file in weight_map.items()
        )
    ):
        parser.error("checkpoint index must have a nonempty string-to-string weight_map")
    shards = set(weight_map.values())
    for shard in shards:
        if Path(shard).name != shard or not (source / shard).is_file():
            parser.error(f"missing or non-local checkpoint shard: {shard}")
        # Validate complete file lengths and index membership before creating
        # a view: a resumable transfer may leave a truncated shard on disk.
        required = {name for name, file_name in weight_map.items() if file_name == shard}
        with safe_open(source / shard, framework="numpy") as handle:
            if not required.issubset(handle.keys()):
                parser.error(f"checkpoint index references absent tensors in {shard}")
    config_bytes = (source / "config.json").read_bytes()
    config = json.loads(config_bytes)
    if args.text_only:
        config["language_model_only"] = True
    destination.mkdir(parents=True)
    for item in sorted(source.iterdir()):
        if not item.is_file() or item.name == "config.json":
            continue
        # Calibration artifacts also use .safetensors, but are not model
        # weights. The generic loader globs this extension rather than index.
        if item.suffix == ".safetensors" and item.name not in shards:
            continue
        (destination / item.name).symlink_to(item)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "source": str(source),
        "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "text_only": args.text_only,
        "weight_shards": sorted(shards),
        "weight_bytes_from_index": index.get("metadata", {}).get("total_size"),
    }
    (destination / "flashnext-view.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
