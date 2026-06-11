#!/usr/bin/env python3
"""
Merge a PyTorch Distributed Checkpoint (DCP / *.distcp + .metadata) directory
into a single torch-save .pth file.

Typical DINOv3 teacher conversion:

  # 1) Inspect keys first
  python merge_distcp_to_pth.py \
      --src /path/to/eval/training_12499/sharded_teacher_checkpoint \
      --list-keys

  # 2) Convert selected tensors to DINOv3-style {"teacher": state_dict}
  python merge_distcp_to_pth.py \
      --src /path/to/eval/training_12499/sharded_teacher_checkpoint \
      --dst /path/to/eval/training_12499/teacher_checkpoint.pth \
      --include-prefix model. \
      --strip-prefix model. \
      --wrap-key teacher \
      --mmap-dir /tmp

If --include-prefix is omitted, all tensor entries in the DCP checkpoint are loaded.
For a full training checkpoint, do NOT convert optimizer tensors unless you really
need them; use --list-keys and choose the teacher/model prefix you need.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def _is_dcp_dir(path: Path) -> bool:
    return path.is_dir() and (path / ".metadata").is_file()


def _prod(xs: Iterable[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _contiguous_stride(size: Iterable[int]) -> tuple[int, ...]:
    size = tuple(int(x) for x in size)
    stride = []
    acc = 1
    for dim in reversed(size):
        stride.append(acc)
        acc *= dim
    return tuple(reversed(stride))


def _tensor_nbytes(size: Iterable[int], dtype: torch.dtype) -> int:
    return _prod(size) * torch.empty((), dtype=dtype).element_size()


def _human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def _selected(key: str, include_prefixes: list[str], exclude_prefixes: list[str]) -> bool:
    if include_prefixes and not any(key.startswith(p) for p in include_prefixes):
        return False
    if exclude_prefixes and any(key.startswith(p) for p in exclude_prefixes):
        return False
    return True


def _strip_key(key: str, strip_prefixes: list[str]) -> str:
    for prefix in strip_prefixes:
        if prefix and key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _make_mmap_tensor(mmap_dir: Path, index: int, key: str, size: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    nbytes = _tensor_nbytes(size, dtype)
    if nbytes == 0:
        return torch.empty(size, dtype=dtype, device="cpu")

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    path = mmap_dir / f"tensor_{index:08d}_{digest}.bin"

    # A CPU tensor backed by a temporary memory-mapped file. DCP writes into it,
    # and torch.save later serializes from it without requiring one huge RAM allocation.
    storage = torch.UntypedStorage.from_file(str(path), shared=True, nbytes=nbytes)
    tensor = torch.empty(0, dtype=dtype, device="cpu")
    return tensor.set_(storage, 0, size, _contiguous_stride(size))


def _make_empty_tensor(size: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(size, dtype=dtype, device="cpu")


def _load_dcp_flat(
    src: Path,
    keys_and_meta: list[tuple[str, TensorStorageMetadata]],
    mmap_dir: Path | None,
) -> OrderedDict[str, torch.Tensor]:
    flat_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for i, (key, meta) in enumerate(keys_and_meta):
        size = tuple(int(x) for x in meta.size)
        dtype = meta.properties.dtype
        if mmap_dir is not None:
            flat_state[key] = _make_mmap_tensor(mmap_dir, i, key, size, dtype)
        else:
            flat_state[key] = _make_empty_tensor(size, dtype)

    planner = DefaultLoadPlanner(flatten_state_dict=False, allow_partial_load=True)

    # Public API in recent PyTorch. The fallback covers older 2.x releases.
    try:
        dcp.load(state_dict=flat_state, checkpoint_id=str(src), planner=planner, no_dist=True)
    except TypeError:
        from torch.distributed.checkpoint.state_dict_loader import load_state_dict

        load_state_dict(
            flat_state,
            storage_reader=FileSystemReader(str(src)),
            planner=planner,
            no_dist=True,
        )

    return flat_state


def _read_metadata(src: Path):
    reader = FileSystemReader(str(src))
    return reader.read_metadata()


def _print_keys(src: Path, limit: int | None = None) -> None:
    metadata = _read_metadata(src)
    entries = list(metadata.state_dict_metadata.items())

    print(f"DCP directory: {src}")
    print(f"Total metadata entries: {len(entries)}")
    print("")

    shown = 0
    for key, meta in entries:
        if isinstance(meta, TensorStorageMetadata):
            size = tuple(int(x) for x in meta.size)
            dtype = meta.properties.dtype
            nbytes = _tensor_nbytes(size, dtype)
            print(f"{key}\tshape={list(size)}\tdtype={dtype}\tsize={_human_bytes(nbytes)}")
        else:
            print(f"{key}\t<non-tensor: {type(meta).__name__}>")
        shown += 1
        if limit is not None and shown >= limit:
            remaining = len(entries) - shown
            if remaining > 0:
                print(f"... {remaining} more entries. Re-run with a larger --list-limit or 0 for all.")
            break


def convert(args: argparse.Namespace) -> None:
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not _is_dcp_dir(src):
        raise FileNotFoundError(f"{src} is not a DCP directory; expected a .metadata file and *.distcp shards")

    metadata = _read_metadata(src)
    selected: list[tuple[str, TensorStorageMetadata]] = []
    skipped_non_tensor = 0

    for key, meta in metadata.state_dict_metadata.items():
        if not _selected(key, args.include_prefix, args.exclude_prefix):
            continue
        if not isinstance(meta, TensorStorageMetadata):
            skipped_non_tensor += 1
            continue
        selected.append((key, meta))

    if not selected:
        raise RuntimeError(
            "No tensor keys selected. Run with --list-keys and then set --include-prefix, "
            "for example --include-prefix model. or --include-prefix teacher."
        )

    total_bytes = sum(_tensor_nbytes(tuple(m.size), m.properties.dtype) for _, m in selected)
    print(f"Selected tensor keys: {len(selected)}")
    print(f"Selected tensor payload: {_human_bytes(total_bytes)}")
    if skipped_non_tensor:
        print(f"Skipped non-tensor entries: {skipped_non_tensor}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp_holder: tempfile.TemporaryDirectory[str] | None = None
    mmap_dir: Path | None = None
    if args.mmap_dir:
        mmap_parent = Path(args.mmap_dir).expanduser().resolve()
        mmap_parent.mkdir(parents=True, exist_ok=True)
        tmp_holder = tempfile.TemporaryDirectory(prefix="dcp_merge_mmap_", dir=str(mmap_parent))
        mmap_dir = Path(tmp_holder.name)
        print(f"Using temporary mmap directory: {mmap_dir}")

    try:
        flat_state = _load_dcp_flat(src, selected, mmap_dir)

        out_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, tensor in flat_state.items():
            out_key = _strip_key(key, args.strip_prefix)
            if args.contiguous and not tensor.is_contiguous():
                tensor = tensor.contiguous()
            out_state[out_key] = tensor

        to_save = {args.wrap_key: out_state} if args.wrap_key else out_state
        print(f"Saving: {dst}")
        torch.save(to_save, str(dst))
        print(f"Done. Output file size: {_human_bytes(dst.stat().st_size)}")
    finally:
        if tmp_holder is not None:
            tmp_holder.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge PyTorch DCP/distcp checkpoint shards into one .pth file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", required=True, help="DCP checkpoint directory containing .metadata and *.distcp files")
    parser.add_argument("--dst", help="Output .pth path; required unless --list-keys is used")
    parser.add_argument("--list-keys", action="store_true", help="Only print checkpoint keys, shapes and dtypes, then exit")
    parser.add_argument("--list-limit", type=int, default=200, help="Max keys to print with --list-keys; use 0 for all")
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Only load tensor keys starting with this prefix. Can be repeated. If omitted, loads all tensor keys.",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Skip tensor keys starting with this prefix. Can be repeated, e.g. --exclude-prefix optimizer.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Strip this prefix from output tensor keys. Can be repeated.",
    )
    parser.add_argument(
        "--wrap-key",
        default="",
        help="Wrap output state_dict under this top-level key, e.g. teacher produces {'teacher': state_dict}. Empty means no wrapping.",
    )
    parser.add_argument(
        "--mmap-dir",
        default="",
        help="Optional temp directory for mmap-backed tensors to reduce CPU RAM usage. Requires extra disk roughly equal to selected tensor payload.",
    )
    parser.add_argument(
        "--contiguous",
        action="store_true",
        help="Force tensors contiguous before saving. Usually unnecessary for DCP-loaded tensors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    src = Path(args.src).expanduser().resolve()

    if not _is_dcp_dir(src):
        parser.error(f"{src} is not a DCP directory; expected .metadata plus *.distcp files")

    if args.list_keys:
        limit = None if args.list_limit == 0 else args.list_limit
        _print_keys(src, limit=limit)
        return 0

    if not args.dst:
        parser.error("--dst is required unless --list-keys is used")

    # Normalize optional empty wrap key.
    if args.wrap_key == "":
        args.wrap_key = None
    if args.mmap_dir == "":
        args.mmap_dir = None

    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


'''
# 先查看 checkpoint 里有哪些 key
python3 tools/merge_distcp_to_ckpt.py \
    --src work_dirs/finetune/dinov3_vit7b16_lora_finetune/ckpt/10999 --list-keys

# 保留backbone，包含 patch_embed、transformer blocks、norm、cls_token 等
# 
python3 tools/merge_distcp_to_ckpt.py \
    --src work_dirs/finetune/dinov3_vit7b16_lora_finetune/ckpt/10999 \
    --dst work_dirs/finetune/dinov3_vit7b16_lora_finetune/ckpt/finetune_backbone_teacher.pth \
    --include-prefix model.teacher.backbone. \
    --strip-prefix model.teacher.backbone. \
    --wrap-key teacher \
    --mmap-dir /tmp

'''