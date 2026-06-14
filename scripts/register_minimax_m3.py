#!/usr/bin/env python3
"""Register MiniMax-M3 into a vLLM install (idempotent, line-anchored).

Run inside the build container after copying patches/ onto the vLLM tree.
Adds the architecture + parser entries that the upstream tree does not ship.

Each entry is inserted immediately after a known stock sibling line, so it
lands in the correct registry dict regardless of how the dicts are ordered
in the base image. Safe to re-run: entries already present are skipped.
"""
from __future__ import annotations
import sys
from pathlib import Path

VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")

# (relative path, [fallback anchors tried in order], block, marker_to_skip_if_present)
EDITS: list[tuple[str, list[str], str, str]] = [
    (
        "model_executor/models/registry.py",
        ['"MiniMaxM2ForCausalLM"', '"MiniMaxText01ForCausalLM"'],  # _TEXT_GENERATION_MODELS
        '\n    "MiniMaxM3SparseForCausalLM": (\n'
        '        "vllm.models.minimax_m3",\n'
        '        "MiniMaxM3SparseForCausalLM",\n'
        '    ),',
        '"MiniMaxM3SparseForCausalLM"',
    ),
    (
        "model_executor/models/registry.py",
        ['"MiniMaxVL01ForConditionalGeneration"', '"LlavaForConditionalGeneration"'],  # _MULTIMODAL_MODELS
        '\n    "MiniMaxM3SparseForConditionalGeneration": (\n'
        '        "vllm.models.minimax_m3",\n'
        '        "MiniMaxM3SparseForConditionalGeneration",\n'
        '    ),',
        '"MiniMaxM3SparseForConditionalGeneration"',
    ),
    (
        "model_executor/models/registry.py",
        ['"MiMoMTPModel"', '"DeepSeekV4MTPModel"', '"DeepSeekMTPModel"'],  # _SPECULATIVE_DECODING_MODELS
        '\n    "MiniMaxM3MTP": ("vllm.models.minimax_m3", "MiniMaxM3MTP"),',
        '"MiniMaxM3MTP"',
    ),
    (
        "reasoning/__init__.py",
        ['"minimax_m2":', '"minimax":'],
        '\n    "minimax_m3": (\n'
        '        "minimax_m3_reasoning_parser",\n'
        '        "MiniMaxM3ReasoningParser",\n'
        '    ),',
        '"minimax_m3":',
    ),
    (
        "tool_parsers/__init__.py",
        ['"minimax_m2":', '"minimax":'],
        '\n    "minimax_m3": (\n'
        '        "minimax_m3_tool_parser",\n'
        '        "MinimaxM3ToolParser",\n'
        '    ),',
        '"minimax_m3":',
    ),
]


def insert_anchored(text: str, anchors: list[str], block: str) -> tuple[str, str] | None:
    """Insert `block` immediately before the first matching anchor line.

    Inserting *before* the anchor (rather than after) is what makes this
    robust to multi-line dict values: the anchor is always the start of a
    complete, well-formed entry, so we land cleanly between two entries and
    the dict stays syntactically valid whether the sibling is single-line
    (`"k": ("m", "C"),`) or multi-line (`"k": (\n ...\n ),`).
    """
    for anchor in anchors:
        idx = text.find(anchor)
        if idx >= 0:
            # back up to the start of the line containing the anchor
            bol = text.rfind("\n", 0, idx)
            bol = 0 if bol < 0 else bol + 1
            return text[:bol] + block + "\n" + text[bol:], anchor
    return None


def register(rel: str, anchors: list[str], block: str, marker: str) -> bool:
    path = VLLM_ROOT / rel
    if not path.exists():
        print(f"  [MISS] {rel}: file not found", file=sys.stderr)
        return False
    text = path.read_text()
    if marker in text:
        print(f"  [skip] {rel}: {marker!r} already present")
        return True
    result = insert_anchored(text, anchors, block)
    if result is None:
        print(f"  [WARN] {rel}: none of {anchors} found — skipping. "
              f"Base differs; insert manually.", file=sys.stderr)
        return False
    updated, used = result
    path.write_text(updated)
    print(f"  [ok]   {rel}: inserted {marker!r} before {used!r}")
    return True


def main() -> int:
    if not VLLM_ROOT.exists():
        print(f"vLLM root not found: {VLLM_ROOT}", file=sys.stderr)
        return 1
    ok = True
    for rel, anchors, block, marker in EDITS:
        ok = register(rel, anchors, block, marker) and ok
    if not ok:
        print("\nSome entries could not be inserted automatically (see above).", file=sys.stderr)
        return 2
    print("\nMiniMax-M3 registration complete.")
    print('Verify: python3 -c "import vllm.model_executor.models.registry as R;'
          'print(any(\'MiniMaxM3\' in str(k) for k in R._VLLM_MODELS))"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
