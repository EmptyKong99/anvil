"""Layered wiki-bundle loader for skill-injection ablations (EXP-004+).

Reads the REAL forge wiki cards (single source of truth, no hand-curated copy) and
assembles them into a system-prompt bundle by *level*, so an ablation can measure
the marginal value of each knowledge layer:

  none        -> ""                              (model's bare ability)
  facts       -> facts/*.md                      (verified exact recipes)
  heuristics  -> facts/ + heuristics/*.md         (+ regime->technique judgment)
  full        -> facts/ + heuristics/ + menu/*.md (+ breadth of what exists)

The cards are injected near-verbatim (only wiki-machinery noise is stripped:
`[[links]]` and the trailing `## Cross-refs` section). We deliberately do NOT
re-summarize them here — the ablation must test the wiki artifact itself, not a
second-order paraphrase.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# subdir stack per level (cumulative)
LEVELS: dict[str, list[str]] = {
    "none": [],
    "facts": ["facts"],
    "heuristics": ["facts", "heuristics"],
    "full": ["facts", "heuristics", "menu"],
}

# anvil/ and forge/ are sibling repos: .../<root>/{anvil,forge}
_DEFAULT_WIKI = Path(__file__).resolve().parents[2] / "forge" / "wiki" / "ptx"

_BUNDLE_HEADER = (
    "KNOWLEDGE BASE — distilled from our OWN kernels and verified by okbench on "
    "RTX 5090 (sm_120). These are facts/heuristics we measured, not latent guesses. "
    "Use them; okbench remains the only correctness oracle.\n"
)


def _clean(text: str) -> str:
    """Strip wiki-machinery that is noise to a kernel-writing model."""
    out = []
    for line in text.splitlines():
        if line.strip().startswith("## Cross-refs"):
            break                       # drop the trailing cross-ref section
        out.append(line.replace("[[", "").replace("]]", ""))
    return "\n".join(out).rstrip()


def load_bundle(
    level: str,
    wiki_dir: Path | str | None = None,
    exclude: Iterable[str] | None = None,
) -> str:
    """Assemble the injected knowledge bundle for `level`.

    `exclude` is a set of card filenames (e.g. {"flash-attention-forward.md"}) to
    drop from the bundle. This is how EXP-006 isolates an op-specific card: arm B =
    facts MINUS the FA card (generic instruction primitives only), arm C = facts
    with it. Matching is by md.name, so a name not present is simply a no-op.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown skill level {level!r}; choose {sorted(LEVELS)}")
    subdirs = LEVELS[level]
    if not subdirs:
        return ""
    root = Path(wiki_dir) if wiki_dir is not None else _DEFAULT_WIKI
    if not root.is_dir():
        raise FileNotFoundError(f"wiki dir not found: {root} (pass --wiki-dir)")
    drop = set(exclude or ())

    parts: list[str] = [_BUNDLE_HEADER]
    for sub in subdirs:
        for md in sorted((root / sub).glob("*.md")):
            if md.name in drop:
                continue
            body = _clean(md.read_text())
            if body:
                parts.append(f"===== {sub}/{md.name} =====\n{body}")
    return "\n\n".join(parts)
