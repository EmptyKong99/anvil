"""Load an OpenKernels op definition (the '题面') from an OpenKernels repo.

We don't redefine the spec — we read it straight from `ops/<op>/` so the agent
always targets exactly what okbench will judge against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# op name -> the okbench CLI subcommand that runs its stable-ABI benchmark.
# Only gemm_bf16_nt is wired up for now (torch reference, no submodule needed).
OKBENCH_BENCH_CMD = {
    "gemm_bf16_nt": "bench-gemm-bf16",
}


@dataclass
class Op:
    name: str
    repo_root: Path
    entry_symbol: str               # the extern "C" launcher the kernel must define
    description: str
    interface_h: str                # contents of ops/<op>/interface.h (the ABI)
    reference_src: str              # contents of ops/<op>/reference.py (ground truth)
    shapes: list[dict] = field(default_factory=list)   # required_5 shapes
    atol: float = 1e-2
    rtol: float = 1e-2

    @property
    def bench_cmd(self) -> str:
        try:
            return OKBENCH_BENCH_CMD[self.name]
        except KeyError:
            raise NotImplementedError(
                f"op {self.name!r} not wired into Anvil yet; supported: "
                f"{list(OKBENCH_BENCH_CMD)}"
            )

    def shape_names(self) -> list[str]:
        return [s["name"] for s in self.shapes]


def load_op(repo_root: str | Path, name: str) -> Op:
    repo_root = Path(repo_root).expanduser().resolve()
    op_dir = repo_root / "ops" / name
    if not op_dir.exists():
        raise FileNotFoundError(f"no such op dir: {op_dir}")

    op_yaml = yaml.safe_load((op_dir / "op.yaml").read_text())
    tol = yaml.safe_load((op_dir / "tolerance.yaml").read_text()).get("default", {})
    shapes_doc = yaml.safe_load((op_dir / "shapes.yaml").read_text())
    suite = shapes_doc["required_suite"]
    shapes = list(shapes_doc["suites"][suite]["shapes"])

    return Op(
        name=name,
        repo_root=repo_root,
        entry_symbol=str(op_yaml.get("entry_symbol", "")),
        description=str(op_yaml.get("description", "")),
        interface_h=(op_dir / "interface.h").read_text(),
        reference_src=(op_dir / "reference.py").read_text(),
        shapes=shapes,
        atol=float(tol.get("allclose_atol", 1e-2)),
        rtol=float(tol.get("allclose_rtol", 1e-2)),
    )
