"""Offline sanity checks for the agent plumbing (no GPU, no okbench, no LLM).

These run anywhere (incl. the Mac) and just verify the pieces wire together.
The real end-to-end check is `python -m anvil.cli smoke` on the GPU server.
"""
import json
from types import SimpleNamespace

from anvil.candidate import Candidate, EvalResult
from anvil.baselines import SMOKE_KERNELS
from anvil.okeval import trim_error
from anvil.agent import AgentRunner


def test_baseline_kernel_has_entry_symbol():
    src = SMOKE_KERNELS["gemm_bf16_nt"]
    assert 'openkernels_launch_gemm_bf16_nt' in src
    assert '__global__' in src
    assert '#include "ops/gemm_bf16_nt/interface.h"' in src


def test_eval_result_from_okbench_parses_speedup_and_correctness():
    fake = {
        "shapes": [
            {"name": "s1", "supported": True, "correct": True,
             "pure_median_ms": 0.20, "pure_over_reference": 1.25,   # 25% slower -> 0.8x
             "correctness": {"max_abs": 0.0}},
            {"name": "s2", "supported": True, "correct": True,
             "pure_median_ms": 0.10, "pure_over_reference": 0.50,   # 2x faster
             "correctness": {"max_abs": 0.0}},
        ]
    }
    r = EvalResult.from_okbench(Candidate(kernel_cu="x"), fake)
    assert r.correct
    # geomean(0.8, 2.0) = 1.264...
    assert abs(r.geomean_speedup - (0.8 * 2.0) ** 0.5) < 1e-6


def test_eval_result_flags_incorrect_shape():
    fake = {"shapes": [
        {"name": "s1", "correct": True, "pure_over_reference": 1.0, "correctness": {"max_abs": 0.0}},
        {"name": "s2", "correct": False, "pure_over_reference": None, "correctness": {"max_abs": 9.9}},
    ]}
    r = EvalResult.from_okbench(Candidate(kernel_cu="x"), fake)
    assert not r.correct
    assert "s2" in r.error


def test_trim_error_surfaces_top_nvcc_error_not_just_tail():
    # nvcc/ptxas print the real cause near the TOP; the bottom is just a count.
    # The old tail-slice ([-N:]) dropped the cause and fed the model the count /
    # okbench's own traceback — this guards that the `error:` line survives.
    top = 'kernel.cu(42): error: identifier "foo" is undefined'
    filler = "\n".join(f"kernel.cu({i}): warning: unused variable v{i}" for i in range(300))
    tail = "2 errors detected in the compilation of kernel.cu."
    out = trim_error(top + "\n" + filler + "\n" + tail, "", limit=600)
    assert "KEY COMPILER ERRORS" in out
    assert 'identifier "foo" is undefined' in out          # the cause survived
    assert len(out) < len(top + filler + tail)             # and it was trimmed


def test_trim_error_unwraps_okbench_compile_runtimeerror():
    # okbench's real compile-failure shape: a python traceback whose last line is
    # RuntimeError(json.dumps({..., "stderr": <nvcc output>, ...})). The nvcc cause
    # must be surfaced and lead; okbench's own traceback frames must be dropped.
    import json
    nvcc = ('submissions/5090/gemm_bf16_nt/v/kernel.cu(51): error: identifier '
            '"foo" is undefined\n      int z = foo;\n              ^\n\n'
            '1 error detected in the compilation of "kernel.cu".\n')
    blob = json.dumps({"plan": {"shell_command": "nvcc -O3 kernel.cu -o k.so"},
                       "returncode": 2, "stderr": nvcc, "stdout": ""}, indent=2)
    tb = ('Traceback (most recent call last):\n'
          '  File ".../okbench/cli.py", line 663, in <module>\n    main()\n'
          '  File ".../okbench/gemm_runner.py", line 457, in _maybe_compile\n'
          '    raise RuntimeError(...)\n'
          'RuntimeError: ' + blob)
    out = trim_error("", tb, limit=4000)
    assert "KEY COMPILER ERRORS" in out
    assert 'identifier "foo" is undefined' in out
    assert "Traceback (most recent call last)" not in out          # okbench noise gone
    assert out.index('identifier "foo"') < out.index("compile command")


def test_trim_error_surfaces_traceback_when_no_compiler_error():
    # okbench crashed before nvcc: no `error:` lines, so just head+tail the log
    # (the head is where the real traceback / cause lives).
    log = "okbench: deploying submission to platform sm120\n" + ("step\n" * 800) + "done"
    out = trim_error(log, "", limit=400)
    assert "KEY COMPILER ERRORS" not in out
    assert "okbench: deploying submission" in out          # head preserved
    assert "[middle trimmed]" in out


# --- agent (Route-AVO-lite) tool-loop, offline (fake client + fake runner) ----

def _fake_op():
    return SimpleNamespace(
        name="gemm_bf16_nt", description="C=A·Bᵀ", entry_symbol="sym",
        atol=1e-2, rtol=1e-2, shapes=[{"name": "s1", "m": 16, "n": 16, "k": 16}],
        interface_h="void sym();", reference_src="def ref(): pass")


def _correct_eval(cand):
    return EvalResult.from_okbench(cand, {"shapes": [
        {"name": "s1", "correct": True, "pure_over_reference": 2.0,   # 2x slower ref -> 0.5x
         "pure_median_ms": 0.10, "correctness": {"max_abs": 0.0}}]})


class _FakeRunner:
    def __init__(self): self.calls = []
    def evaluate(self, cand, variant):
        self.calls.append(variant)
        return _correct_eval(cand)


def _scripted_client(messages):
    """OpenAI-compat client stand-in: returns the scripted assistant messages in order."""
    seq = iter(messages)
    def create(**_kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(seq))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _tool_call(cid, **args):
    return SimpleNamespace(id=cid, function=SimpleNamespace(
        name="bench_kernel", arguments=json.dumps(args)))


def test_agent_loop_drives_bench_tool_then_stops_and_tracks_best():
    # turn 1: model calls bench_kernel; turn 2: model replies without a tool call -> stop.
    client = _scripted_client([
        SimpleNamespace(content="trying v1",
                        tool_calls=[_tool_call("c1", kernel_cu="#include <x>\nvoid sym(){}", notes="v1")]),
        SimpleNamespace(content="done, 0.5x", tool_calls=None),
    ])
    runner = _FakeRunner()
    report = AgentRunner(_fake_op(), runner, client=client, max_attempts=4, verbose=False).run()

    assert runner.calls == ["gucheng_a01"]                 # exactly one bench happened
    assert report.best is not None
    assert abs(report.best.geomean_speedup - 0.5) < 1e-9   # best tracked from the tool result
    assert len(report.history) == 1


def test_agent_loop_respects_attempt_budget():
    # model keeps calling bench_kernel forever; the runner must be capped at max_attempts.
    forever = [SimpleNamespace(content="again",
                               tool_calls=[_tool_call(f"c{i}", kernel_cu="#include <x>", notes=str(i))])
               for i in range(10)]
    runner = _FakeRunner()
    AgentRunner(_fake_op(), runner, client=_scripted_client(forever),
                max_attempts=3, target_speedup=99.0, verbose=False).run()
    assert len(runner.calls) == 3                            # budget enforced
