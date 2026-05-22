# llmCLI — Stage-Axis Decomposition Audit

**Date:** 2026-05-20
**Auditor:** Claude (read-only)
**Reference framework:** lyra #1277 stage-axis decomposition strategy
**Source doc:** `~/projects/lyra/artifacts/analyses/1277-stage-axis-refactor-strategy.mdx`

---

## Verdict Summary

llmCLI is **not** exhibiting the N×M cascade. The codebase is small (2 741 LOC total), the per-target engine files are thin (113–150 LOC each), concerns are already extracted horizontally (`_common.py`, `gpu.py`, `GenerationMixin`), and the one historical sibling-fix instance (`waitpid` zombie fix applied to `vllm.py`, then carried to `llamacpp.py` via commit `0a528a7`) was caught and consolidated within the same working session.

Three latent risks exist:
1. `pynvml` is owned by both `gpu.py` and `nats/llm_adapter.py` — a second GPU capability will duplicate again.
2. Daemon error propagation erases exception types at two bus boundaries (AF_UNIX wire).
3. `_ENGINE_REGISTRY` is duplicated between `engines/__init__.py` and `daemon.py`.

None of these is actively cascading yet; all are pre-cascade. **Stage-axis pivot is not warranted at current scale**, but the pynvml and registry duplications are worth consolidating now before the 4th engine arrival.

---

## Section 1 — Axis of Decomposition

### Module map

```
src/llmcli/
  engine.py              44 LOC  — Protocol (Engine) + EngineInstance dataclass
  engines/
    _common.py           81 LOC  — shared: _wait_ready() helper (health poll loop)
    __init__.py          28 LOC  — registry dict + get_engine() factory
    llamacpp.py         150 LOC  — target: vanilla llama-server wrapper
    llamacpp_tq3.py      16 LOC  — target: TurboQuant fork (binary override only)
    vllm.py             113 LOC  — target: vllm serve wrapper
  config.py             208 LOC  — stage: catalog loader, TOML parse, VRAM budget
  daemon.py             265 LOC  — stage: AF_UNIX socket server, engine dispatch
  gpu.py                221 LOC  — stage: VRAM probe + VRAMSampler + KV overhead
  litellm_config.py     190 LOC  — stage: LiteLLM config block builder/writer
  providers.py           22 LOC  — stage: provider registry (pure data)
  cli/                          — stage: Typer CLI commands (9 files, 880 LOC)
  nats/                         — stage: NATS adapter (3 files, 430 LOC)
```

### Per-engine concerns (engines/*.py)

| Concern | llamacpp.py | vllm.py | llamacpp_tq3.py |
|---|---|---|---|
| HF cache path resolution | yes (`_hf_hub_root`, `_repo_to_dir_name`, `_gguf_path`) | no (uses `spec.repo`) | no (inherits) |
| Command builder | yes (`_build_cmd`) | yes (`_build_cmd`) | inherited |
| Subprocess spawn | yes | yes | inherited |
| Health-poll loop | → delegated to `_common._wait_ready` | → delegated | inherited |
| Process-group kill | `os.kill(pid)` only | `os.killpg(pgid)` (vLLM workers) | inherited |
| SIGTERM→SIGKILL escalation | yes, shared pattern | yes, shared pattern | inherited |
| Zombie reaping (`waitpid`) | yes | yes | inherited |
| Binary availability guard | no (FileNotFoundError from OS) | yes (`shutil.which`) | inherited |
| Wait timeout constant | `_WAIT_TIMEOUT = 60` | `_WAIT_TIMEOUT = 180` | inherited |

### How concerns are shared

- **Health poll loop** extracted to `_common._wait_ready` via `0a528a7` (2026-04-28). Deliberate horizontal extraction triggered by the first duplication.
- **TQ3 engine** = pure single-class inheritance (`LlamaCppTQ3Engine(LlamaCppEngine)`) with exactly one class-attribute override (`binary`). Zero logic duplication. Correct pattern.
- **VLLMEngine** = standalone class. Shares structural pattern but re-implements `stop()` with `os.killpg` (process group required for vLLM workers) — semantic difference, not duplication.

**Verdict: mixed-axis, trending toward stage-axis.** Horizontal extractions (`_common.py`, `gpu.py`, `GenerationMixin`) are present and deliberate. Residual per-target accretion: `_ENGINE_REGISTRY` dup + pynvml lifecycle dup (see Section 5).

---

## Section 2 — Cascade Symptoms

### `f"...{exc}"` patterns — 11 sites

| File | Line | Pattern | Bus? |
|---|---|---|---|
| `daemon.py:197` | `return f"ERR vram budget exceeded: {exc}"` | **yes** — AF_UNIX wire |
| `daemon.py:215` | `return f"ERR swap failed: {exc}"` | **yes** — AF_UNIX wire |
| `nats/llm_adapter.py:111` | `raise RuntimeError(f"llmCLI daemon SWAP failed: {reply}")` | **yes** — across NATS/daemon boundary |
| `cli/swap.py:35` | Rich display | CLI only |
| `cli/proxy.py:67,72,82,84,118` | Rich display | CLI only |
| `cli/lifecycle.py:46,74,90` | Rich display | CLI only |

**Bus-bound exception leaks: 3 sites.** They erase exception type before crossing a protocol boundary.

### `str(exc)` in WorkerError payloads — 4 sites

`nats/_generation.py:99,112,116,120` — all guarded with `or "fallback"`, mitigated but still type-erasing. `WorkerError.message` carries raw exception `str()` over NATS.

### `except Exception` count — 24 total

| File | Count |
|---|---|
| `gpu.py` | 6 |
| `daemon.py` | 5 |
| `nats/llm_adapter.py` | 3 |
| `nats/_generation.py` | 3 |
| `engines/_common.py` | 2 |
| `engines/llamacpp.py` | 1 |
| `engines/vllm.py` | 1 |
| `cli/*.py` | 3 |

16 sites are annotated with `noqa: BLE001` (GPU probe fallbacks or NATS error-fence — intentional). The **4 daemon `except Exception` at lines 80, 85, 103, 106** are unannotated and broad — flag.

### Duplicated private helpers

- `_build_cmd` in both `llamacpp.py:76` and `vllm.py:36` — same name, **different bodies** (semantically required). Not duplication.
- `_check_health` / `_start_process` / `_pick_port` — none duplicated, consolidated to `_common.py`.

### `__init_subclass__` / VALIDATE_* class-attr overrides

**None found.** Only one class-attr override: `binary: str` on `LlamaCppTQ3Engine` — appropriate mechanism.

---

## Section 3 — Quantitative

### LOC per file — 0 files over 300-line gate

```
bench.py       282   ← 18 lines from gate
daemon.py      265
proxy.py       255
llm_adapter.py 222
gpu.py         221
_generation.py 207
litellm_config 190
llamacpp.py    150
lifecycle.py   116
vllm.py        113
```

### Folder file counts — `src/llmcli/` exactly at 12-file gate

| Folder | Files |
|---|---|
| `src/llmcli/` | **12 (at gate)** |
| `src/llmcli/cli/` | 9 |
| `src/llmcli/engines/` | 6 |
| `src/llmcli/nats/` | 4 |

One more top-level module triggers the gate.

### Sibling-fix signatures in git history

| Commit | Date | Fix | Files |
|---|---|---|---|
| `17bfae4` | 2026-04-28 | `waitpid` zombie fix in `stop()` | `vllm.py` only |
| `39a9c94` | 2026-04-19 | zombie reap on startup failure | `llamacpp.py` only |
| `0a528a7` | 2026-04-28 | extract `_wait_ready` to `_common.py` | both engines |

**One detected sibling-fix event, caught and consolidated at N=2 within same session.** No ongoing cascade.

---

## Section 4 — DEBT Inventory

### `artifacts/debt/` directory

**Does not exist.** No structured debt tracking. Latent risks live only in code comments or implicitly in ADRs.

### Inline debt annotations

`grep -rn 'TODO|FIXME|XXX|HACK|DEBT' src/` → **0 results.** Zero inline debt markers across 2 741 LOC.

### `noqa` annotations — 18 total, all documented

- 16× `noqa: BLE001` — broad exception in probe/fallback paths
- 1× `noqa: S603` — `subprocess.Popen` security suppression
- 1× `noqa: S101` — `assert` (litellm_config.py:170, sentinel-guarded)

### DEBT slugs from lyra #1277

| Slug | Present? | Evidence |
|---|---|---|
| `boundary-broad-catch` | **partial** | `daemon.py:80,85,103,106` — bare `except Exception` without `noqa` on serve loop |
| `wiring-bootstrap-deps` | no | `cli/__init__.py` re-export is intentional |
| `defensive-narrow-payloads` | **partial** | `_generation.py:49` — `safe_payload` filters request_id/trace_id |
| `complexity-residual` | no | No files near 300 LOC |
| `re-export-init` | **present** | `cli/__init__.py:19-21` — 4 re-exports for test patching |
| `protocol-private-ducktyping` | no | `Engine` is typed `Protocol`; `GenerationMixin` documents required attrs |
| `module-level-patch-fixtures` | no | Not found |
| `adapter-magic-constants` | **partial** | `_WAIT_TIMEOUT` per-engine (semantically required) |
| `adapter-dispatch-complexity` | **partial** | `_ENGINE_REGISTRY` duplicated (see §5) |

---

## Section 5 — Composition vs Inheritance for Capabilities

### Health check — **duplicated verbatim**

`llamacpp.py:117-123` ≡ `vllm.py:79-85` — identical 7 lines:

```python
def health(self, instance: EngineInstance) -> bool:
    try:
        resp = httpx.get(f"{instance.base_url}/health", timeout=2.0)
        return resp.status_code < 300
    except Exception:  # noqa: BLE001
        return False
```

At N=2 harmless. At N=4 a bug requires N edits.

### Process supervision (stop/SIGTERM/SIGKILL/waitpid)

Identical structure, semantic divergence (`os.kill` vs `os.killpg`). ~24 lines per file. Extractable as parameterized helper but not urgent.

### Retry / timeout / circuit-breaker

**None present.** No retry loops anywhere. NATS layer marks errors `retryable=True/False` and delegates retry to caller. Correct.

### Port assignment

No port-pick logic — ports statically declared in `ModelSpec.port` (catalog-driven). Correct.

### Engine registry — **duplicated**

`_ENGINE_REGISTRY` exists in 2 modules:
- `engines/__init__.py:7-11` — used by `get_engine()`, consumed by `bench.py`
- `daemon.py:161-165` — local dict inside `_engine_for_spec`, same keys/values

Out of sync risk: `daemon.py` has `engine=remote` guard logic; `__init__.py` does not. A 4th engine requires two edits.

### Engine Protocol

`engine.py:39-44` defines `Engine` as typed `Protocol` (3 lifecycle verbs: `start/stop/health`). **Not** ABC with abstract methods. Protocol enforcement via pyright at type-check time. **Does not** encode a stage pipeline — appropriate for current 3-engine surface.

---

## Section 6 — Result[T,E] vs Raising

### Engine boundary — correctly typed exceptions

Engines raise: `FileNotFoundError` (model not pulled), `RuntimeError` (startup timeout), `ImportError` (vLLM not installed). No `RuntimeError(str(exc))` type-erasure.

### Daemon boundary (AF_UNIX wire) — **stringly-typed**

```python
# daemon.py:214
except Exception as exc:
    return f"ERR swap failed: {exc}"
# daemon.py:197
except ValueError as exc:
    return f"ERR vram budget exceeded: {exc}"
```

Wire protocol: plaintext `OK ...` / `ERR ...`. Type info lost. Client (`llm_adapter.py:111`) matches `startswith("OK")` only. `FileNotFoundError` (no GGUF) and `RuntimeError` (boot timeout) produce identical wire-level structure.

### NATS boundary — well-structured

`WorkerError(code=, message=, retryable=)` at `_generation.py:45-46`. Typed dot-notation `code` field (`"worker.timeout"`, `"upstream.5xx"`). `message` is `str(exc) or "fallback"` — type-erasing but structured by code. Adequate.

### Custom exception types

**Zero.** All errors are stdlib (`RuntimeError`, `ValueError`, `FileNotFoundError`, `ImportError`). Intentional at current scale.

---

## Section 7 — Cross-Repo Coupling with Lyra

### pyproject.toml dependency

```toml
[project.optional-dependencies]
nats = [
    "nats-py>=2.6,<3",
    "nkeys>=0.1",
    "roxabi-nats",
    "roxabi-contracts",
]
```

Both `roxabi-nats` and `roxabi-contracts` are `branch = "staging"` git sources. **Declared directly** (unlike voiceCLI/imageCLI which only declare `roxabi-nats`).

### NATS subjects — **owned upstream**

Subjects are **not** defined in llmCLI. Consumed from `roxabi_contracts.llm.SUBJECTS` (`llm_adapter.py:28`). Dependency arrow points llmCLI → contracts. Correct — **llmCLI sets the model for the other 2 projects.**

### NATS codec

Entirely from `roxabi_contracts`: `build_llm_chunk`, `build_llm_response`, `LlmChunkEvent`, `LlmResponse`, `WorkerError` (`_generation.py:20-23`). Zero codec duplication.

### pynvml duplication

`gpu.py` has `VRAMSampler` (lines 55-137) with full nvml lifecycle. `llm_adapter.py:132-179` re-implements nvml init/handle/shutdown for heartbeat efficiency (cached handle). Both call `pynvml.nvmlInit()`, `nvmlDeviceGetHandleByIndex(0)`, `nvmlShutdown()`.

Valid optimization, but **if voiceCLI/imageCLI grow similar adapters this triplicates.**

---

## Section 8 — Findings + Recommendations

### Finding 1 — `health()` duplicated verbatim (pre-cascade, latent)

**Evidence:** `engines/llamacpp.py:117-123` ≡ `engines/vllm.py:79-85` — identical 7 lines.
**Status:** Pre-cascade. Harmless at N=2. At N=4 a bug requires N edits.
**Recommendation:** Move to `_common.py` as `_default_health(base_url: str) -> bool`. Net: -6 LOC.

### Finding 2 — `_ENGINE_REGISTRY` duplicated (pre-cascade, latent)

**Evidence:** `engines/__init__.py:7-11` and `daemon.py:161-165` define identical dict.
**Status:** Pre-cascade. 4th engine requires two edits in two files.
**Recommendation:** `daemon._engine_for_spec()` delegates to `engines.get_engine(spec)`, catches `ValueError`. `remote` guard stays in daemon (daemon-domain logic).

### Finding 3 — AF_UNIX protocol has no typed error codes (pre-cascade, structural)

**Evidence:** `daemon.py:197,215` — all engine exceptions collapse to `f"ERR {exc}"`. Client matches `startswith("OK")`.
**Status:** Pre-cascade. Maintenance burden if routing logic needs to differentiate (e.g., retry only on timeout, not on GGUF-not-found).
**Recommendation:** Introduce `ERR.<code> <message>`, e.g., `ERR.swap_timeout ...`, `ERR.vram_exceeded ...`. 5 LOC change in `daemon.py`, 3 LOC change in `llm_adapter.py`. Same plaintext protocol, additive.

### Finding 4 — pynvml lifecycle duplicated (latent)

**Evidence:** `gpu.py` `VRAMSampler` (55-137) + `llm_adapter.py` (132-179) both manage nvml.
**Status:** Latent. Triplicates if voiceCLI/imageCLI add similar adapters.
**Recommendation:** Expose `VRAMMonitor` API from `gpu.py` returning `(free_gib, used_gib)` from cached handle. `llm_adapter.py` consumes it.

### Finding 5 — `src/llmcli/` at 12-file folder gate (latent, process risk)

**Evidence:** `ls src/llmcli/ | wc -l` → 12 (at gate). `tools/folder_exemptions.txt` is empty.
**Status:** Latent. Next top-level module triggers gate.
**Recommendation:** Proactively create `src/llmcli/support/` and move `providers.py` + `litellm_config.py` there. 2-3 module headroom.

---

## Top 3 Actions

1. **Consolidate `health()` + `_ENGINE_REGISTRY`** (Findings 1 + 2). Pre-cascade triggers guaranteed to fire at engine #4. Single PR, ~30 LOC net reduction, zero behavior change.

2. **Typed error codes on AF_UNIX protocol** (Finding 3). Prevents wire from becoming stringly-typed black hole. 8 LOC total across `daemon.py` and `llm_adapter.py`. Low risk, high future value.

3. **Move `providers.py` + `litellm_config.py` to `src/llmcli/support/`** (Finding 5). Buys 2 folder-gate slots before next violation. Pre-emptive.

---

## Cross-repo notes

- **llmCLI is the reference implementation** for the worker-NATS pattern: it declares `roxabi-contracts` as direct dep + imports subject strings from `roxabi_contracts.llm.SUBJECTS` (no magic constants). voiceCLI + imageCLI should converge.
- llmCLI engine count (3) gives lowest cascade pressure of the 3 projects. Don't pivot to stage-axis yet — pivot when N≥5 or quant-variant pressure arrives.
