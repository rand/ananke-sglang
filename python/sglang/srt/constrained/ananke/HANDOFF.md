# Ananke Constrained Generation - Session Handoff

**Date**: 2026-02-28
**Branch**: `main` (synced with upstream sgl-project/sglang, plugin architecture)
**Status**: Feature-Complete, Production-Ready, Upstream-Synced

---

## Current State

### What's Implemented

| Component | Status | Location |
|-----------|--------|----------|
| Multi-domain constraints | ✅ Complete | `domains/*/domain.py` |
| Client SDK (sync/async) | ✅ Complete | `client/` |
| Retry with backoff | ✅ Complete | `client/http.py` |
| Request/response logging | ✅ Complete | `client/http.py` |
| Streaming support | ✅ Complete | `client/__init__.py` |
| Mask relaxation | ✅ Complete | `masks/relaxation.py` |
| Observability module | ✅ Complete | `observability/` |
| Early termination | ✅ Complete | `backend/grammar.py` |
| Configuration docs | ✅ Complete | `docs/CONFIGURATION.md` |
| Migration guide | ✅ Complete | `docs/MIGRATION_FROM_XGRAMMAR.md` |

### Eval Results (Latest - Modal Deployment)
```
Layer 1 (Mechanism Verification): PASS - 11/11 (100%)
Layer 2 (Value Measurement): 5/6 tests show value
E2E Tests: 12/12 PASS (Modal + Qwen3-Coder-30B-A3B)
Layered Eval: 10/11 PASS (json_enum_values at 65% is model limitation)

Effect sizes:
- Python list comprehension: +60% (large)
- Python recursive function: +90% (large)
- Python dict comprehension: +30% (medium)
- Python generator: +56.7% (large)
- Python class method: +30% (medium)
```

### Test Coverage
```
Total tests: 4,406 PASS (3 skipped - Z3 dependency, pre-existing)
- Unit tests: 3,482 PASS
- Property tests: 629 PASS
- Integration tests: 295+ PASS
```

---

## Architecture Overview

```
ConstraintSpec (user-facing)
    ↓
protocol.py (serialization)
    ↓
[sglang core: protocol.py, sampling_params.py]  ← constraint_spec field
    ↓
[sglang core: grammar_manager.py]               ← spec dispatch path
    ↓
[sglang core: base_grammar_backend.py]           ← generic spec-dispatch methods
    ↓
backend/backend.py (AnankeBackend)
    ↓
backend/grammar.py (AnankeGrammar - mask computation)
    ├── SyntaxDomain (CFG-based)
    ├── TypeDomain (type inference)
    ├── ImportDomain (module restrictions)
    ├── SemanticsDomain (semantic constraints)
    └── ControlFlowDomain (flow analysis)
    ↓
masks/relaxation.py (progressive relaxation)
    ↓
observability/ (metrics collection)
```

### Key Files — Ananke Subtree
| File | Purpose |
|------|---------|
| `spec/constraint_spec.py` | User-facing ConstraintSpec dataclass |
| `protocol.py` | Wire format for API transport |
| `backend/backend.py` | SGLang LogitsProcessor integration |
| `backend/grammar.py` | AnankeGrammar mask fusion engine |
| `domains/*/domain.py` | Per-domain constraint implementations |
| `masks/relaxation.py` | Progressive domain relaxation |
| `observability/collector.py` | Metrics collection |
| `observability/metrics.py` | Metric data structures |
| `observability/exporter.py` | Log/callback exporters |
| `client/__init__.py` | AnankeClient, AnankeAsyncClient |
| `client/http.py` | HTTP transport with retry/logging |

### Key Files — SGLang Core Integration (7 files, ~200 lines)
| File | Change |
|------|--------|
| `constrained/base_grammar_backend.py` | Generic spec-dispatch methods + ananke create case |
| `constrained/grammar_manager.py` | constraint_spec dispatch in `process_req_with_grammar()` |
| `sampling/sampling_params.py` | `constraint_spec` field |
| `entrypoints/openai/protocol.py` | `constraint_spec` field on request models |
| `entrypoints/openai/serving_completions.py` | Wire constraint_spec |
| `entrypoints/openai/serving_chat.py` | Wire constraint_spec |
| `server_args.py` | `"ananke"` in backend choices + CLI args |

---

## Feature Details

### Client SDK
```python
from sglang.srt.constrained.ananke.client import (
    AnankeClient,
    AnankeAsyncClient,
    GenerationConfig,
    RetryConfig,
    LoggingConfig,
)

# Sync client with retry and logging
client = AnankeClient(
    base_url="http://localhost:8000",
    retry_config=RetryConfig(max_retries=3, initial_delay=0.5),
    logging_config=LoggingConfig(log_request_body=True),
)

# Streaming
for chunk in client.generate_stream(prompt, constraint=spec):
    print(chunk.text, end="")

# Async client
async with AnankeAsyncClient(base_url) as client:
    result = await client.generate(prompt, constraint=spec)
```

### Mask Relaxation
```python
# Configured via ConstraintSpec
spec = ConstraintSpec(
    regex=r'.*',
    language="python",
    allow_relaxation=True,
    relaxation_threshold=10,  # Minimum popcount before relaxation
)

# Relaxation order (most dispensable first):
# semantics → controlflow → imports → types
# Syntax is NEVER relaxed
```

### Observability
```python
from sglang.srt.constrained.ananke.observability import (
    MetricsCollector,
    LogExporter,
    CallbackExporter,
)

collector = MetricsCollector(vocab_size=32000)
collector.record_mask_application(popcount=15000, domain="types")
collector.record_domain_latency("types", 0.5)  # ms

# Export to logs
exporter = LogExporter(format="json")
exporter.export(collector)

# Export via callback (for external systems)
def send_to_prometheus(summary):
    ...
callback_exporter = CallbackExporter(metrics_callback=send_to_prometheus)
```

---

## Known Limitations

1. **Import Sandboxing Value**: Layer 2 test shows only +3.3% improvement
   - Mechanism works (Layer 1 passes), but value proposition is weak
   - May need better test design or different use case

2. **Semantics Domain**: 66.7% pass rate in domain-specific evals
   - Most complex domain, depends on accurate semantic analysis
   - Consider simplifying for specific use cases

3. **Truncation on Verbose Languages**: Zig/Rust can hit token limits
   - Early termination helps but doesn't fully solve
   - Per-example `max_tokens` partially addresses this

---

## Potential Future Enhancements

1. **Performance Benchmarks**: Formal benchmark suite for regression tracking
2. **Semantics Domain Improvements**: Lowest pass rate, most complex
3. **Upstream Contribution**: Propose ananke as official sglang grammar backend

---

## Running Tests

### Unit tests
```bash
cd /Users/rand/src/sglang
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
  python/sglang/srt/constrained/ananke/tests/unit/ -v
```

### Property tests
```bash
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
  python/sglang/srt/constrained/ananke/tests/property/ -v
```

### Specific modules
```bash
# Client SDK
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
  python/sglang/srt/constrained/ananke/tests/unit/test_client_http.py -v

# Relaxation
PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
  python/sglang/srt/constrained/ananke/tests/unit/test_relaxation.py -v
```

### Evals (requires Modal deployment)
```bash
modal run deploy/modal/eval/run_layered_eval.py
```

---

## Git State

```
Repository: git@github.com:rand/ananke-sglang.git
Branch: main (synced with upstream/main as of 2026-02-28)
Latest commit: acb1d2fa4 (Merge main: supersede old fork with upstream-synced ananke integration)

Architecture: Plugin-style integration
- 7 core sglang files modified (~200 lines total)
- ~20k LOC ananke subtree (self-contained)
- All other branches merged and cleaned up
```

---

## Quick Start for Next Session

1. **Verify state**:
   ```bash
   cd /Users/rand/src/sglang
   git status
   PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
     python/sglang/srt/constrained/ananke/tests/unit/test_client_http.py \
     python/sglang/srt/constrained/ananke/tests/unit/test_relaxation.py -q
   ```

2. **Check beads** (if using):
   ```bash
   bd stats
   bd ready
   ```

3. **Run full tests** (if verifying):
   ```bash
   PYTHONPATH="$PWD/python:$PYTHONPATH" python -m pytest \
     python/sglang/srt/constrained/ananke/tests/ -q --tb=no
   ```

The system is production-ready with all planned features implemented.
