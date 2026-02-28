# SGLang Agent Guide

SGLang is a fast serving framework for large language models (LLMs) and vision-language models (VLMs). This guide helps AI agents work effectively in this codebase.

## Project Overview

SGLang consists of three major components:

| Component | Language | Location | Description |
|-----------|----------|----------|-------------|
| **sglang** (main) | Python | `python/sglang/` | LLM/VLM inference runtime, scheduler, model executor |
| **sgl-kernel** | CUDA/C++ | `sgl-kernel/` | High-performance CUDA kernels |
| **sgl-model-gateway** | Rust | `sgl-model-gateway/` | HTTP/gRPC routing gateway with load balancing |

Additional submodules:
- **ananke** (`python/sglang/srt/constrained/ananke/`): Constraint system for verified code generation

---

## Essential Commands

### Python Main Package

```bash
# Install in development mode
cd python
pip install -e ".[dev]"

# Run linting/formatting
pre-commit run --all-files

# Run a single test file
python3 test/srt/test_srt_endpoint.py

# Run a single test case
python3 test/srt/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run a CI test suite
python3 test/run_suite.py --hw cuda --suite stage-a-test-1

# Launch server
python -m sglang.launch_server --model-path <model>
```

### sgl-kernel (CUDA Kernels)

```bash
cd sgl-kernel

# Install in development mode
make install

# Build wheel
make build

# Run tests
make test

# Format code
make format

# Clean build
make clean
```

### sgl-model-gateway (Rust Router)

```bash
cd sgl-model-gateway

# Build release binary
make build           # or: cargo build --release

# Run tests
make test            # or: cargo test

# Run linting
make check           # cargo check + clippy

# Format code
make fmt             # cargo +nightly fmt

# Build Python bindings (dev mode)
make python-dev

# Build Python wheel
make python-build

# Run Python tests
make python-test     # pytest py_test/ -v
```

---

## Code Organization

### Python (`python/sglang/`)

```
sglang/
├── cli/              # CLI entry points
├── lang/             # Frontend language DSL
├── srt/              # SRT = SGLang RunTime (core serving logic)
│   ├── configs/      # Model configurations
│   ├── constrained/  # Constrained decoding (ananke)
│   ├── entrypoints/  # HTTP/gRPC server entry points
│   ├── layers/       # Model layers (attention, MoE)
│   ├── managers/     # Tokenizer/IO managers
│   ├── mem_cache/    # KV cache, radix tree
│   ├── model_executor/  # GPU execution
│   ├── models/       # Model implementations
│   ├── sampling/     # Sampling backends
│   └── speculative/  # Speculative decoding
├── test/             # Test utilities
└── bench_*.py        # Benchmarking scripts
```

### Rust (`sgl-model-gateway/src/`)

```
src/
├── core/           # Worker management, circuit breaker, metrics
├── routers/        # HTTP/gRPC/OpenAI routing logic
│   ├── http/       # HTTP router
│   ├── grpc/       # gRPC router
│   └── openai/     # OpenAI-compatible endpoints
├── policies/       # Load balancing (round_robin, cache_aware, etc.)
├── protocols/      # Request/response types
├── tokenizer/      # Tokenizer and chat templates
├── tool_parser/    # Function call parsing
├── mcp/            # Model Context Protocol support
├── wasm/           # WebAssembly extension support
└── observability/  # Logging, metrics, tracing
```

### Tests

```
test/
├── registered/     # Per-commit tests with CI registration
├── nightly/        # Nightly-only tests
├── srt/            # SRT runtime tests
├── lang/           # Frontend language tests
├── manual/         # Manual tests (not in CI)
└── run_suite.py    # CI test runner
```

---

## Testing

### Test Registration (CI Integration)

Tests declare CI metadata via lightweight markers:

```python
from sglang.test.ci.ci_register import register_cuda_ci

# Per-commit test (runs on every PR)
register_cuda_ci(est_time=80, suite="stage-a-test-1")

# Nightly test (runs once per day)
register_cuda_ci(est_time=240, suite="nightly-1-gpu", nightly=True)

# Disabled test
register_cuda_ci(est_time=60, suite="stage-a-test-1", disabled="flaky on H100")
```

**Suite names:**
- Per-commit: `stage-a-test-1`, `stage-b-test-small-1-gpu`
- Nightly: `nightly-1-gpu`, `nightly-2-gpu`, `nightly-4-gpu`, `nightly-8-gpu`

### Running Tests

```bash
# Python tests use unittest or pytest
python3 test/srt/test_srt_endpoint.py                    # unittest
pytest test/srt/test_metrics.py -v                       # pytest

# Ensure test files have proper main block:
# unittest: unittest.main()
# pytest:   pytest.main([__file__])

# Rust tests
cd sgl-model-gateway && cargo test

# Gateway Python tests
cd sgl-model-gateway && pytest py_test/unit -v
```

---

## Code Style & Conventions

### Python

- **Formatting**: black, isort (via pre-commit)
- **Linting**: ruff (F401, F821 rules)
- **Type hints**: Extensively used with `Optional`, `Union`, `Literal`
- **Docstrings**: Brief module/class docstrings
- **Naming**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`
- **Logger**: `logger = logging.getLogger(__name__)` per module
- **License header**: Apache 2.0 at top of each file

**Import order:**
```python
# 1. License header
# 2. __future__ imports
# 3. Standard library (alphabetical)
# 4. Third-party packages
# 5. Local imports (sglang.*)
```

### Rust

- **Formatting**: `cargo +nightly fmt`
- **Linting**: `cargo clippy --all-targets --all-features -- -D warnings`
- **Error handling**: `thiserror` for error enums, `Result<T, Error>` pattern
- **Async**: `tokio` runtime, `async_trait` for trait methods
- **Serde**: Heavy use for serialization with tags and renames
- **Documentation**: `///` for items, `//!` for modules

### CUDA/C++

- **Formatting**: clang-format (version 18)
- Files: `*.cu`, `*.cuh`, `*.h`, `*.cpp`, `*.cc`

---

## Important Patterns

### Mixin Pattern (Python)

Large classes are split into mixins for organization:

```python
class Scheduler(
    SchedulerOutputProcessorMixin,
    SchedulerUpdateWeightsMixin,
    SchedulerProfilerMixin,
    ...
):
    pass
```

### Dataclass with Post-Init (Python)

```python
@dataclasses.dataclass
class ServerArgs:
    model_path: str
    tokenizer_path: Optional[str] = None

    def __post_init__(self):
        self._handle_deprecated_args()
```

### Builder Pattern (Rust)

```rust
RouterConfig::builder()
    .mode(mode)
    .policy(policy)
    .build()
```

### Error Type Alias (Rust)

```rust
pub type McpResult<T> = Result<T, McpError>;
```

---

## Gotchas & Non-Obvious Details

1. **Proto files must stay in sync**: `python/sglang/srt/grpc/sglang_scheduler.proto` and `sgl-model-gateway/src/proto/sglang_scheduler.proto` must be identical (CI enforces this).

2. **Generated proto files**: Don't edit `*_pb2.py` or `*_pb2_grpc.py` files directly—they're generated.

3. **Test timeout**: Default 1200s per file. Use smaller models and reuse servers to keep tests fast.

4. **GPU tests**: Prefer 1-2 GPU tests. Don't run long tests on 8-GPU runners.

5. **FlashInfer alignment**: Keep `flashinfer_python` and `flashinfer_cubin` versions aligned with Dockerfile.

6. **sccache for Rust**: Gateway uses sccache for compilation caching. Run `make setup-sccache` for local development.

7. **CI labels**: PRs need `run-ci` label for gateway tests to run. High-priority PRs get `high priority` label for faster parallelism.

8. **Nightly tests**: Place in `test/nightly/` and use `NightlyBenchmarkRunner` helper from `nightly_utils.py`.

9. **Model lists**: Add new text models to `python/sglang/test/test_utils.py`; VLMs go in `test/srt/nightly/test_vlms_mmmu_eval.py`.

---

## CI/CD Overview

### Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `lint.yml` | Push/PR to main | Pre-commit, clang-format, proto sync |
| `pr-test.yml` | PR to main | Python tests (CUDA) |
| `pr-test-rust.yml` | PR to main (gateway changes) | Rust tests, Python gateway tests |
| `nightly-test-nvidia.yml` | Nightly | Extended test suite |

### Test Suites

**Per-commit (on every PR):**
- `stage-a-test-1`: Core functionality
- `stage-b-test-small-1-gpu`: Small GPU tests

**Nightly:**
- `nightly-1-gpu`, `nightly-2-gpu`, `nightly-4-gpu`, `nightly-8-gpu`

---

## Key Files Reference

| Purpose | File |
|---------|------|
| Server arguments | `python/sglang/srt/server_args.py` |
| Model configs | `python/sglang/srt/configs/model_config.py` |
| Test utilities | `python/sglang/test/test_utils.py` |
| CI registration | `python/sglang/test/ci/ci_register.py` |
| Gateway main | `sgl-model-gateway/src/main.rs` |
| Gateway config | `sgl-model-gateway/src/config/` |

---

## Development Tips

1. **Before editing**: Always read the file first to understand context and exact formatting.

2. **Run tests after changes**: Use the most specific test first, then broaden.

3. **Use git blame**: For understanding why code exists in its current form.

4. **Check similar code**: When adding features, look for similar implementations.

5. **Memory files**: Check CLAUDE.md for additional project-specific instructions.

6. **Pre-commit**: Run `pre-commit run --all-files` before committing.

7. **Benchmarking**: Use `bench_serving.py` for serving benchmarks, `bench_one_batch.py` for single-batch tests.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

<!-- BEGIN BEADS INTEGRATION -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs with git:

- Exports to `.beads/issues.jsonl` after changes (5s debounce)
- Imports from JSONL when newer (e.g., after `git pull`)
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

<!-- END BEADS INTEGRATION -->
