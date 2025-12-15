# Developer Experience Improvements for Ananke

This document identifies opportunities to improve the developer experience (DX) for users of the Ananke constrained generation system.

---

## Completed Improvements

### 1. ✅ Add "ananke" to GRAMMAR_BACKEND_CHOICES

**Status:** COMPLETED

**File:** `python/sglang/srt/server_args.py:140`

```python
GRAMMAR_BACKEND_CHOICES = ["xgrammar", "outlines", "llguidance", "ananke", "none"]
```

Users can now select `--grammar-backend=ananke` via CLI.

---

### 2. ✅ Expose Ananke-Specific CLI Arguments

**Status:** COMPLETED

**Files Modified:**
- `python/sglang/srt/server_args.py` (ServerArgs class + argument parser)
- `python/sglang/srt/constrained/ananke/backend/backend.py` (factory function)

**New CLI Arguments:**

```bash
--ananke-language LANG         # python, typescript, go, rust, kotlin, swift, zig
--ananke-max-rollback-tokens N # Maximum rollback history (default: 200)
--ananke-enabled-domains DOMS  # Comma-separated: syntax,types,imports,controlflow,semantics
```

**Example Usage:**

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --grammar-backend ananke \
    --ananke-language typescript \
    --ananke-max-rollback-tokens 100 \
    --ananke-enabled-domains "syntax,types,imports"
```

---

## High Priority Improvements

### 3. Better Error Messages for Constraint Violations

**Current State:** When constraints become unsatisfiable, the grammar simply terminates with no explanation.

**Improvement:** Add diagnostic messages explaining which domain caused the violation.

```python
# In AnankeGrammar.accept_token()
def accept_token(self, token: int) -> None:
    # ... existing code ...

    if self.constraint.is_bottom():
        # NEW: Identify which domain caused the violation
        violation_domains = []
        if self.constraint.types.is_bottom():
            violation_domains.append("types")
        if self.constraint.imports.is_bottom():
            violation_domains.append("imports")
        # ... etc

        logger.warning(
            f"Constraint violation in domains: {violation_domains}. "
            f"Token '{token_text}' (id={token}) caused unsatisfiable state."
        )
        self.finished = True
```

### 4. Add Request-Level Domain Configuration

**Current State:** Domains are configured per-backend, not per-request.

**Improvement:** Allow request-level configuration via `sampling_params`:

```python
# Example API usage
response = requests.post(url, json={
    "text": prompt,
    "sampling_params": {
        "json_schema": schema,
        "ananke_config": {  # NEW
            "enabled_domains": ["syntax", "types"],
            "language": "typescript",
        }
    }
})
```

### 5. Add Constraint Visualization for Debugging

**Current State:** No way to inspect active constraints during generation.

**Improvement:** Add a debug endpoint or method:

```python
# Add to AnankeGrammar
def get_constraint_summary(self) -> Dict[str, Any]:
    """Get human-readable summary of current constraints."""
    return {
        "position": self.context.position,
        "domains": {
            name: {
                "is_top": domain.top == self.constraint.types,  # etc
                "is_bottom": domain.bottom == self.constraint.types,
                "satisfiability": str(self.constraint.types.satisfiability()),
            }
            for name, domain in self.domains.items()
        },
        "finished": self.finished,
    }
```

---

## Medium Priority Improvements

### 6. Progressive Type Error Reporting

**Current State:** Type errors are detected but details are lost.

**Improvement:** Track and surface type error details:

```python
@dataclass
class TypeErrorInfo:
    expected: Type
    actual: Type
    position: int
    context: str

# In TypeDomain.observe_token()
if unification_failed:
    self._last_error = TypeErrorInfo(
        expected=expected_type,
        actual=inferred_type,
        position=context.position,
        context=context.generated_text[-50:],
    )
```

### 7. Add Streaming Constraint Feedback

For long generations, provide periodic constraint status:

```python
# In fill_vocab_mask, emit status every N tokens
if self.context.position % 100 == 0:
    logger.info(
        f"Position {self.context.position}: "
        f"Types={self.constraint.types.satisfiability()}, "
        f"Imports={len(self._detected_imports)} imports detected"
    )
```

### 8. Precomputed Type Masks Documentation

Document how to extend the precomputed type masks for custom types:

```python
# In TypeDomain
def register_custom_type_mask(
    self,
    type_name: str,
    valid_tokens: Set[int],
) -> None:
    """Register a custom type mask.

    Use this for domain-specific types (e.g., custom classes,
    library types) that aren't covered by built-in masks.

    Example:
        domain.register_custom_type_mask("MyClass", {token_id_1, token_id_2})
    """
    self._extended_type_masks[type_name] = valid_tokens
```

---

## Low Priority Improvements

### 9. OpenAI API Integration Example

Add examples for OpenAI-compatible endpoint with Ananke features:

```python
# In GETTING_STARTED.md, expand OpenAI section:
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="na")

# Ananke-specific configuration via extra_body
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Write a function:"}],
    extra_body={
        "regex": r"def \w+\([^)]*\) -> \w+:",
        # Proposed: Ananke config
        # "ananke_language": "python",
        # "ananke_enabled_domains": ["syntax", "types"],
    }
)
```

### 10. Add Metrics Export

Support Prometheus metrics for monitoring:

```python
# Proposed metrics
ananke_tokens_generated_total{domain="types"} 12345
ananke_constraint_violations_total{domain="types"} 5
ananke_mask_compute_seconds{domain="types"} 0.0005
ananke_cache_hit_ratio{domain="types"} 0.85
ananke_rollback_count_total 12
```

### 11. Type Stub Generation

Generate `.pyi` stubs for IDE autocomplete:

```bash
# Proposed command
python -m sglang.srt.constrained.ananke.tools.generate_stubs \
    --output-dir ./stubs
```

---

## Implementation Priority

| Issue | Priority | Effort | Impact |
|-------|----------|--------|--------|
| 1. Add to GRAMMAR_BACKEND_CHOICES | Critical | Low | High |
| 2. Expose CLI arguments | Critical | Low | High |
| 3. Better error messages | High | Medium | High |
| 4. Request-level config | High | Medium | Medium |
| 5. Constraint visualization | High | Medium | High |
| 6. Type error details | Medium | Medium | Medium |
| 7. Streaming feedback | Medium | Low | Low |
| 8. Custom type masks | Medium | Low | Medium |
| 9. OpenAI examples | Low | Low | Medium |
| 10. Metrics export | Low | High | Low |
| 11. Type stubs | Low | Medium | Low |

---

## Quick Wins (Can Implement Now)

### Fix GRAMMAR_BACKEND_CHOICES

```diff
--- a/python/sglang/srt/server_args.py
+++ b/python/sglang/srt/server_args.py
@@ -137,7 +137,7 @@ SUPPORTED_BACKENDS = {
     Backend.FLASHINFER: "flashinfer",
 }

-GRAMMAR_BACKEND_CHOICES = ["xgrammar", "outlines", "llguidance", "none"]
+GRAMMAR_BACKEND_CHOICES = ["xgrammar", "outlines", "llguidance", "ananke", "none"]
```

### Add CLI Arguments

```diff
--- a/python/sglang/srt/server_args.py
+++ b/python/sglang/srt/server_args.py
@@ -397,6 +397,10 @@ class ServerArgs:
     # Constrained Decoding
     grammar_backend: Optional[str] = None

+    # Ananke-specific
+    ananke_language: str = "python"
+    ananke_max_rollback_tokens: int = 200
+
     # Speculative Decoding
     speculative_algorithm: Optional[str] = None
```

---

## Summary

The most impactful DX improvements are:

1. **Enable CLI selection** - Users can't even try Ananke without this
2. **Expose configuration** - Currently requires code changes
3. **Improve diagnostics** - Hard to debug constraint violations

These three changes would dramatically improve the Ananke user experience with minimal implementation effort.
