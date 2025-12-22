# Ananke Constraint Evaluation Report

**Date**: 2025-12-19
**Model**: Qwen3 Coder (via Modal deployment)
**Framework**: Ananke constrained generation

## Executive Summary

- **Level 1 (Constraint Validity)**: 136/136 (100%) - All constraint examples are well-formed
- **Level 2 (Generation Eval)**: 53/92 (57.6%) - Model generates constraint-satisfying output ~58% of the time

## Level 1: Constraint Validation (No LLM)

All 136 constraint examples pass validation:
- valid_outputs match their regex/EBNF constraints
- EBNF grammars compile successfully

| Language | Examples | Pass Rate |
|----------|----------|-----------|
| Python | 16 | 100% |
| Rust | 20 | 100% |
| Zig | 19 | 100% |
| TypeScript | 19 | 100% |
| Go | 21 | 100% |
| Kotlin | 21 | 100% |
| Swift | 20 | 100% |

## Level 2: Generation Evaluation

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total Examples | 92 |
| Passed | 53 (57.6%) |
| Failed | 39 (42.4%) |
| Avg Latency | 1,105ms |
| P95 Latency | 1,840ms |

### Pass Rate by Language

| Language | Pass Rate | Analysis |
|----------|-----------|----------|
| **TypeScript** | 84.2% (16/19) | Best performer - likely due to model familiarity |
| **Go** | 61.9% (13/21) | Good performance |
| **Kotlin** | 55.0% (11/20) | Moderate |
| **Rust** | 50.0% (1/2) | Limited sample |
| **Swift** | 50.0% (1/2) | Limited sample |
| **Python** | 44.4% (4/9) | Surprisingly low for common language |
| **Zig** | 36.8% (7/19) | Worst performer - model struggles with Zig syntax |

### Pass Rate by Domain

| Domain | Pass Rate | Analysis |
|--------|-----------|----------|
| **imports** | 85.7% (12/14) | Simpler structural patterns work well |
| **types** | 71.4% (15/21) | Type annotations are well-handled |
| **semantics** | 64.3% (9/14) | Moderate complexity |
| **syntax** | 53.8% (7/13) | Mixed results |
| **controlflow** | 40.9% (9/22) | Complex patterns struggle |
| **coroutines** | 20.0% (1/5) | Async patterns are hard |
| **comptime** | 0.0% (0/3) | Zig comptime completely fails |

## Failure Analysis

### Failure Categories

| Category | Count | % of Failures | Root Cause |
|----------|-------|---------------|------------|
| **model_deviation** | 21 | 53.8% | Model generates syntactically valid but non-matching output |
| **truncated** | 15 | 38.5% | Output exceeds max_tokens (200) |
| **empty_output** | 2 | 5.1% | Model produces nothing |
| **generation_error** | 1 | 2.6% | Actual generation error |

### Key Insights

1. **model_deviation is the dominant failure mode (54%)**
   - The model produces code that doesn't match our regex patterns
   - Many outputs show strange behavior: excessive whitespace, natural language mixing
   - This suggests constraints may not be properly enforced during generation

2. **truncation is significant (38%)**
   - 200 max_tokens may be insufficient for longer code snippets
   - Some constraints implicitly require multi-line outputs

3. **Language-specific issues**
   - Zig performs worst (36.8%) - the model struggles with Zig syntax
   - TypeScript performs best (84.2%) - familiar language with clear patterns

4. **Domain-specific insights**
   - **imports** domain excels (85.7%) - simpler, structural patterns
   - **comptime** domain fails completely (0%) - too specialized

## Actionable Recommendations

### Immediate Fixes

1. **Increase max_tokens to 400** for longer code examples
2. **Widen regex patterns** that are too specific
3. **Add Zig training data** or use a model with better Zig knowledge

### Constraint Improvements

1. **For comptime domain**: Simplify constraints to accept broader valid Zig comptime patterns
2. **For coroutines domain**: Break complex patterns into smaller, composable constraints
3. **For Python**: Review why pass rate is low despite model familiarity

### Investigation Needed

1. Verify constraint_spec is being properly passed to the grammar backend
2. Check if AnankeGrammar is correctly enforcing regex constraints
3. Investigate why some outputs show excessive whitespace/degraded quality

## Methodology Notes

Following Hamel's eval methodology:
- **Binary pass/fail** - Output either matches constraint or doesn't
- **Error analysis first** - Focus on understanding failure modes
- **Traces over metrics** - Detailed output analysis reveals insights

## Files

- Results: `/tmp/level2_full.json`
- Level 1 results: `/tmp/level1_results.json`
- Eval runner: `tests/eval/run_eval.py`
