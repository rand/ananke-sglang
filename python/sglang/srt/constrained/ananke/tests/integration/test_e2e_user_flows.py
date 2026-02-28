"""End-to-end user flow validation for Ananke.

Tests the complete user experience across key jobs-to-be-done:
1. ConstraintSpec creation and serialization
2. ConstraintBuilder fluent API
3. Client SDK creation
4. Multi-language support
5. Observability
6. Cache behavior
7. Constraint algebra
8. Adaptive intensity
9. SamplingParams integration
"""
import pytest
import sys

# Use conftest.py path setup for proper imports


# =============================================================================
# Flow 1: ConstraintSpec Creation
# =============================================================================
class TestConstraintSpecCreation:
    def test_create_with_json_schema(self):
        from spec.constraint_spec import ConstraintSpec, CacheScope
        spec = ConstraintSpec(
            json_schema='{"type": "object", "properties": {"name": {"type": "string"}}}',
            language="python",
            type_bindings=[],
            cache_scope=CacheScope.SYNTAX_ONLY,
        )
        assert spec.has_syntax_constraint()
        assert spec.get_syntax_constraint_type() == "json_schema"
        assert spec.language == "python"
        key = spec.compute_cache_key()
        assert len(key) > 0

    def test_create_from_dict(self):
        from spec.constraint_spec import ConstraintSpec
        spec = ConstraintSpec.from_dict({
            'regex': r'\d+',
            'language': 'typescript',
            'type_bindings': [{'name': 'x', 'type_expr': 'number'}],
        })
        assert spec.regex == r'\d+'
        assert spec.language == 'typescript'
        assert len(spec.type_bindings) == 1

    def test_dict_roundtrip(self):
        from spec.constraint_spec import ConstraintSpec
        orig = ConstraintSpec(regex=r'[a-z]+', language='go', imports=[])
        d = orig.to_dict()
        restored = ConstraintSpec.from_dict(d)
        assert restored.regex == orig.regex
        assert restored.language == orig.language


# =============================================================================
# Flow 2: ConstraintBuilder Fluent API
# =============================================================================
class TestConstraintBuilder:
    def test_basic_builder(self):
        from client import ConstraintBuilder
        from spec.constraint_spec import ConstraintSpec
        # Use to_dict() + from_dict() to avoid relative import issues
        # when client is imported as a top-level package via conftest
        d = (ConstraintBuilder()
            .regex(r'\w+\(\)')
            .language('python')
            .type_binding('x', 'int')
            .expected_type('str')
            .to_dict())
        assert d['regex'] == r'\w+\(\)'
        assert d['language'] == 'python'
        assert d['expected_type'] == 'str'
        assert len(d['type_bindings']) == 1
        # Verify roundtrip through ConstraintSpec
        spec = ConstraintSpec.from_dict(d)
        assert spec.regex == r'\w+\(\)'
        assert spec.language == 'python'
        assert spec.expected_type == 'str'

    def test_builder_with_controlflow(self):
        from client import ConstraintBuilder
        from spec.constraint_spec import ConstraintSpec
        d = (ConstraintBuilder()
            .json_schema('{"type": "object"}')
            .language('rust')
            .in_function('process', return_type='Result<(), Error>')
            .in_loop(depth=2)
            .to_dict())
        assert d['json_schema'] == '{"type": "object"}'
        assert 'control_flow' in d
        assert d['control_flow']['function_name'] == 'process'
        # Verify roundtrip through ConstraintSpec
        spec = ConstraintSpec.from_dict(d)
        assert spec.json_schema == '{"type": "object"}'
        assert spec.control_flow is not None
        assert spec.control_flow.function_name == 'process'


# =============================================================================
# Flow 3: Client SDK
# =============================================================================
class TestClientSDK:
    def test_client_creation(self):
        from client import AnankeClient, RetryConfig, LoggingConfig
        client = AnankeClient(
            base_url="http://localhost:8000",
            retry_config=RetryConfig(max_retries=3, initial_delay=0.5),
            logging_config=LoggingConfig(log_request_body=True),
        )
        assert client.base_url == "http://localhost:8000"
        assert client._transport.retry_config.max_retries == 3

    def test_async_client_creation(self):
        pytest.importorskip("aiohttp", reason="aiohttp required for async client")
        from client import AnankeAsyncClient
        client = AnankeAsyncClient(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

    def test_generation_config(self):
        from client import GenerationConfig
        config = GenerationConfig(
            max_tokens=512, temperature=0.7, top_p=0.9, seed=42,
        )
        assert config.max_tokens == 512
        assert config.temperature == 0.7


# =============================================================================
# Flow 4: Multi-language Support
# =============================================================================
class TestMultiLanguage:
    def test_language_detection(self):
        from spec.language_detector import LanguageDetector
        detector = LanguageDetector()
        # detect() takes text as positional arg, returns language string
        assert detector.detect("fn main() { println!(\"hello\"); }") == "rust"
        assert detector.detect("package main\n\nimport \"fmt\"\n\nfunc main() {}") == "go"
        assert detector.detect("def foo():\n    return 1") == "python"

    @pytest.mark.parametrize("lang", [
        "python", "typescript", "go", "rust", "kotlin", "swift", "zig"
    ])
    def test_constraint_spec_per_language(self, lang):
        from spec.constraint_spec import ConstraintSpec
        spec = ConstraintSpec(regex=r'\w+', language=lang)
        assert spec.language == lang
        assert len(spec.compute_cache_key()) > 0


# =============================================================================
# Flow 5: Observability
# =============================================================================
class TestObservability:
    def test_metrics_collector(self):
        from observability.collector import MetricsCollector
        collector = MetricsCollector(vocab_size=32000)
        collector.record_mask_application(popcount=15000, domain="types")
        collector.record_domain_latency("types", 0.5)
        summary = collector.get_summary()
        assert summary is not None
        assert "mask_metrics" in summary

    def test_log_exporter(self):
        from observability.exporter import LogExporter
        exporter = LogExporter(format="json")
        assert exporter is not None

    def test_callback_exporter(self):
        from observability.exporter import CallbackExporter
        received = []
        exporter = CallbackExporter(metrics_callback=lambda s: received.append(s))
        assert exporter is not None


# =============================================================================
# Flow 6: Cache Behavior
# =============================================================================
class TestCacheBehavior:
    def test_cache_key_stability(self):
        from spec.constraint_spec import ConstraintSpec
        spec1 = ConstraintSpec(json_schema='{"type":"object"}', language="python")
        spec2 = ConstraintSpec(json_schema='{"type":"object"}', language="python")
        assert spec1.compute_cache_key() == spec2.compute_cache_key()

    def test_cache_key_differentiation(self):
        from spec.constraint_spec import ConstraintSpec
        spec1 = ConstraintSpec(json_schema='{"type":"object"}', language="python")
        spec2 = ConstraintSpec(json_schema='{"type":"array"}', language="python")
        assert spec1.compute_cache_key() != spec2.compute_cache_key()


# =============================================================================
# Flow 7: Constraint Algebra
# =============================================================================
class TestConstraintAlgebra:
    def test_semilattice_identity_annihilation(self):
        from core.constraint import TopConstraint, BottomConstraint
        top = TopConstraint()
        bottom = BottomConstraint()
        assert top.meet(top).is_top()
        assert top.meet(bottom).is_bottom()

    def test_unified_constraint(self):
        from core.unified import UNIFIED_TOP, UNIFIED_BOTTOM
        assert UNIFIED_TOP.is_top()
        assert UNIFIED_BOTTOM.is_bottom()
        assert UNIFIED_TOP.meet(UNIFIED_BOTTOM).is_bottom()


# =============================================================================
# Flow 8: Adaptive Intensity
# =============================================================================
class TestAdaptiveIntensity:
    def test_intensity_levels(self):
        from adaptive.intensity import (
            ConstraintIntensity, TaskComplexityAssessor, IntensityConfig,
        )
        assessor = TaskComplexityAssessor(config=IntensityConfig())
        simple = assessor.assess(prompt="x = 1", expected_tokens=5)
        assert simple.value <= ConstraintIntensity.STANDARD.value
        complex_p = assessor.assess(
            prompt="class MyService:\n    async def handle(self, request):\n        try:\n",
            expected_tokens=500,
        )
        assert complex_p.value >= ConstraintIntensity.STANDARD.value
