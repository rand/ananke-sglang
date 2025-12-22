"""Debug constrained generation at different temperatures."""

import modal

DEPLOYED_APP = "qwen3-coder-ananke"
DEPLOYED_CLASS = "Qwen3CoderAnanke"

app = modal.App("debug-constraints")

@app.local_entrypoint()
def main():
    print("Connecting to deployed model...")
    Qwen3CoderAnanke = modal.Cls.from_name(DEPLOYED_APP, DEPLOYED_CLASS)
    server = Qwen3CoderAnanke()

    prompt = "def fibonacci(n: int) -> int:"

    for temp in [0.3, 0.6, 0.9]:
        print(f"\n{'='*60}")
        print(f"Temperature: {temp}")
        print(f"{'='*60}")

        # Test unconstrained
        print("\nUnconstrained:")
        try:
            result = server.generate.remote(
                prompt=prompt,
                max_tokens=100,
                temperature=temp,
            )
            print(f"  Success: {result[:100]}...")
        except Exception as e:
            print(f"  ERROR: {e}")

        # Test constrained
        print("\nConstrained:")
        try:
            result = server.generate_constrained.remote(
                prompt=prompt,
                constraint_spec={"language": "python"},
                max_tokens=100,
                temperature=temp,
            )
            print(f"  Success: {result.get('text', '')[:100]}...")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "="*60)
    print("Debug complete")
