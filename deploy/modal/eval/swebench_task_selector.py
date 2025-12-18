"""
SWE-bench Task Selector for Ananke Evaluation

Principled selection of 20-50 tasks from SWE-bench Lite based on:
1. Repository diversity - cover multiple codebases
2. Problem complexity - mix of easy/medium/hard
3. Patch size - prefer smaller, focused patches
4. Test clarity - clear FAIL_TO_PASS tests
5. Code domain - variety of problem types
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class TaskInfo:
    """Information about a SWE-bench task."""
    instance_id: str
    repo: str
    problem_statement: str
    patch_size: int  # Number of lines changed
    test_count: int  # Number of tests in FAIL_TO_PASS
    created_at: str
    version: str

    @property
    def repo_short(self) -> str:
        """Short repository name."""
        return self.repo.split("/")[-1] if "/" in self.repo else self.repo


def analyze_dataset():
    """Analyze SWE-bench Lite dataset and select tasks."""
    from datasets import load_dataset

    print("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # Analyze repository distribution
    repo_counts = defaultdict(int)
    repo_tasks = defaultdict(list)

    tasks = []
    for item in dataset:
        instance_id = item["instance_id"]
        repo = item["repo"]
        patch = item["patch"]
        problem = item["problem_statement"]
        fail_tests = item["FAIL_TO_PASS"]

        # Parse test count
        try:
            test_list = json.loads(fail_tests) if isinstance(fail_tests, str) else fail_tests
            test_count = len(test_list) if isinstance(test_list, list) else 1
        except:
            test_count = 1

        # Count patch lines
        patch_lines = len([l for l in patch.split("\n") if l.startswith("+") or l.startswith("-")])

        task = TaskInfo(
            instance_id=instance_id,
            repo=repo,
            problem_statement=problem[:500],  # Truncate for display
            patch_size=patch_lines,
            test_count=test_count,
            created_at=item.get("created_at", ""),
            version=item.get("version", ""),
        )
        tasks.append(task)
        repo_counts[repo] += 1
        repo_tasks[repo].append(task)

    print(f"\nTotal tasks: {len(tasks)}")
    print("\nRepository distribution:")
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        print(f"  {repo}: {count}")

    return tasks, repo_tasks, repo_counts


def select_tasks(
    tasks: list[TaskInfo],
    repo_tasks: dict,
    repo_counts: dict,
    target_count: int = 30,
    max_patch_size: int = 50,
) -> list[TaskInfo]:
    """
    Select tasks using principled criteria.

    Selection Strategy:
    1. Proportional repository sampling (maintain diversity)
    2. Prefer smaller patches (more focused, testable)
    3. Prefer tasks with 1-3 failing tests (clear signal)
    4. Exclude very large patches (> max_patch_size lines)
    """
    selected = []

    # Calculate proportional allocation per repo
    total = len(tasks)
    repo_allocation = {}
    for repo, count in repo_counts.items():
        # Proportional, but ensure at least 1 task per repo
        allocation = max(1, int(target_count * count / total))
        repo_allocation[repo] = allocation

    # Adjust to hit target count
    while sum(repo_allocation.values()) > target_count:
        # Reduce from largest allocation
        max_repo = max(repo_allocation, key=repo_allocation.get)
        if repo_allocation[max_repo] > 1:
            repo_allocation[max_repo] -= 1

    while sum(repo_allocation.values()) < target_count:
        # Add to smallest allocation with available tasks
        for repo in sorted(repo_counts.keys(), key=lambda r: repo_allocation.get(r, 0)):
            if len(repo_tasks[repo]) > repo_allocation.get(repo, 0):
                repo_allocation[repo] = repo_allocation.get(repo, 0) + 1
                break

    print(f"\nTarget allocation per repo:")
    for repo, alloc in sorted(repo_allocation.items()):
        print(f"  {repo}: {alloc}")

    # Select tasks from each repo
    for repo, allocation in repo_allocation.items():
        candidates = repo_tasks[repo]

        # Sort by selection criteria:
        # 1. Smaller patches preferred
        # 2. 1-3 tests preferred
        # 3. Filter out very large patches
        scored = []
        for task in candidates:
            if task.patch_size > max_patch_size:
                continue

            # Score: lower is better
            # Prefer patch sizes 5-20, test counts 1-3
            patch_score = abs(task.patch_size - 12)  # Ideal ~12 lines
            test_score = abs(task.test_count - 2) * 5  # Ideal 2 tests
            score = patch_score + test_score
            scored.append((score, task))

        # Sort and select
        scored.sort(key=lambda x: x[0])
        for _, task in scored[:allocation]:
            selected.append(task)

    return selected


def format_selection(selected: list[TaskInfo]) -> dict:
    """Format selected tasks for evaluation."""
    result = {
        "metadata": {
            "total_tasks": len(selected),
            "selection_criteria": [
                "Repository diversity (proportional sampling)",
                "Patch size preference (5-30 lines)",
                "Test clarity (1-3 failing tests)",
                "Problem complexity mix",
            ],
        },
        "tasks": [],
        "instance_ids": [],
    }

    by_repo = defaultdict(list)
    for task in selected:
        by_repo[task.repo].append(task)
        result["instance_ids"].append(task.instance_id)
        result["tasks"].append({
            "instance_id": task.instance_id,
            "repo": task.repo,
            "patch_size": task.patch_size,
            "test_count": task.test_count,
            "problem_preview": task.problem_statement[:200] + "...",
        })

    result["by_repository"] = {
        repo: [t.instance_id for t in tasks]
        for repo, tasks in by_repo.items()
    }

    return result


def main():
    """Main entry point."""
    print("=" * 70)
    print("SWE-bench Task Selector for Ananke Evaluation")
    print("=" * 70)

    tasks, repo_tasks, repo_counts = analyze_dataset()

    print("\n" + "=" * 70)
    print("Selecting 30 tasks with principled criteria...")
    print("=" * 70)

    selected = select_tasks(
        tasks, repo_tasks, repo_counts,
        target_count=30,
        max_patch_size=50,
    )

    result = format_selection(selected)

    print(f"\nSelected {len(selected)} tasks:")
    print("\nBy repository:")
    for repo, ids in result["by_repository"].items():
        print(f"  {repo}: {len(ids)} tasks")
        for id in ids[:3]:  # Show first 3
            print(f"    - {id}")
        if len(ids) > 3:
            print(f"    ... and {len(ids) - 3} more")

    # Save selection
    output_path = "/Users/rand/src/sglang/deploy/modal/eval/selected_tasks.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSelection saved to: {output_path}")

    # Print instance IDs for easy copy
    print("\n" + "=" * 70)
    print("Instance IDs (for evaluation):")
    print("=" * 70)
    for id in result["instance_ids"]:
        print(f"  {id}")

    return result


if __name__ == "__main__":
    main()
