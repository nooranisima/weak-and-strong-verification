"""
Tree-based data pre-generation for Sudoku experiments.

Generates a tree of candidate moves for each puzzle, where:
- Each node is one cell-fill attempt (with weak score + correctness label).
- Depth = n steps, branching factor = m candidates per step.

Supports checkpointing, resume, and appending additional puzzles.
"""

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ============================================================================
# Thread-safe counter
# ============================================================================

class AtomicCounter:
    def __init__(self, start: int = 0):
        self._value = start
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class TreeNode:
    node_id: str
    parent_id: Optional[str]
    depth: int
    row: Optional[int] = None
    col: Optional[int] = None
    value: Optional[int] = None
    reasoning: Optional[str] = None
    weak_score: Optional[float] = None
    is_correct: Optional[bool] = None
    path_from_root: List[Dict] = field(default_factory=list)
    sample_index: int = 0

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "row": self.row,
            "col": self.col,
            "value": self.value,
            "reasoning": self.reasoning,
            "weak_score": self.weak_score,
            "is_correct": self.is_correct,
            "path_from_root": self.path_from_root,
            "sample_index": self.sample_index,
        }


@dataclass
class PuzzleTree:
    puzzle_id: int
    puzzle: List[List[int]]
    solution: List[List[int]]
    nodes: Dict[str, TreeNode] = field(default_factory=dict)
    root_id: str = ""
    depth_n: int = 3
    branching_m: int = 4
    total_nodes: int = 0
    total_correct: int = 0
    generation_time_seconds: float = 0.0

    def get_all_paths(self) -> List[List[TreeNode]]:
        leaf_nodes = [
            n for n in self.nodes.values() if n.depth == self.depth_n
        ]
        paths = []
        for leaf in leaf_nodes:
            path = []
            current = leaf
            while current.parent_id is not None:
                path.append(current)
                current = self.nodes[current.parent_id]
            path.reverse()
            paths.append(path)
        return paths

    def to_dict(self) -> Dict:
        return {
            "puzzle_id": self.puzzle_id,
            "puzzle": self.puzzle,
            "solution": self.solution,
            "depth_n": self.depth_n,
            "branching_m": self.branching_m,
            "total_nodes": self.total_nodes,
            "total_correct": self.total_correct,
            "generation_time_seconds": self.generation_time_seconds,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "root_id": self.root_id,
        }


# ============================================================================
# Single-candidate generation (thread worker)
# ============================================================================

def generate_single_candidate(
    puzzle, solution, previous_steps, parent_id, depth, sample_index,
    generator, weak_verifier, strong_verifier,
):
    """Generate one candidate node (called inside thread pool)."""
    node_id = str(uuid.uuid4())[:8]
    try:
        row, col, value, reasoning = generator._get_next_move_with_history(
            puzzle, previous_steps
        )
        if row is None:
            return TreeNode(
                node_id=node_id, parent_id=parent_id, depth=depth,
                reasoning=f"Failed: {reasoning}", weak_score=0.0,
                is_correct=False, path_from_root=previous_steps.copy(),
                sample_index=sample_index,
            )

        weak_score = weak_verifier.score_cell_placement(
            puzzle, previous_steps, row, col, value,
            generator_reasoning=reasoning,
        )
        is_correct, _ = strong_verifier.verify_cell_correctness(
            solution, row, col, value
        )
        new_path = previous_steps + [
            {"row": row, "col": col, "value": value, "reasoning": reasoning}
        ]

        return TreeNode(
            node_id=node_id, parent_id=parent_id, depth=depth,
            row=row, col=col, value=value, reasoning=reasoning,
            weak_score=weak_score, is_correct=is_correct,
            path_from_root=new_path, sample_index=sample_index,
        )
    except Exception as e:
        return TreeNode(
            node_id=node_id, parent_id=parent_id, depth=depth,
            reasoning=f"Exception: {e}", weak_score=0.0, is_correct=False,
            path_from_root=previous_steps.copy(), sample_index=sample_index,
        )


# ============================================================================
# Puzzle-level tree generation
# ============================================================================

def generate_puzzle_tree(
    problem, n, m, max_workers, generator, weak_verifier, strong_verifier,
    progress_counter=None,
):
    """Generate a complete tree for one puzzle."""
    start_time = time.time()
    puzzle, solution = problem["puzzle"], problem["solution"]

    tree = PuzzleTree(
        puzzle_id=problem["id"], puzzle=puzzle, solution=solution,
        depth_n=n, branching_m=m,
    )
    root = TreeNode(node_id="root", parent_id=None, depth=0, path_from_root=[])
    tree.nodes["root"] = root
    tree.root_id = "root"

    current_frontier = [root]
    for depth in range(1, n + 1):
        tasks = [
            {
                "puzzle": puzzle, "solution": solution,
                "previous_steps": p.path_from_root,
                "parent_id": p.node_id, "depth": depth, "sample_index": i,
            }
            for p in current_frontier
            for i in range(m)
        ]

        next_frontier = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    generate_single_candidate,
                    t["puzzle"], t["solution"], t["previous_steps"],
                    t["parent_id"], t["depth"], t["sample_index"],
                    generator, weak_verifier, strong_verifier,
                )
                for t in tasks
            ]
            for future in as_completed(futures):
                try:
                    node = future.result()
                    tree.nodes[node.node_id] = node
                    next_frontier.append(node)
                    if node.is_correct:
                        tree.total_correct += 1
                except Exception as e:
                    print(f"Node error: {e}")
        current_frontier = next_frontier

    tree.total_nodes = len(tree.nodes) - 1
    tree.generation_time_seconds = time.time() - start_time

    if progress_counter:
        count = progress_counter.increment()
        print(
            f"  ✓ Puzzle {problem['id']} done ({count} total) - "
            f"{tree.total_correct}/{tree.total_nodes} correct, "
            f"{tree.generation_time_seconds:.1f}s"
        )
    return tree


# ============================================================================
# Checkpointing helpers
# ============================================================================

def _get_checkpoint_path(save_path: str) -> str:
    base, ext = os.path.splitext(save_path)
    return f"{base}_checkpoint{ext}"


def _save_checkpoint(trees, completed_ids, metadata, checkpoint_path):
    data = {
        "metadata": metadata,
        "completed_puzzle_ids": list(completed_ids),
        "trees": [t.to_dict() for t in trees],
        "checkpoint_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    temp_path = checkpoint_path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f)
    os.replace(temp_path, checkpoint_path)
    print(f"💾 Checkpoint: {len(trees)} puzzles saved")


def _load_trees_from_json(data: dict) -> List[PuzzleTree]:
    """Parse a list of PuzzleTree from a JSON dict."""
    trees = []
    for td in data.get("trees", []):
        tree = PuzzleTree(
            puzzle_id=td["puzzle_id"], puzzle=td["puzzle"],
            solution=td["solution"], depth_n=td["depth_n"],
            branching_m=td["branching_m"], total_nodes=td["total_nodes"],
            total_correct=td["total_correct"],
            generation_time_seconds=td["generation_time_seconds"],
            root_id=td["root_id"],
        )
        for nid, nd in td["nodes"].items():
            tree.nodes[nid] = TreeNode(
                node_id=nd["node_id"], parent_id=nd["parent_id"],
                depth=nd["depth"], row=nd["row"], col=nd["col"],
                value=nd["value"], reasoning=nd["reasoning"],
                weak_score=nd["weak_score"], is_correct=nd["is_correct"],
                path_from_root=nd["path_from_root"],
                sample_index=nd["sample_index"],
            )
        trees.append(tree)
    return trees


def _load_existing_data(path: str):
    if not os.path.exists(path):
        return [], {}, set()
    with open(path, "r") as f:
        data = json.load(f)
    trees = _load_trees_from_json(data)
    completed_ids = {t.puzzle_id for t in trees}
    return trees, data.get("metadata", {}), completed_ids


def _load_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        return [], set(), {}
    with open(checkpoint_path, "r") as f:
        data = json.load(f)
    trees = _load_trees_from_json(data)
    return trees, set(data.get("completed_puzzle_ids", [])), data.get("metadata", {})


# ============================================================================
# Main generation function
# ============================================================================

def generate_tree_dataset(
    dataset: List[Dict],
    generator,
    weak_verifier,
    strong_verifier,
    num_problems: int = None,
    num_additional: int = None,
    n: int = 3,
    m: int = 4,
    puzzle_parallelism: int = 3,
    node_parallelism: int = 8,
    save_path: Optional[str] = None,
    checkpoint_every: int = 5,
    resume_from_checkpoint: bool = True,
) -> List[PuzzleTree]:
    """
    Generate trees with parallelization, checkpointing, and append mode.

    Args:
        dataset: List of puzzle dicts (from load_sudoku_dataset_hf).
        generator: SudokuGenerator instance.
        weak_verifier: WeakVerifier instance.
        strong_verifier: StrongVerifier instance.
        num_problems: Total puzzles wanted (use this OR num_additional).
        num_additional: Add this many NEW puzzles (use this OR num_problems).
        n: Tree depth (steps per puzzle).
        m: Branching factor (candidates per step).
        puzzle_parallelism: Puzzles to process in parallel.
        node_parallelism: Nodes to generate in parallel per puzzle.
        save_path: Path to save JSON results.
        checkpoint_every: How often to checkpoint.
        resume_from_checkpoint: Whether to resume.

    Returns:
        List of PuzzleTree objects.
    """
    if num_problems is None and num_additional is None:
        raise ValueError("Must specify either num_problems or num_additional")
    if num_problems is not None and num_additional is not None:
        raise ValueError("Specify only one of num_problems or num_additional")

    # Load existing data
    existing_trees: List[PuzzleTree] = []
    completed_ids: Set = set()

    if save_path and os.path.exists(save_path):
        print(f"📂 Loading existing data from {save_path}...")
        existing_trees, _, completed_ids = _load_existing_data(save_path)
        print(f"   Found {len(existing_trees)} existing puzzles")

    checkpoint_path = _get_checkpoint_path(save_path) if save_path else "checkpoint.json"
    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        print(f"📂 Found checkpoint, loading...")
        cp_trees, cp_ids, _ = _load_checkpoint(checkpoint_path)
        for tree in cp_trees:
            if tree.puzzle_id not in completed_ids:
                existing_trees.append(tree)
                completed_ids.add(tree.puzzle_id)
        print(f"   After merging: {len(existing_trees)} puzzles")

    num_existing = len(existing_trees)
    if num_additional is not None:
        target_total = num_existing + num_additional
    else:
        target_total = num_problems

    all_problems = dataset[:target_total]
    remaining_problems = [p for p in all_problems if p["id"] not in completed_ids]

    print(f"\n{'=' * 70}")
    print("TREE GENERATION")
    print(f"{'=' * 70}")
    print(f"  Existing: {num_existing} puzzles")
    print(f"  Target: {target_total} puzzles")
    print(f"  To generate: {len(remaining_problems)} puzzles")
    print(f"  Parallelization: {puzzle_parallelism} puzzles × {node_parallelism} nodes")
    print(f"{'=' * 70}\n")

    if not remaining_problems:
        print("✅ Already have enough puzzles!")
        return existing_trees

    metadata = {
        "num_problems": target_total,
        "depth_n": n,
        "branching_m": m,
        "generator_model": getattr(generator.config, "generator_model", "unknown"),
        "weak_verifier_model": getattr(generator.config, "weak_verifier_model", "unknown"),
    }

    start_time = time.time()
    results = list(existing_trees)
    all_completed_ids = set(completed_ids)
    progress = AtomicCounter(len(existing_trees))

    for batch_start in range(0, len(remaining_problems), puzzle_parallelism):
        batch = remaining_problems[batch_start : batch_start + puzzle_parallelism]
        batch_num = batch_start // puzzle_parallelism + 1
        total_batches = (len(remaining_problems) + puzzle_parallelism - 1) // puzzle_parallelism
        print(f"\n🔄 Batch {batch_num}/{total_batches} ({len(batch)} puzzles)")

        with ThreadPoolExecutor(max_workers=puzzle_parallelism) as executor:
            futures = {
                executor.submit(
                    generate_puzzle_tree, p, n, m, node_parallelism,
                    generator, weak_verifier, strong_verifier, progress,
                ): p
                for p in batch
            }
            for future in as_completed(futures):
                problem = futures[future]
                try:
                    tree = future.result()
                    results.append(tree)
                    all_completed_ids.add(problem["id"])
                except Exception as e:
                    print(f"  ❌ Puzzle {problem['id']} failed: {e}")

        newly_generated = len(results) - num_existing
        if (
            save_path
            and newly_generated % checkpoint_every < puzzle_parallelism
            and newly_generated > 0
        ):
            _save_checkpoint(results, all_completed_ids, metadata, checkpoint_path)

    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("✅ COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Previously had: {num_existing}")
    print(f"  Newly generated: {len(results) - num_existing}")
    print(f"  Total now: {len(results)}")
    total_nodes = sum(t.total_nodes for t in results)
    total_correct = sum(t.total_correct for t in results)
    if total_nodes > 0:
        print(f"  Accuracy: {total_correct / total_nodes:.2%}")
    print(f"  Time: {total_time:.1f}s")

    if save_path:
        metadata["total_time_seconds"] = total_time
        with open(save_path, "w") as f:
            json.dump(
                {"metadata": metadata, "trees": [t.to_dict() for t in results]},
                f,
                indent=2,
            )
        print(f"\n✓ Saved {len(results)} puzzles to {save_path}")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    return results


# ============================================================================
# Load and analyse
# ============================================================================

def load_tree_data(path: str):
    """Load tree data from JSON. Returns (trees, metadata)."""
    with open(path, "r") as f:
        data = json.load(f)
    trees = _load_trees_from_json(data)
    return trees, data["metadata"]


def analyze_tree_data(trees: List[PuzzleTree]) -> Dict:
    """Print summary statistics for generated tree data."""
    all_nodes = [
        {
            "puzzle_id": t.puzzle_id,
            "depth": n.depth,
            "weak_score": n.weak_score,
            "is_correct": n.is_correct,
        }
        for t in trees
        for n in t.nodes.values()
        if n.depth > 0
    ]
    df = pd.DataFrame(all_nodes)

    print(f"\n{'=' * 70}")
    print("ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Puzzles: {len(trees)}")
    print(f"Total nodes: {len(df)}")
    print(f"Accuracy: {df['is_correct'].mean():.2%}")
    print(f"\nBy Depth:")
    for d in sorted(df["depth"].unique()):
        sub = df[df["depth"] == d]
        print(
            f"  Depth {d}: accuracy={sub['is_correct'].mean():.2%}, "
            f"weak_score={sub['weak_score'].mean():.3f}"
        )
    return {"total_nodes": len(df), "accuracy": df["is_correct"].mean()}
