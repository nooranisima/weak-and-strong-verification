"""
LLM-based Sudoku move generator.

Fast + robust version with internal retries. Sends rules/examples once
in the system message; returns one move per call.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from .config import SudokuConfig
from .constants import EXAMPLE_PUZZLES, SUDOKU_RULES, format_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_example_text() -> str:
    """One short example placed once in system context."""
    ex = EXAMPLE_PUZZLES[0]
    return (
        "One solved example (for format and style only):\n"
        f"Puzzle (0=empty):\n{format_grid(ex['puzzle'])}\n\n"
        f"Solution:\n{format_grid(ex['solution'], show_zeros=False)}\n\n"
        f"Example reasoning style: {ex['explanation']}\n"
    )


def _safe_json_loads(s: str) -> Optional[dict]:
    """Try to parse JSON; also extract first {...} block if needed."""
    if s is None:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _to_int_maybe(x) -> Optional[int]:
    """Robustly coerce to int."""
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str):
            return int(float(x.strip()))
        return int(x)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SudokuGenerator:
    """
    LLM-based Sudoku move generator.

    - Does NOT resend rules/examples every step (system message reuse).
    - Returns ONE move per call: (row, col, value, reasoning_string).
    - Internal retries if model outputs invalid JSON or illegal move.
    """

    def __init__(self, config: SudokuConfig):
        self.config = config
        self.client = config.get_generator_client()
        self.call_count = 0

        example_text = _create_example_text()

        system_instructions = (
            "You solve 4x4 mini Sudoku by proposing ONE move at a time.\n\n"
            "Rules:\n"
            "- Grid is 4x4 with 2x2 boxes.\n"
            "- Fill each cell with 1..4.\n"
            "- Each row/col/2x2 box must contain 1,2,3,4 exactly once.\n"
            "- 0 means empty.\n\n"
            "Your job each turn:\n"
            "- Choose one NEW empty cell (0 in the ORIGINAL puzzle and not "
            "already filled in previous moves).\n"
            "- Output ONLY a JSON object with keys: row, col, value, why, "
            "confidence.\n\n"
            "Output schema (STRICT):\n"
            '{\n  "row": 0-3,\n  "col": 0-3,\n  "value": 1-4,\n'
            '  "why": "One short LOCAL deduction (row/col/box). No global '
            'reasoning.",\n  "confidence": 0.0-1.0\n}\n\n'
            "No extra keys. No markdown. No surrounding text.\n\n"
            f"{example_text}"
        )

        self.base_messages = [
            {"role": "system", "content": system_instructions},
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_zero_shot(
        self, puzzle: List[List[int]]
    ) -> Tuple[List[List[int]], str]:
        """Generate complete solution in one shot (optional utility)."""
        self.call_count += 1

        prompt = (
            "Solve this 4x4 Sudoku completely.\n"
            'Return ONLY JSON: {"solution": [[...],[...],[...],[...]], '
            '"reasoning": "..."}\n\n'
            f"Puzzle (0=empty):\n{format_grid(puzzle)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.generator_model,
                messages=self.base_messages + [{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            result = _safe_json_loads(content) or {}
            solution = result.get("solution", [[0] * 4 for _ in range(4)])
            reasoning = result.get("reasoning", "")
            if not isinstance(solution, list) or len(solution) != 4:
                solution = [[0] * 4 for _ in range(4)]
            return solution, reasoning
        except Exception as e:
            print(f"Error in generator: {e}")
            return [[0] * 4 for _ in range(4)], f"Error: {e}"

    def _get_next_move_with_history(
        self,
        original_puzzle: List[List[int]],
        previous_steps: List[Dict],
    ) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
        """
        Get next move using ONLY original puzzle + previous moves.

        Returns (row, col, value, reasoning_string).
        Internal retries so we rarely return None.
        """
        self.call_count += 1

        empty_cells = sum(
            1 for row in original_puzzle for cell in row if cell == 0
        )
        num_previous = len(previous_steps)
        remaining = empty_cells - num_previous
        is_final_move = remaining == 1

        # Render previous moves
        if previous_steps:
            steps_text = f"You have made {num_previous} moves so far:\n"
            for i, step in enumerate(previous_steps, 1):
                steps_text += (
                    f"  Move {i}: placed {step['value']} at "
                    f"(row={step['row']}, col={step['col']})\n"
                )
        else:
            steps_text = "You haven't made any moves yet.\n"

        base_step_prompt = (
            f"ORIGINAL PUZZLE (0=empty):\n{format_grid(original_puzzle)}\n\n"
            f"PREVIOUS MOVES:\n{steps_text}\n"
            "NEXT TASK:\n"
            + (
                "This is the FINAL move. Only 1 empty cell remains."
                if is_final_move
                else f"Choose the next move. {remaining} empty cells remain."
            )
            + "\n\nReturn ONLY JSON with keys row,col,value,why,confidence.\n"
            "Remember:\n"
            "- (row,col) must be an empty cell in the ORIGINAL puzzle (value 0 there)\n"
            "- and must NOT be one of the previous moves.\n"
        )

        MAX_RETRIES = 5
        last_err = None

        for _attempt in range(1, MAX_RETRIES + 1):
            prompt = base_step_prompt
            if last_err is not None:
                prompt += (
                    f"\nYour previous output was invalid because: {last_err}\n"
                    "Try again. Return ONLY valid JSON."
                )

            try:
                response = self.client.chat.completions.create(
                    model=self.config.generator_model,
                    messages=self.base_messages
                    + [{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                result = _safe_json_loads(content)

                if result is None:
                    last_err = "could not parse JSON"
                    continue

                row = _to_int_maybe(result.get("row"))
                col = _to_int_maybe(result.get("col"))
                value = _to_int_maybe(result.get("value"))
                why = result.get("why", "")

                if row is None or col is None or value is None:
                    last_err = "missing/invalid row/col/value"
                    continue
                if not (0 <= row < 4 and 0 <= col < 4):
                    last_err = f"row/col out of range: ({row},{col})"
                    continue
                if not (1 <= value <= 4):
                    last_err = f"value out of range: {value}"
                    continue
                if any(
                    s["row"] == row and s["col"] == col for s in previous_steps
                ):
                    last_err = f"cell ({row},{col}) already filled"
                    continue
                if original_puzzle[row][col] != 0:
                    last_err = f"cell ({row},{col}) is not empty in original"
                    continue

                reasoning = (why or "").strip()
                if is_final_move:
                    reasoning = f"[FINAL STEP] {reasoning}"
                return row, col, value, reasoning

            except Exception as e:
                last_err = f"API/exception: {e}"
                continue

        return (
            None,
            None,
            None,
            f"Failed after {MAX_RETRIES} retries. Last error: {last_err}",
        )

    def _get_next_move(
        self,
        puzzle: List[List[int]],
        current_grid: List[List[int]],
    ) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
        """Legacy wrapper: converts grid diff to previous_steps."""
        previous_steps = []
        for r in range(4):
            for c in range(4):
                if puzzle[r][c] == 0 and current_grid[r][c] != 0:
                    previous_steps.append(
                        {"row": r, "col": c, "value": current_grid[r][c]}
                    )
        return self._get_next_move_with_history(puzzle, previous_steps)
