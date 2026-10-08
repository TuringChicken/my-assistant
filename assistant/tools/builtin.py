"""Simple built-in tools used to prove out the tool-calling loop (stage 2),
without the extra complexity of MCP.
"""
from __future__ import annotations

import ast
import operator as op
from datetime import datetime
from typing import Any

from .registry import Tool, ToolRegistry


# --- get_current_time -------------------------------------------------------

async def _current_time(_: dict[str, Any]) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- calculate --------------------------------------------------------------
# A small safe arithmetic evaluator (no eval()): supports + - * / // % ** and
# unary minus on numeric literals only.

_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


async def _calculate(args: dict[str, Any]) -> str:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return "Error: 'expression' is required."
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree.body))


# --- registration -----------------------------------------------------------

def register_builtin(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="get_current_time",
            description=(
                "Get the current local date and time. "
                "Use when the user asks what time or date it is."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_current_time,
        )
    )
    registry.register(
        Tool(
            name="calculate",
            description=(
                "Evaluate a basic arithmetic expression "
                "(supports + - * / // % ** and parentheses)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '(3 + 4) * 2'",
                    }
                },
                "required": ["expression"],
            },
            handler=_calculate,
        )
    )
