#!/usr/bin/env python3
"""
fp_ai.py – a tiny abstract interpreter for floating‑point C code
----------------------------------------------------------------
Given

   ①  a C source file,
   ②  a function name inside that file,
   ③  a YAML file that describes which concrete inputs are possible,

the script walks the function’s AST and prints, for *every* expression,
whether its value might become ±Inf or NaN.

The analysis is conservative: if we say “maybe Inf = False”, you really
cannot get an infinity at run‑time for any of the admissible inputs.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import typing

import yaml
import pycparser
from pycparser import c_ast, parse_file

# ---------------------------------------------------------------------------
# Toolbox constants
# ---------------------------------------------------------------------------

# pycparser ships with a fake set of <stdio.h> etc.  We need that directory
# when we run the pre‑processor.
FAKE_LIBC_INCLUDE = os.path.join(
    os.path.dirname(pycparser.__file__), "utils", "fake_libc_include"
)

# Largest finite value a 32‑bit float can hold – handy for clipping.
FLT_MAX = 3.4028234663852886e38

# ---------------------------------------------------------------------------
# Structures that feed the final report
# ---------------------------------------------------------------------------

# A line like:  (17, 'inf', True)   ==  “line 17 may overflow to ±Inf”
expr_report: list[tuple[int, str, bool]] = []

# We don’t want duplicate rows when the same AST node is evaluated twice.
expr_seen: set[tuple[int, str]] = set()

# ---------------------------------------------------------------------------
# Helper to log one expression result
# ---------------------------------------------------------------------------


def _record(node: c_ast.Node, absval: "AbsVal") -> None:
    """Log INF and (when relevant) NAN info for one expression."""
    line = getattr(getattr(node, "coord", None), "line", -1)
    nid = id(node)

    # Record INF unconditionally.
    if (nid, "inf") not in expr_seen:
        expr_report.append((line, "inf", bool(absval.maybe_inf)))
        expr_seen.add((nid, "inf"))

    # Record NAN only when the expression might be NaN.
    if (nid, "nan") not in expr_seen:
        expr_report.append((line, "nan", bool(absval.maybe_nan)))
        expr_seen.add((nid, "nan"))


# ---------------------------------------------------------------------------
# The abstract value: interval + two Boolean flags
# ---------------------------------------------------------------------------


class AbsVal:
    """[lo, hi]  ⊎  {±Inf?}  ⊎  {NaN?}"""

    __slots__ = ("min_val", "max_val", "maybe_inf", "maybe_nan")

    def __init__(
        self,
        min_val: float = -FLT_MAX,
        max_val: float = FLT_MAX,
        maybe_inf: bool = False,
        maybe_nan: bool = False,
    ):
        # Keep the bounds sane.
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        self.min_val = min_val
        self.max_val = max_val
        self.maybe_inf = maybe_inf
        self.maybe_nan = maybe_nan

    # ––– short helpers –––

    def __repr__(self):
        return (
            f"AbsVal(lo={self.min_val}, hi={self.max_val}, "
            f"inf={self.maybe_inf}, nan={self.maybe_nan})"
        )

    @staticmethod
    def top() -> "AbsVal":
        """The unconstrained element ⊤."""
        return AbsVal(-FLT_MAX, FLT_MAX, False, False)

    def copy(self) -> "AbsVal":
        return AbsVal(
            self.min_val, self.max_val, self.maybe_inf, self.maybe_nan
        )

    def join(self, other: "AbsVal") -> "AbsVal":
        """Least upper bound of two abstract values."""
        return AbsVal(
            min(self.min_val, other.min_val),
            max(self.max_val, other.max_val),
            self.maybe_inf or other.maybe_inf,
            self.maybe_nan or other.maybe_nan,
        )

    # ––– constructing from concrete samples –––
    @staticmethod
    def from_values(vals: typing.Iterable[float]) -> "AbsVal":
        lo = hi = None
        inf = nan = False
        for v in vals:
            if math.isnan(v):
                nan = True
            elif math.isinf(v):
                inf = True
            else:
                lo = v if lo is None else min(lo, v)
                hi = v if hi is None else max(hi, v)
        if lo is None:  # no finite value seen
            lo = hi = 0.0
        return AbsVal(lo, hi, inf, nan)


# ---------------------------------------------------------------------------
# YAML → initial environment
# ---------------------------------------------------------------------------


def _bits_to_float32(x) -> float:
    """32‑bit pattern (int or hex string) → float."""
    if isinstance(x, str):
        base = 16 if x.lower().startswith("0x") else 10
        x = int(x, base)
    x &= 0xFFFFFFFF
    return struct.unpack("!f", struct.pack("!I", x))[0]


def load_inputs(yaml_path: str) -> dict[str, AbsVal]:
    """Read the input specification YAML file."""
    with open(yaml_path, "r", encoding="utf‑8") as fp:
        data = yaml.safe_load(fp) or {}

    env: dict[str, AbsVal] = {}
    for name, raw in data.items():
        if isinstance(raw, str) and raw.strip() == ">":  # fully unconstrained
            env[name] = AbsVal.top()
            continue

        vals = raw if isinstance(raw, list) else [raw]
        env[name] = AbsVal.from_values(_bits_to_float32(v) for v in vals)

    return env


# ---------------------------------------------------------------------------
# Abstract arithmetic (+ – * /) over AbsVal
# ---------------------------------------------------------------------------


def _clip(lo: float, hi: float) -> tuple[float, float, bool]:
    """Clip lo/hi to finite range – returns new bounds plus Inf‑flag."""
    inf = False
    if hi > FLT_MAX:
        hi = FLT_MAX
        inf = True
    if lo < -FLT_MAX:
        lo = -FLT_MAX
        inf = True
    return lo, hi, inf


def arithmetic_op(op: str, a: AbsVal, b: AbsVal) -> AbsVal:
    """Abstract transfer for one binary arithmetic operator."""
    maybe_nan = a.maybe_nan or b.maybe_nan
    maybe_inf = a.maybe_inf or b.maybe_inf

    # --- + and – : endpoint math is enough ---
    if op in {"+", "-"}:
        if op == "+":
            combos = [
                a.min_val + b.min_val,
                a.min_val + b.max_val,
                a.max_val + b.min_val,
                a.max_val + b.max_val,
            ]
        else:
            combos = [
                a.min_val - b.max_val,
                a.min_val - b.min_val,
                a.max_val - b.max_val,
                a.max_val - b.min_val,
            ]
        lo, hi = _clip(min(combos), max(combos))[:2]

    # --- * :  four endpoint products ---
    elif op == "*":
        prods = [
            a.min_val * b.min_val,
            a.min_val * b.max_val,
            a.max_val * b.min_val,
            a.max_val * b.max_val,
        ]
        lo, hi, overflow = _clip(min(prods), max(prods))
        maybe_inf = maybe_inf or overflow
        # Inf * 0 => NaN possible
        if (a.maybe_inf and 0.0 in (b.min_val, b.max_val)) or (
            b.maybe_inf and 0.0 in (a.min_val, a.max_val)
        ):
            maybe_nan = True

    # --- / : need to watch out for 0 in the denominator ---
    elif op == "/":
        if b.min_val <= 0.0 <= b.max_val:
            # could divide by zero
            return AbsVal(-FLT_MAX, FLT_MAX, True, True or maybe_nan)
        if b.maybe_inf and not a.maybe_inf:
            # x / Inf is ÷ by huge number – interval shrinks towards 0
            maybe_inf = maybe_inf  # unchanged
        pairs = [
            (a.min_val, b.min_val),
            (a.min_val, b.max_val),
            (a.max_val, b.min_val),
            (a.max_val, b.max_val),
        ]
        vals = [x / y for x, y in pairs if y != 0.0]
        if not vals:  # should not happen, but stay safe
            return AbsVal.top()
        lo, hi, overflow = _clip(min(vals), max(vals))
        maybe_inf = maybe_inf or overflow
    else:
        return AbsVal.top()

    return AbsVal(lo, hi, maybe_inf, maybe_nan)


# ---------------------------------------------------------------------------
# Condition evaluation  – returns (can_be_true, can_be_false)
# ---------------------------------------------------------------------------


def evaluate_condition(node, env):
    # binary comparisons
    if isinstance(node, c_ast.BinaryOp):
        op = node.op
        if op in {">", "<", ">=", "<="}:
            l = evaluate_expression(node.left, env)
            r = evaluate_expression(node.right, env)

            if op == ">":
                if l.min_val > r.max_val:
                    return True, False
                if l.max_val <= r.min_val:
                    return False, True
            if op == "<":
                if l.max_val < r.min_val:
                    return True, False
                if l.min_val >= r.max_val:
                    return False, True
            if op == ">=":
                if l.min_val >= r.max_val:
                    return True, False
                if l.max_val < r.min_val:
                    return False, True
            if op == "<=":
                if l.max_val <= r.min_val:
                    return True, False
                if l.min_val > r.max_val:
                    return False, True
            return True, True  # uncertain

        # logical AND / OR
        if op in {"&&", "||"}:
            lt, lf = evaluate_condition(node.left, env)
            rt, rf = evaluate_condition(node.right, env)
            if op == "&&":
                return lt and rt, lf or rf
            else:
                return lt or rt, lf and rf

        # equality / inequality
        if op in {"==", "!="}:
            l = evaluate_expression(node.left, env)
            r = evaluate_expression(node.right, env)
            overlap = not (l.max_val < r.min_val or l.min_val > r.max_val)
            can_equal = overlap and not l.maybe_nan and not r.maybe_nan
            if op == "==":
                return can_equal, True
            else:  # "!="
                return True, not can_equal

    # unary !
    if isinstance(node, c_ast.UnaryOp) and node.op == "!":
        t, f = evaluate_condition(node.expr, env)
        return f, t

    # anything else: treat as numeric and compare with zero
    v = evaluate_expression(node, env)
    can_true = (
        v.maybe_inf
        or v.maybe_nan
        or v.min_val < 0.0
        or v.max_val > 0.0
    )
    can_false = v.min_val <= 0.0 <= v.max_val
    return can_true, can_false


# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------


def evaluate_expression(node, env) -> AbsVal:
    # constants -------------------------------------------------------------
    if isinstance(node, c_ast.Constant):
        typ = getattr(node, "type", "")
        s = node.value
        if typ == "int":
            res = AbsVal.from_values([float(int(s, 0))])
        elif typ == "float":
            res = AbsVal.from_values([float(s.rstrip("fF"))])
        else:
            res = AbsVal.top()
        _record(node, res)
        return res

    # identifiers -----------------------------------------------------------
    if isinstance(node, c_ast.ID):
        res = env.get(node.name, AbsVal.top())
        _record(node, res)
        return res

    # unary ops -------------------------------------------------------------
    if isinstance(node, c_ast.UnaryOp):
        if node.op == "+":
            res = evaluate_expression(node.expr, env)
        elif node.op == "-":
            v = evaluate_expression(node.expr, env)
            res = AbsVal(-v.max_val, -v.min_val, v.maybe_inf, v.maybe_nan)
        elif node.op == "!":
            t, f = evaluate_condition(node.expr, env)
            res = AbsVal.from_values(([1.0] if f else []) + ([0.0] if t else []))
        else:  # unsupported unary
            res = AbsVal.top()
        _record(node, res)
        return res

    # binary ops ------------------------------------------------------------
    if isinstance(node, c_ast.BinaryOp):
        if node.op in {"+", "-", "*", "/"}:
            l = evaluate_expression(node.left, env)
            r = evaluate_expression(node.right, env)
            res = arithmetic_op(node.op, l, r)
        else:  # logical / comparison – returns 0 or 1
            t, f = evaluate_condition(node, env)
            res = AbsVal.from_values(([1.0] if t else []) + ([0.0] if f else []))
        _record(node, res)
        return res

    # assignment used as an expression -------------------------------------
    if isinstance(node, c_ast.Assignment):
        val = evaluate_expression(node.rvalue, env)
        if isinstance(node.lvalue, c_ast.ID):
            env[node.lvalue.name] = val
        _record(node, val)
        return val

    # anything else – give up but stay sound --------------------------------
    res = AbsVal.top()
    _record(node, res)
    return res


# ---------------------------------------------------------------------------
# Statement executor
# ---------------------------------------------------------------------------


def execute_statement(stmt, env):
    """Execute one C statement (abstractly)."""
    if isinstance(stmt, c_ast.Decl):
        env[stmt.name] = (
            evaluate_expression(stmt.init, env) if stmt.init else AbsVal.top()
        )
        return None

    if isinstance(stmt, c_ast.Assignment):
        val = evaluate_expression(stmt.rvalue, env)
        if isinstance(stmt.lvalue, c_ast.ID):
            env[stmt.lvalue.name] = val
        return None

    if isinstance(stmt, c_ast.Return):
        return evaluate_expression(stmt.expr, env) if stmt.expr else AbsVal.top()

    if isinstance(stmt, c_ast.If):
        t, f = evaluate_condition(stmt.cond, env)
        ret_val = None
        if t:
            env_true = {k: v.copy() for k, v in env.items()}
            ret_val = execute_statement_or_block(stmt.iftrue, env_true)
        if f:
            env_false = {k: v.copy() for k, v in env.items()}
            r2 = execute_statement_or_block(stmt.iffalse, env_false)
            if ret_val is None:
                ret_val = r2
            elif r2 is not None:
                ret_val = ret_val.join(r2)
        return ret_val

    if isinstance(stmt, c_ast.Compound):
        for s in stmt.block_items or []:
            r = execute_statement(s, env)
            if r is not None:
                return r
        return None

    # any unsupported statement: no effect
    return None


def execute_statement_or_block(node, env):
    return (
        execute_statement(node, env)
        if not isinstance(node, c_ast.Compound)
        else execute_statement(node, env)
    )


def interpret_function(func_node, env):
    """Entry point: run the function body under the initial environment."""
    env = {k: v.copy() for k, v in env.items()}  # isolate caller
    return execute_statement(func_node.body, env) or AbsVal.top()


# ---------------------------------------------------------------------------
# A robust “parse C” that tries three different pre‑processors
# ---------------------------------------------------------------------------


def parse_c_file(c_file: str):
    cpp_args = [
        "-E",
        "-I",
        FAKE_LIBC_INCLUDE,
        "-D__attribute__(x)=",
        "-D__extension__=",
        "-D__asm__(x)=",
        f"-DFLT_MAX={FLT_MAX}f",
    ]

    try:
        return parse_file(c_file, use_cpp=True, cpp_args=cpp_args)
    except Exception:
        pass

    try:
        return parse_file(
            c_file, use_cpp=True, cpp_path="clang", cpp_args=cpp_args
        )
    except Exception:
        pass

    # last resort: remove pre‑processor lines completely
    with open(c_file) as fin, tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".c"
    ) as tmp:
        tmp.writelines(line for line in fin if not line.lstrip().startswith("#"))
    return parse_file(tmp.name, use_cpp=False)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) not in {4, 5}:
        print(
            "Usage: python3 fp_ai.py <c_file> <function> <input.yaml> "
            "[output.yaml]"
        )
        sys.exit(1)

    c_file, func_name, inputs_yaml = sys.argv[1:4]
    output_yaml = sys.argv[4] if len(sys.argv) == 5 else None

    ast = parse_c_file(c_file)
    func_def = next(
        (
            n
            for n in ast.ext
            if isinstance(n, c_ast.FuncDef) and n.decl.name == func_name
        ),
        None,
    )
    if func_def is None:
        sys.stderr.write(f"Function '{func_name}' not found in {c_file}\n")
        sys.exit(1)

    env = load_inputs(inputs_yaml)
    interpret_function(func_def, env)

    # -------- collate + output --------------------------------------------
    combined: dict[tuple[int, str], bool] = {}
    for line, kind, maybe in expr_report:
        key = (line, kind)
        combined[key] = combined.get(key, False) or bool(maybe)


    ordered_keys = sorted(
        combined.keys(),
        key=lambda t: (t[0], 0 if t[1] == "inf" else 1)
    )

    if output_yaml:
        records = [
            {"line": int(line), "kind": kind, "maybe": combined[(line, kind)]}
            for line, kind in ordered_keys
        ]
        with open(output_yaml, "w") as f:
            yaml.safe_dump(records, f, sort_keys=False)
    else:
        for line, kind in ordered_keys:
            print(f"{line} {kind} {combined[(line, kind)]}")


if __name__ == "__main__":
    main()
