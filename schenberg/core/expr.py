"""A tiny symbolic expression IR for pricing formulas.

A formula is no longer an opaque Polars closure: it is an :class:`Expr` tree built
from ``var``/``lit`` and a small set of operators (``+ - * / ** -``) and functions
(``exp``, ``log``, ``abs_``, ``sqrt``, ``where`` + comparisons). The *same* tree is
interpreted many ways:

* :func:`compile_polars` — to a lazy ``pl.Expr`` (execution, now);
* :func:`grad` — to an analytic derivative via JAX autodiff (greeks, later);
* :func:`to_latex` — to LaTeX *derived from the formula itself*, so the rendered
  math can never drift from what runs.

This is deliberately **not** SymPy: no solver, no simplification, no algebraic
expansion, no integration. Just a small IR the backends can walk.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

_UNARY = {"neg", "exp", "log", "abs", "sqrt"}
_BINARY = {"add", "sub", "mul", "div", "pow", "gt", "ge", "lt", "le", "eq"}


@dataclass(frozen=True, slots=True)
class Expr:
    """One node of the formula IR: an operator over child ``args``, a named
    variable, or a literal value."""

    op: str
    args: tuple[Expr, ...] = ()
    name: str | None = None
    value: Any | None = None

    # arithmetic ---------------------------------------------------------------
    def __add__(self, other: Any) -> Expr:
        return Expr("add", (self, lit(other)))

    def __radd__(self, other: Any) -> Expr:
        return Expr("add", (lit(other), self))

    def __sub__(self, other: Any) -> Expr:
        return Expr("sub", (self, lit(other)))

    def __rsub__(self, other: Any) -> Expr:
        return Expr("sub", (lit(other), self))

    def __mul__(self, other: Any) -> Expr:
        return Expr("mul", (self, lit(other)))

    def __rmul__(self, other: Any) -> Expr:
        return Expr("mul", (lit(other), self))

    def __truediv__(self, other: Any) -> Expr:
        return Expr("div", (self, lit(other)))

    def __rtruediv__(self, other: Any) -> Expr:
        return Expr("div", (lit(other), self))

    def __pow__(self, other: Any) -> Expr:
        return Expr("pow", (self, lit(other)))

    def __rpow__(self, other: Any) -> Expr:
        return Expr("pow", (lit(other), self))

    def __neg__(self) -> Expr:
        return Expr("neg", (self,))

    # comparisons (for `where`) ------------------------------------------------
    def __gt__(self, other: Any) -> Expr:
        return Expr("gt", (self, lit(other)))

    def __ge__(self, other: Any) -> Expr:
        return Expr("ge", (self, lit(other)))

    def __lt__(self, other: Any) -> Expr:
        return Expr("lt", (self, lit(other)))

    def __le__(self, other: Any) -> Expr:
        return Expr("le", (self, lit(other)))

    def __eq__(self, other: Any) -> Expr:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return Expr("eq", (self, lit(other)))

    def __hash__(self) -> int:
        return id(self)


# ---- constructors ------------------------------------------------------------


def var(name: str) -> Expr:
    """A named input: resolves to a column (Polars) or an env entry (numeric)."""
    return Expr("var", name=name)


def lit(x: Any) -> Expr:
    """A literal constant (passes through if already an :class:`Expr`)."""
    return x if isinstance(x, Expr) else Expr("lit", value=x)


def exp(x: Any) -> Expr:
    return Expr("exp", (lit(x),))


def log(x: Any) -> Expr:
    return Expr("log", (lit(x),))


def abs_(x: Any) -> Expr:
    return Expr("abs", (lit(x),))


def sqrt(x: Any) -> Expr:
    return Expr("sqrt", (lit(x),))


def where(cond: Expr, a: Any, b: Any) -> Expr:
    """Branch: ``a`` where ``cond`` is true, else ``b``."""
    return Expr("where", (cond, lit(a), lit(b)))


# ---- backend: Polars ---------------------------------------------------------

_POLARS_BINARY: dict[str, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
}


def compile_polars(e: Expr) -> pl.Expr:
    """Interpret the IR as a lazy Polars expression."""
    match e:
        case Expr(op="var", name=name):
            assert name is not None
            return pl.col(name)
        case Expr(op="lit", value=value):
            return pl.lit(value)
        case Expr(op="neg", args=(x,)):
            return -compile_polars(x)
        case Expr(op="exp", args=(x,)):
            return compile_polars(x).exp()
        case Expr(op="log", args=(x,)):
            return compile_polars(x).log()
        case Expr(op="abs", args=(x,)):
            return compile_polars(x).abs()
        case Expr(op="sqrt", args=(x,)):
            return compile_polars(x).sqrt()
        case Expr(op="where", args=(cond, a, b)):
            return (
                pl.when(compile_polars(cond)).then(compile_polars(a)).otherwise(compile_polars(b))
            )
        case Expr(op=op, args=(a, b)) if op in _POLARS_BINARY:
            return _POLARS_BINARY[op](compile_polars(a), compile_polars(b))
        case _:
            raise NotImplementedError(f"compile_polars: {e!r}")


# ---- backend: numeric (numpy / JAX) ------------------------------------------

_NUMERIC_BINARY: dict[str, Callable[[Any, Any], Any]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
}


def _eval_numeric(e: Expr, env: Mapping[str, Any], xp: Any) -> Any:
    """Evaluate the IR with an array module ``xp`` (``numpy`` or ``jax.numpy``)."""
    match e:
        case Expr(op="var", name=name):
            assert name is not None
            return env[name]
        case Expr(op="lit", value=value):
            return value
        case Expr(op="neg", args=(x,)):
            return -_eval_numeric(x, env, xp)
        case Expr(op="exp", args=(x,)):
            return xp.exp(_eval_numeric(x, env, xp))
        case Expr(op="log", args=(x,)):
            return xp.log(_eval_numeric(x, env, xp))
        case Expr(op="abs", args=(x,)):
            return xp.abs(_eval_numeric(x, env, xp))
        case Expr(op="sqrt", args=(x,)):
            return xp.sqrt(_eval_numeric(x, env, xp))
        case Expr(op="where", args=(cond, a, b)):
            return xp.where(
                _eval_numeric(cond, env, xp),
                _eval_numeric(a, env, xp),
                _eval_numeric(b, env, xp),
            )
        case Expr(op=op, args=(a, b)) if op in _NUMERIC_BINARY:
            return _NUMERIC_BINARY[op](_eval_numeric(a, env, xp), _eval_numeric(b, env, xp))
        case _:
            raise NotImplementedError(f"_eval_numeric: {e!r}")


def compile_numeric(e: Expr) -> Callable[[Mapping[str, Any], Any], Any]:
    """A callable ``f(env, xp)`` evaluating the IR over an array module ``xp``."""

    def f(env: Mapping[str, Any], xp: Any) -> Any:
        return _eval_numeric(e, env, xp)

    return f


def grad(e: Expr, wrt: str) -> Callable[[Mapping[str, Any]], Any]:
    """An elementwise analytic derivative ``d(expr)/d(wrt)`` via JAX autodiff.

    Returns ``df(env)`` that, given arrays for every variable, returns the
    elementwise partial derivative w.r.t. ``wrt``. Requires JAX::

        df = grad((var("F") - var("K")) * exp(-var("r") * var("T")), "F")
        df({"F": [..], "K": [..], "r": [..], "T": [..]})  # == exp(-rT)
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError("grad() needs JAX: install schenberg[jax]") from exc

    def df(env: Mapping[str, Any]) -> Any:
        others = [k for k in env if k != wrt]

        def scalar(x: Any, rest: tuple[Any, ...]) -> Any:
            local = dict(zip(others, rest, strict=True))
            local[wrt] = x
            return _eval_numeric(e, local, jnp)

        batched = jax.vmap(jax.grad(scalar), in_axes=(0, 0))
        x_arr = jnp.asarray(env[wrt], dtype=float)
        rest_arrs = tuple(jnp.asarray(env[k], dtype=float) for k in others)
        return batched(x_arr, rest_arrs)

    return df


# ---- backend: LaTeX ----------------------------------------------------------

_LATEX_BINOP = {
    "add": "+",
    "sub": "-",
    "gt": ">",
    "ge": r"\geq",
    "lt": "<",
    "le": r"\leq",
    "eq": "=",
}


def to_latex(e: Expr) -> str:
    """Render the IR as LaTeX — derived from the formula, single source of truth."""

    def go(node: Expr, parent_prec: int = 0) -> str:
        match node:
            case Expr(op="var", name=name):
                assert name is not None
                return _latex_symbol(name)
            case Expr(op="lit", value=value):
                return f"{value}"
            case Expr(op="neg", args=(x,)):
                return f"-{go(x, 3)}"
            case Expr(op="exp", args=(x,)):
                return f"e^{{{go(x)}}}"
            case Expr(op="log", args=(x,)):
                return rf"\ln\left({go(x)}\right)"
            case Expr(op="abs", args=(x,)):
                return rf"\left|{go(x)}\right|"
            case Expr(op="sqrt", args=(x,)):
                return rf"\sqrt{{{go(x)}}}"
            case Expr(op="div", args=(a, b)):
                return rf"\frac{{{go(a)}}}{{{go(b)}}}"
            case Expr(op="mul", args=(a, b)):
                inner = rf"{go(a, 2)} \cdot {go(b, 2)}"
                return rf"\left({inner}\right)" if parent_prec > 2 else inner
            case Expr(op="pow", args=(a, b)):
                return f"{go(a, 4)}^{{{go(b)}}}"
            case Expr(op="where", args=(cond, a, b)):
                return (
                    rf"\begin{{cases}} {go(a)} & {go(cond)} \\ "
                    rf"{go(b)} & \text{{else}} \end{{cases}}"
                )
            case Expr(op=op, args=(a, b)) if op in _LATEX_BINOP:
                inner = f"{go(a, 1)} {_LATEX_BINOP[op]} {go(b, 1)}"
                return rf"\left({inner}\right)" if parent_prec > 1 else inner
            case _:
                raise NotImplementedError(f"to_latex: {node!r}")

    return go(e)


def _latex_symbol(name: str) -> str:
    return rf"\mathrm{{{name.replace('_', r'\_')}}}"
