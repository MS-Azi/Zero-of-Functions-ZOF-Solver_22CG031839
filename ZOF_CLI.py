#!/usr/bin/env python3
"""
ZOF_CLI.py
Command-line interface for six root-finding methods:
1. Bisection
2. Regula Falsi (False Position)
3. Secant
4. Newton-Raphson
5. Fixed Point Iteration
6. Modified Secant

Usage: python ZOF_CLI.py
Interactive prompts will collect the equation and method parameters.
"""

import math
import sys
from typing import Callable, List, Tuple, Any

# ------------------ helper: parse expression ------------------

def make_f(expr: str) -> Callable[[float], float]:
    """Return a function f(x) from a user expression. Allowed names: math.*"""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    def f(x):
        try:
            return eval(expr, {"__builtins__": {}}, {**allowed, "x": x})
        except Exception as e:
            raise ValueError(f"Error evaluating function at x={x}: {e}")
    return f

# ------------------ numerical methods ------------------

def bisection(f: Callable, a: float, b: float, tol: float, max_iter: int):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Bisection.")
    iters = []
    for i in range(1, max_iter+1):
        c = (a + b)/2.0
        fc = f(c)
        err = abs(b-a)/2.0
        iters.append((i, a, b, c, fc, err))
        if abs(fc) < tol or err < tol:
            return c, err, iters
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return (a+b)/2.0, abs(b-a)/2.0, iters


def regula_falsi(f: Callable, a: float, b: float, tol: float, max_iter: int):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Regula Falsi.")
    iters = []
    fa, fb = f(a), f(b)
    x_old = a
    for i in range(1, max_iter+1):
        x = (a*fb - b*fa)/(fb - fa)
        fx = f(x)
        err = abs(x - x_old)
        iters.append((i, a, b, x, fx, err))
        if abs(fx) < tol or err < tol:
            return x, err, iters
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        x_old = x
    return x, err, iters


def secant(f: Callable, x0: float, x1: float, tol: float, max_iter: int):
    iters = []
    for i in range(1, max_iter+1):
        f0, f1 = f(x0), f(x1)
        if (f1 - f0) == 0:
            raise ValueError("Denominator zero in secant method.")
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        err = abs(x2 - x1)
        iters.append((i, x0, x1, x2, f(x2), err))
        if abs(f(x2)) < tol or err < tol:
            return x2, err, iters
        x0, x1 = x1, x2
    return x2, err, iters


def newton_raphson(f: Callable, df: Callable, x0: float, tol: float, max_iter: int):
    iters = []
    x = x0
    for i in range(1, max_iter+1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero; Newton-Raphson fails.")
        x_new = x - fx/dfx
        err = abs(x_new - x)
        iters.append((i, x, fx, dfx, x_new, err))
        if abs(fx) < tol or err < tol:
            return x_new, err, iters
        x = x_new
    return x, err, iters


def fixed_point(g: Callable, x0: float, tol: float, max_iter: int):
    iters = []
    x = x0
    for i in range(1, max_iter+1):
        x_new = g(x)
        err = abs(x_new - x)
        iters.append((i, x, x_new, err))
        if err < tol:
            return x_new, err, iters
        x = x_new
    return x, err, iters


def modified_secant(f: Callable, x0: float, delta: float, tol: float, max_iter: int):
    iters = []
    x = x0
    for i in range(1, max_iter+1):
        f_x = f(x)
        denom = f(x + delta*x) - f_x
        if denom == 0:
            raise ValueError("Denominator zero in modified secant (bad delta).")
        x_new = x - (delta * x * f_x) / denom
        err = abs(x_new - x)
        iters.append((i, x, f_x, x_new, err))
        if abs(f_x) < tol or err < tol:
            return x_new, err, iters
        x = x_new
    return x, err, iters

# ------------------ CLI interface ------------------

def print_iter_table(rows: List[Tuple[Any, ...]]):
    for r in rows:
        print(" | ".join(str(x) for x in r))


def main():
    print("ZOF CLI - Root finding methods")
    expr = input("Enter function f(x) (use 'x' and math functions, e.g. 'x**3 - 2*x -5'): ")
    f = make_f(expr)
    print("Choose method:\n1 Bisection\n2 Regula Falsi\n3 Secant\n4 Newton-Raphson\n5 Fixed Point\n6 Modified Secant")
    choice = input("Method number: ")
    tol = float(input("Tolerance (e.g. 1e-6): ") or 1e-6)
    max_iter = int(input("Max iterations (e.g. 50): ") or 50)

    try:
        if choice == '1':
            a = float(input("a: "))
            b = float(input("b: "))
            root, err, iters = bisection(f, a, b, tol, max_iter)
            print_iter_table(iters)
        elif choice == '2':
            a = float(input("a: "))
            b = float(input("b: "))
            root, err, iters = regula_falsi(f, a, b, tol, max_iter)
            print_iter_table(iters)
        elif choice == '3':
            x0 = float(input("x0: "))
            x1 = float(input("x1: "))
            root, err, iters = secant(f, x0, x1, tol, max_iter)
            print_iter_table(iters)
        elif choice == '4':
            dexpr = input("Enter derivative f'(x) (e.g. '3*x**2 - 2'): ")
            df = make_f(dexpr)
            x0 = float(input("Initial x0: "))
            root, err, iters = newton_raphson(f, df, x0, tol, max_iter)
            print_iter_table(iters)
        elif choice == '5':
            gexpr = input("Enter iteration function g(x) (so that x = g(x)): ")
            g = make_f(gexpr)
            x0 = float(input("Initial x0: "))
            root, err, iters = fixed_point(g, x0, tol, max_iter)
            print_iter_table(iters)
        elif choice == '6':
            x0 = float(input("Initial x0: "))
            delta = float(input("Delta (relative perturbation, e.g. 1e-3): ") or 1e-3)
            root, err, iters = modified_secant(f, x0, delta, tol, max_iter)
            print_iter_table(iters)
        else:
            print("Unknown choice")
            return
        print(f"\nEstimated root: {root}\nFinal estimated error: {err}\nIterations: {len(iters)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()