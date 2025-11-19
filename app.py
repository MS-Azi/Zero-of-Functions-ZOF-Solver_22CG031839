
from flask import Flask, render_template, request
import math

app = Flask(__name__)

# ------------------ safe evaluator ------------------
def make_f(expr: str):
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith('__')}
    def f(x):
        try:
            return eval(expr, {'__builtins__': {}}, {**allowed, 'x': x})
        except Exception as e:
            raise ValueError(f"Error evaluating expression at x={x}: {e}")
    return f

# ------------------ web wrappers for methods ------------------

def bisection_web(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, 'f(a) and f(b) must have opposite signs.'
    iters = []
    for i in range(1, max_iter+1):
        c = (a + b) / 2.0
        fc = f(c)
        err = abs(b - a) / 2.0
        iters.append({'i': i, 'a': a, 'b': b, 'c': c, 'fc': fc, 'err': err})
        if abs(fc) < tol or err < tol:
            return iters, None
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return iters, None


def regula_falsi_web(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, 'f(a) and f(b) must have opposite signs.'
    iters = []
    fa, fb = f(a), f(b)
    x_old = a
    for i in range(1, max_iter+1):
        x = (a * fb - b * fa) / (fb - fa)
        fx = f(x)
        err = abs(x - x_old)
        iters.append({'i': i, 'a': a, 'b': b, 'x': x, 'fx': fx, 'err': err})
        if abs(fx) < tol or err < tol:
            return iters, None
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        x_old = x
    return iters, None


def secant_web(f, x0, x1, tol, max_iter):
    iters = []
    for i in range(1, max_iter+1):
        f0, f1 = f(x0), f(x1)
        denom = (f1 - f0)
        if denom == 0:
            return None, 'Zero denominator in secant update.'
        x2 = x1 - f1 * (x1 - x0) / denom
        err = abs(x2 - x1)
        iters.append({'i': i, 'x0': x0, 'x1': x1, 'x2': x2, 'fx2': f(x2), 'err': err})
        if abs(f(x2)) < tol or err < tol:
            return iters, None
        x0, x1 = x1, x2
    return iters, None


def newton_web(f, df, x0, tol, max_iter):
    iters = []
    x = x0
    for i in range(1, max_iter+1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            return None, 'Derivative is zero; Newton method fails.'
        x_new = x - fx / dfx
        err = abs(x_new - x)
        iters.append({'i': i, 'x': x, 'fx': fx, 'dfx': dfx, 'x_new': x_new, 'err': err})
        if abs(fx) < tol or err < tol:
            return iters, None
        x = x_new
    return iters, None


def fixed_point_web(g, x0, tol, max_iter):
    iters = []
    x = x0
    for i in range(1, max_iter+1):
        x_new = g(x)
        err = abs(x_new - x)
        iters.append({'i': i, 'x': x, 'x_new': x_new, 'err': err})
        if err < tol:
            return iters, None
        x = x_new
    return iters, None


def modified_secant_web(f, x0, delta, tol, max_iter):
    iters = []
    x = x0
    for i in range(1, max_iter+1):
        f_x = f(x)
        denom = f(x + delta * x) - f_x
        if denom == 0:
            return None, 'Zero denominator in modified secant (bad delta).'
        x_new = x - (delta * x * f_x) / denom
        err = abs(x_new - x)
        iters.append({'i': i, 'x': x, 'f_x': f_x, 'x_new': x_new, 'err': err})
        if abs(f_x) < tol or err < tol:
            return iters, None
        x = x_new
    return iters, None


# ------------------ Flask route ------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        expr = request.form.get('expr', '').strip()
        method = request.form.get('method', '')
        tol = float(request.form.get('tol', '1e-6'))
        max_iter = int(request.form.get('max_iter', '50'))

        # quick validation
        if expr == '':
            return render_template('index.html', error='Please enter function f(x).')
        f = make_f(expr)

        try:
            if method == 'bisection':
                a = float(request.form.get('a','0'))
                b = float(request.form.get('b','0'))
                iters, err = bisection_web(f, a, b, tol, max_iter)
                return render_template('index.html', result=iters, method=method, error=err)

            elif method == 'regula':
                a = float(request.form.get('a','0'))
                b = float(request.form.get('b','0'))
                iters, err = regula_falsi_web(f, a, b, tol, max_iter)
                return render_template('index.html', result=iters, method=method, error=err)

            elif method == 'secant':
                x0 = float(request.form.get('x0','0'))
                x1 = float(request.form.get('x1','0'))
                iters, err = secant_web(f, x0, x1, tol, max_iter)
                return render_template('index.html', result=iters, method=method, error=err)

            elif method == 'newton':
                dexpr = request.form.get('dexpr','').strip()
                if dexpr == '':
                    return render_template('index.html', error="Newton requires derivative f'(x).")
                df = make_f(dexpr)
                x0 = float(request.form.get('x0','0'))
                iters, err = newton_web(f, df, x0, tol, max_iter)
                return render_template('index.html', result=iters, method=method, error=err)

            elif method == 'fixed':
                gexpr = request.form.get('gexpr','').strip()
                if gexpr == '':
                    return render_template('index.html', error="Fixed point requires g(x).")
                g = make_f(gexpr)
                x0 = float(request.form.get('x0','0'))
                iters, err = fixed_point_web(g, x0, tol, max_iter)
                return render_template('index.html', result=iters, method=method, error=err)

            elif method == 'modified':
                x0 = float(request.form.get('x0','0'))
                delta = float(request.form.get('delta','1e-3'))
                iters, err = modified_secant_web(f, x0, delta, tol, max_iter)
                return render_template('index.html', result=iters, method=method, error=err)

            else:
                return render_template('index.html', error='Unsupported method (should not occur).')

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)