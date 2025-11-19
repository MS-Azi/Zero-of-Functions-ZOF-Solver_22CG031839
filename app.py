from flask import Flask, render_template, request, redirect, url_for
import math

app = Flask(__name__)

# reuse the same simple evaluator as CLI
def make_f(expr: str):
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith('__')}
    def f(x):
        return eval(expr, {'__builtins__': {}}, {**allowed, 'x': x})
    return f

# ================== ALL SIX METHODS FOR WEB ==================

# 1. Bisection (kept concise — same algorithms as CLI)
# ... For brevity in this file, include essential methods used by web UI

def bisection_web(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, 'f(a) and f(b) must have opposite signs.'
    iters = []
    for i in range(1, max_iter+1):
        c = (a+b)/2
        fc = f(c)
        err = abs(b-a)/2
        iters.append({'i':i,'a':a,'b':b,'c':c,'fc':fc,'err':err})
        if abs(fc) < tol or err < tol:
            return iters, None
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return iters, None

# ------------------------------

def regula_falsi_web(f,a,b,tol,max_iter):
    if f(a)*f(b) >= 0:
        return None, 'f(a) and f(b) must have opposite signs.'
    iters=[]
    fa,fb=f(a),f(b)
    x_old=a
    for i in range(1,max_iter+1):
        x=(a*fb - b*fa)/(fb-fa)
        fx=f(x)
        err=abs(x-x_old)
        iters.append({'i':i,'a':a,'b':b,'x':x,'fx':fx,'err':err})
        if abs(fx)<tol or err<tol:
            return iters,None
        if fa*fx<0:
            b,fb=x,fx
        else:
            a,fa=x,fx
        x_old=x
    return iters,None

# 3. Secant (example)
def secant_web(f, x0, x1, tol, max_iter):
    iters = []
    for i in range(1, max_iter+1):
        f0, f1 = f(x0), f(x1)
        if (f1 - f0) == 0:
            return None, 'Denominator zero in secant.'
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        err = abs(x2 - x1)
        iters.append({'i':i,'x0':x0,'x1':x1,'x2':x2,'fx2':f(x2),'err':err})
        if abs(f(x2)) < tol or err < tol:
            return iters, None
        x0, x1 = x1, x2
    return iters, None

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        expr = request.form['expr']
        method = request.form['method']
        tol = float(request.form.get('tol','1e-6'))
        max_iter = int(request.form.get('max_iter','50'))
        f = make_f(expr)
        try:
            if method == 'bisection':
                a = float(request.form['a'])
                b = float(request.form['b'])
                iters, err = bisection_web(f,a,b,tol,max_iter)
                return render_template('index.html', result=iters, method=method, error=err)
            elif method == 'secant':
                x0 = float(request.form['x0'])
                x1 = float(request.form['x1'])
                iters, err = secant_web(f,x0,x1,tol,max_iter)
                return render_template('index.html', result=iters, method=method, error=err)
            else:
                return render_template('index.html', error='Unsupported method (should not occur)..')
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)