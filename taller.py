import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, Function, Eq, dsolve, cos, sin, pi


x = symbols('x')
y = Function('y')(x)


eqs = [
    Eq(y.diff(x), sin(x) * cos(y)),
    Eq(y.diff(x), x + y),
    Eq(y.diff(x), 0.2 * x**2 + y),
    Eq(y.diff(x), y - cos((pi/2) * x))
]


soluciones = []
for eq in eqs:
    sol = dsolve(eq, y)
    if isinstance(sol, list):
        soluciones.append(sol[0].rhs)  
    else:
        soluciones.append(sol.rhs)


t_span = (0, 10) 
y0 = [1]  

def f1(t, y): return np.sin(t) * np.cos(y)
def f2(t, y): return t + y
def f3(t, y): return 0.2 * t**2 + y
def f4(t, y): return y - np.cos((np.pi/2) * t)

funcs = [f1, f2, f3, f4]
colores = ['r', 'g', 'b', 'm']
titulos = [
    "y' = sin(x) * cos(y)",
    "y' = x + y",
    "y' = 0.2x² + y",
    "y' = y - cos(π/2 * x)"
]

X, Y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))


for i in range(4):
    plt.figure(figsize=(6, 5)) 
    U = np.ones_like(X)
    V = funcs[i](X, Y)
    V /= np.sqrt(U**2 + V**2)  
    sol_num = solve_ivp(funcs[i], t_span, y0, t_eval=np.linspace(0, 10, 100))

    
    plt.quiver(X, Y, U, V, alpha=0.5)
    plt.plot(sol_num.t, sol_num.y[0], colores[i], label='Solución numérica')

    plt.title(titulos[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()  


plt.figure(figsize=(10, 6))
plt.axis('off')  

text = "\n".join([f"{titulos[i]}:\n  y = {soluciones[i]}" for i in range(4)])
plt.text(0.05, 0.5, text, fontsize=12, verticalalignment="center")

plt.show()
