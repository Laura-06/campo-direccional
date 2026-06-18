import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, Function, Eq, dsolve, cos, sin, pi

# Definir la variable independiente y la función incógnita
x = symbols('x')
y = Function('y')(x)

# Definir las ecuaciones diferenciales
eqs = [
    Eq(y.diff(x), sin(x) * cos(y)),
    Eq(y.diff(x), x + y),
    Eq(y.diff(x), 0.2 * x**2 + y),
    Eq(y.diff(x), y - cos((pi/2) * x))
]

# Resolver simbólicamente las ecuaciones diferenciales
soluciones = []
for eq in eqs:
    sol = dsolve(eq, y)
    if isinstance(sol, list):
        soluciones.append(sol[0].rhs)  # Tomar la primera solución si hay varias
    else:
        soluciones.append(sol.rhs)

# Parámetros para la simulación numérica
t_span = (0, 10)  # Intervalo de tiempo
y0 = [1]  # Condición inicial

# Funciones para resolver numéricamente con solve_ivp
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

# ----------- PRIMER GRÁFICO: Campos direccionales y soluciones numéricas -----------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

X, Y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))

for i in range(4):
    ax = axes[i]
    U = np.ones_like(X)
    V = funcs[i](X, Y)
    V /= np.sqrt(U**2 + V**2)  # Normalización
    
    # Resolver numéricamente
    sol_num = solve_ivp(funcs[i], t_span, y0, t_eval=np.linspace(0, 10, 100))

    # Graficar el campo direccional y la solución
    ax.quiver(X, Y, U, V, alpha=0.5)
    ax.plot(sol_num.t, sol_num.y[0], colores[i], label='Solución numérica')

    ax.set_title(titulos[i])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

# ----------- SEGUNDO GRÁFICO: Soluciones simbólicas -----------
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.axis('off')  # Ocultar los ejes

text = "\n".join([f"{titulos[i]}:\n  y = {soluciones[i]}" for i in range(4)])
ax2.text(0.05, 0.5, text, fontsize=12, verticalalignment="center")

plt.show()
