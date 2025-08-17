"""
A pendulum problem simulation using Newton's laws of motion.
Mass m hanging from a string of length L.
This code simulates the motion of a simple pendulum using numerical integration.
swings in a gravitational field.
angle θ^t measures deviation from vertical.

geometry (constaints)

person A position vector in cartesian co-ordinartes:
\\vec{r}(t) = L \\sin(\\theta(t)) \\, \\hat{i} - L \\cos(\\theta(t)) \\, \\hat{j}

x = L \\sin(\\theta)
y = -L \\cos(\\theta)

Velocity and Acceleration vectors:
differentiate position vector with respect to time t:

\vec{v}(t) = \frac{d\vec{r}}{dt}
\vec{a}(t) = \frac{d^2\vec{r}}{dt^2}
"""
import sympy as sp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basic_tools import vectors, derivative, t
#--- Define pendulum angle as a function of time ---
theta = sp.Function('theta')(t)  # θ(t) is the angle of the pendulum as a function of time
L, g = sp.symbols('L g', real=True, positive=True)
#--- Position vector in Cartesian coordinates ---
r = vectors(L*sp.sin(theta), -L*sp.cos(theta))  # r(t) = L sin(θ(t)) i - L cos(θ(t)) j
#--- Velocity vector ---
v = derivative(r)  # v(t) = dr/dt
#--- Acceleration vector ---
a = derivative(v)  # a(t) = dv/dt
#--- Forces acting on the pendulum ---
"""
forces acting: gravity F_g = mgj downward , tension in the string T unknown along string
Equation of motion:
ma = T + mg

projecting onto tangential direction (the only one that matters for motion):
θ''(t) + g/L θ(t) = 0

then small angle approximation:
θ''(t) + g/L θ(t) = 0
simple harmonic oscillor motion
θ(t) = θ_0 cos(√(g/L) t + v_0.√(g/L)(t))

"""
# define ODS theta '' (t) + g/L θ(t) = 0
ode_small = sp.Eq(sp.diff(theta, t, t) + (g/L) * theta, 0)  # Small angle approximation
solution = sp.dsolve(ode_small)  # θ(t) = θ_0 cos(√(g/L) t + v_0.√(g/L)(t))
solution # Display the solution to the ODE

"""
Non linear pendulum equation:

derivation, using Newton's laws of motion, leads to a second-order differential equation for the angle θ(t):
θ''(t) + (g/L) * sin(θ(t)) = 0
# This equation is nonlinear due to the sine term.
# For small angles, we can use the small angle approximation:
# sin(θ) ≈ θ, leading to the linearized equation:
using conserved energy, elliptic integrals, asymptotic expansions, or numerical methods.

set up and notation
θ(t) = angle of the pendulum at time t from the vertical (radians)
L = length of rigit, massless rod(m)
m = mass of the bob (kg)
g = acceleration due to gravity (m/s²)
θ_. = dθ/dt θ_.. = d^2θ/dt^2
positive direction of θ is counterclockwise
assume motion confined to a plane and the rod is inextensible (length L is constant).
"""

"""
using Lagrangian mechanics, we can derive the equations of motion for the pendulum.
The Lagrangian is given by:
L = T - V

where T is the kinetic energy and V is the potential energy.
so we use mL^2 θ_.. = -mgL sin(θ)
cancel mL from both sides:
θ_.. = -g/L sin(θ) = 0

first integral (energy Conservation):
mulitply θ_.. = -g/L sin(θ) = 0 by θ and integrate with respect to time:

θ_. θ_.. + g/L sin(θ,θ_.) = 0
# integrate to get the first integral of motion

d/dt(1/2(θ)^2 - g/L cos(θ)) = 0

energy E is conserved. Multiply by mL^2 to get Mechanical energy:

E = 1/2ml^2 θ_.^2 - mgL cos(θ) = constant 

this is the total mechanical energy of the pendulum system which remains constant over time

divide by mL^2 to get the dimensionless form:

1/2 θ_.^2 - (g/L) cos(θ) = ℇ 

if the initial consitions are θ(0) = θ_0 and θ_(0) = θ_0, then the energy is given by:

ℇ = (g/L)(1 -cos(θ_0)

solving for θ_., we get: 

θ_. = ±√(2(g/L)(cos(θ) - cos(θ_0)))

then we look to the elliptic integral of the first kind to solve for θ(t):
to get T period of a full oscillation for ampltiudes θ_0 < π/2, we can use the formula:

dt = dθ/θ_. = dθ / √(2(g/L)(cos(θ) - cos(θ_0)))

one quarter period corresponds to motion form θ_0 we can integrate from θ_0 to 0_0 usng the elliptic integral of the first kind:

defined by 

k ≣ sin(θ_0/2)

T = 4√(L/g) K(k)

where K(k) is the complete elliptic integral of the first kind, defined as:
K(k) = ∫[0,π/2] dφ / √(1 - k^2 sin^2(φ))

for small amplitude θ_o -> 0, we can use the small angle approximation and K(k) ≈ π/2(1 + k^2/4 + k^4/64 + ...)

T_0 = 2π √(L/g)  # period of small oscillations

as θ_o -> π (near inverted pendulum), the period diverges to infinity, indicating that the pendulum will not complete a full oscillation.

Series? Asymptotic expansion for period 

one convenient expansion in powers of sin^2(θ_0/2) or θ^2_0:

using expansion of K(k) in powers of k^2:

Series / asymptotic expansion for period (large-but-finite amplitude)

One convenient expansion in powers of \\sin^2(\\theta_0/2) or \\theta_0^2:

Using expansion of K(k),

T(θ_0) = T_0(1 + 1/16(θ_0^2) + 11/3072(θ_0^4) + ...)

so the leading correction is +1/16(θ_0^2) to the period of small oscillations T_0 = 2π √(L/g).
    
"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def exact_period(L, g, theta_0):
    # theta_0 is in radians
    k = np.sin(theta_0 / 2)  # k = sin(θ_0/2)
    K = special.ellipk(k**2)  # Complete elliptic integral of the first kind
    return 4 * np.sqrt(L / g) * K  # T = 4√(L/g) K(k)

g, L = 9.81, 1.0  # gravitational acceleration and length of the pendulum

def pendulum(t, y):
    theta, omega = y  # y[0] = theta, y[1] = omega
    dydt = [omega, -g / L * np.sin(theta)]  # dtheta/dt = omega, domega/dt = -g/L * sin(theta)
    return dydt

def simulate_pendulum(theta_0, omega_0, t_max, dt):
    t = np.arange(0, t_max, dt)
    y = np.zeros((len(t), 2))
    y[0] = [theta_0, omega_0]  # initial conditions
    for i in range(1, len(t)):
        dydt = pendulum(t[i-1], y[i-1])
        y[i] = y[i-1] + np.array(dydt) * dt
    return t, y

def plot_pendulum_motion(t, y):
    plt.figure(figsize=(10, 5))
    plt.plot(t, y[:, 0], label='Theta (rad)', color='blue')
    plt.plot(t, y[:, 1], label='Omega (rad/s)', color='orange')
    plt.title('Pendulum Motion')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle and Angular Velocity')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
theta_0 = np.pi / 4  # initial angle in radians
omega_0 = 0  # initial angular velocity
t_max = 10  # total time for simulation in seconds
dt = 0.01  # time step for simulation in seconds
t, y = simulate_pendulum(theta_0, omega_0, t_max, dt)
plot_pendulum_motion(t, y)
T_exact = exact_period(L, g, theta_0)
print(f'The exact period of small oscillations is {T_exact:.4f} seconds.')

def verlet(theta0, omega0, L, g, h, n_steps):
    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    theta[0] = theta0
    omega[0] = omega0
    for n in range(n_steps - 1):
        # Verlet integration step
        omega_half = omega[n] + 0.5 * h * (-g / L * np.sin(theta[n]))  # Half-step for omega
        omega[n + 1] = omega[n] + 0.5 * (-g / L * np.sin(theta[n]) - g / L * np.sin(theta[n + 1])) * h
        omega[n + 1] = omega_half - 0.5 * h * (g / L)*np.sin(theta[n + 1])  # Update omega using the new theta
    return theta, omega
# Example usage of Verlet integration
h = 0.01  # time step for Verlet integration
n = 50000
theta, omega = verlet(1.0, 0.0, L=1.0, g=9.81, h=h, n_steps=n)

