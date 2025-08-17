"""
Jacobi elliptic functions, show a short derivation sketch, state the closed form, and give runnable Python code (using scipy.special.ellipj / ellipk) so you can plot and compare with numeric integration.

theta(t): angle (radians)

L: pendulum length (m)

g: gravity (m/s²)

\omega_0 \equiv \sqrt{g/L}
 {theta_0: maximum amplitude (initial angle when \dot\theta(0)=0)

k \equiv \sin\!\big(\tfrac{\theta_0}{2}\big) (elliptic modulus)

⸻

Quick derivation sketch (transform to elliptic form)

Start from the first integral (energy), for initial \theta(0)=\theta_0,\ \dot\theta(0)=0:

\dot theta^2 = frac{2g}{L}big(\cos theta - \cos theta_0\big)
= \frac{4g}{L}\Big(\sin^2\!\frac{\theta_0}{2} - \sin^2\!\frac{\theta}{2}\Big).

Let \phi(t)=\dfrac{\theta(t)}{2}. Then

\dot\phi = \tfrac12\dot\theta = \pm \sqrt{\frac{g}{L}}\,\sqrt{\,k^2 - \sin^2\phi\, }.

Rearrange and separate variables:

\frac{d\phi}{\sqrt{\,k^2 - \sin^2\phi\,}} = \sqrt{\frac{g}{L}}\,dt.

Now use the substitution \sin\phi = k\,\operatorname{sn}(u,k) (or equivalently introduce an amplitude function \psi where \sin\phi = k\sin\psi, then \psi=\operatorname{am}(u,k)). The left-hand integral is the incomplete elliptic integral of the first kind, whose inversion yields Jacobi elliptic functions. After fixing the integration constant to match \phi(0)=\theta_0/2 (and \dot\phi(0)=0), one convenient explicit form is obtained with the Jacobi cn function.

⸻

Closed-form solution (convenient phase choice)

With the modulus k=\sin(\theta_0/2) and \omega_0=\sqrt{g/L}:

\boxed{\;\theta(t) \;=\; 2\,\arcsin\!\Big( k\,\operatorname{cn}\!\big(\omega_0\,t \,,\, k\big)\Big)\;}

This form satisfies \theta(0)=2\arcsin(k)=\theta_0 and \dot\theta(0)=0.

Notes:
	•	\operatorname{cn}(u,k) is the Jacobi cosine — for k\to 0 it reduces to \cos u, recovering the small-angle harmonic solution \theta(t)\approx\theta_0\cos(\omega_0 t).
	•	Another equivalent form (phase-shifted) uses \operatorname{sn}:
\theta(t)=2\arcsin\!\Big( k\,\operatorname{sn}\!\big(\omega_0 t + K(k),\,k\big)\Big),
where K(k) is the complete elliptic integral of the first kind. These forms are related by standard shifts/identities of Jacobi functions.

⸻

Period (restate)

The exact period as a function of amplitude \theta_0 is

\boxed{T(\theta_0) = 4\sqrt{\frac{L}{g}}\,K\!\big(k\big),\qquad k=\sin\!\frac{\theta_0}{2},}

where K(k)=\displaystyle\int_0^{\pi/2}\!\frac{d\phi}{\sqrt{1-k^2\sin^2\phi}} is the complete elliptic integral.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.integrate import solve_ivp

g = 9.81
L = 1.0
omega0 = np.sqrt(g / L)
theta0 = np.pi / 4  # Initial angle in radians
k = np.sin(theta0 / 2)

#time array
tmax = 10  # seconds
nt = 1000
t = np.linspace(0, tmax, nt)

#--- Exact (elliptic / Jacobi cn) solution ---
m = k**2  # modulus squared
u = omega0 * t  # argument for Jacobi functions
sn, cn, dn, ph = special.ellipj(u, m)  # Jacobi elliptic functions
theta_eplliptic = 2.0 * np.arcsin(k * cn)  # Exact solution using Jacobi cn
#--- Numerical integration ---

def pen(t, y):
    th, w = y
    return [w, -g / L * np.sin(th)]

y0 = [theta0, 0.0]  # Initial conditions: theta(0) = theta0, omega(0) = 0
sol = solve_ivp(pen, [0, tmax], y0, t_eval=t)
theta_num = sol.y[0]  # Numerical solution for theta
#--- Plotting the results ---
plt.figure(figsize=(10, 6))
plt.plot(t, theta_eplliptic, label='Exact (elliptic / Jacobi cn)', color='blue')
plt.plot(t, theta_num, label='Numerical Integration', linestyle='--', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Theta (radians)')
plt.title('Pendulum Motion: Exact vs Numerical Solution')
plt.legend()
plt.grid()
plt.show()
#--- Phase portrait comparison ---
plt.figure(figsize=(10, 6))
plt.plot(theta_num, sol.y[1], label='Numerical Phase Portrait', color='orange')
plt.plot(theta_eplliptic, np.zeros_like(theta_eplliptic), label='Exact Phase Portrait', color='blue')
plt.xlabel('Theta (radians)')
plt.ylabel('Omega (rad/s)')
plt.title('Phase Portrait of Pendulum Motion')
plt.legend()
plt.grid(True)
plt.show()

#--- Period calculation ---
K = special.ellipk(m)  # Complete elliptic integral of the first kind
T_exact = 4 * np.sqrt(L / g) * K  # Exact period
T_small = 2 * np.pi*np.sqrt(L/g)
print(f'Exact period for theta0={theta0:.2f} rad: T_exact = {T_exact:.4f} s')
print(f'Approximate period for theta0={theta0:.2f} rad: T_small = {T_small:.4f} s')