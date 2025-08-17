"""
Start point (exact)

T(θ_0) = 4√(L/g)K(k), where k = sin(θ_0/2) and K(k) is the complete elliptic integral of the first kind.

its normalised form is:

T(θ_0)/T_0 = 2/π K(k), where T_0 = 2√(L/g)π

expand K(k) in powers of m = k^2:
   K(m) = π/2(1 + 1/4(m)^2 + 9/64(m)^3 + ...) m = K^2

then substitute m = k^2 = sin^2(θ_0/2) into the expansion

which results in the series expansion for the period of a pendulum with large amplitude:

\boxed{\frac{T(\theta_0)}{T_0} = 1
\;+\; \frac{1}{16}\theta_0^2
\;+\; \frac{11}{3072}\theta_0^4
\;+\; \frac{173}{737280}\theta_0^6
\;+\; \frac{22931}{1321205760}\theta_0^8
\;+\; \mathcal{O}(\theta_0^{10})}

so the period expansion up to \mathcal{O}(\theta_0^8) is:
T(\theta_0) \approx T_0\left(1 + \frac{1}{16}\theta_0^2 + \frac{11}{3072}\theta_0^4 + \frac{173}{737280}\theta_0^6 + \frac{22931}{1321205760}\theta_0^8\right).

Interpretation & accuracy
	•	The leading correction is \tfrac{1}{16}\theta_0^2 \approx 0.0625\theta_0^2. For \theta_0=0.5 rad this gives a +1.56% correction; the full series converges quickly for \theta_0\lesssim 1 rad.
	•	The power-series is not good as \theta_0 approaches \pi (it diverges / fails to represent the logarithmic blow-up).

⸻

2) Large-amplitude asymptotics (near the inverted point \theta_0\to\pi)

As k\to 1^{-} the elliptic integral has a logarithmic singularity:
K(k) \sim \ln\!\left(\frac{4}{\sqrt{1-k^2}}\right) + \text{(regular corrections)}.
With k=\sin(\theta_0/2), we have 1-k^2=\cos^2(\theta_0/2) and \sqrt{1-k^2}=\cos(\theta_0/2). Thus

K(k) \sim \ln\!\left(\frac{4}{\cos(\theta_0/2)}\right).

Hence the period behaves as

\boxed{T(\theta_0) \sim 4\sqrt{\frac{L}{g}}\;\ln\!\left(\frac{4}{\cos(\theta_0/2)}\right),\qquad\theta_0\to\pi^-}

If you parameterize the closeness to \pi by \varepsilon=\pi-\theta_0 (small \varepsilon), then \cos(\theta_0/2)=\cos(\pi/2 - \varepsilon/2)=\sin(\varepsilon/2)\sim \varepsilon/2. Substituting:

\boxed{T(\theta_0) \sim 4\sqrt{\frac{L}{g}}\;\ln\!\left(\frac{8}{\varepsilon}\right)
\;=\;4\sqrt{\frac{L}{g}}\;\ln\!\left(\frac{8}{\pi-\theta_0}\right),\qquad \varepsilon\ll 1.}

So period diverges logarithmically as the amplitude approaches the inverted position.

Refinements: one can add constant + regular series corrections to the log term; the leading singular behaviour is this log.

"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Parameters
g = 9.81
L = 1.0
T0 = 2*np.pi*np.sqrt(L/g)

def T_exact(theta0):
    k = np.sin(theta0/2.0)
    m = k**2
    K = special.ellipk(m)    # scipy uses m = k^2
    return 4.0 * np.sqrt(L/g) * K

# Power-series approximation up to theta0^8
def T_series(theta0):
    x = theta0
    # coefficients from expansion T/T0 = 1 + a2 x^2 + a4 x^4 + ...
    a2 = 1/16
    a4 = 11/3072
    a6 = 173/737280
    a8 = 22931/1321205760
    factor = 1 + a2*x**2 + a4*x**4 + a6*x**6 + a8*x**8
    return T0 * factor

# Logarithmic asymptotic for theta0 close to pi
def T_log_asym(theta0):
    eps = np.pi - theta0
    if eps <= 0:
        return np.inf
    return 4.0 * np.sqrt(L/g) * np.log(8.0/eps)

# Test amplitudes
thetas = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]  # radians (3.0 rad is large, near pi ~3.1416)
print("theta0  T_exact   T_series  rel_err_series  T_log_asym   note")
for th in thetas:
    Tex = T_exact(th)
    Ts = T_series(th)
    relerr = (Ts - Tex)/Tex
    Tl = T_log_asym(th)
    note = ""
    if th < 1.0:
        note = "small amp good"
    elif th > 2.5:
        note = "near large-amplitude"
    print(f"{th:5.2f}  {Tex:8.5f}  {Ts:8.5f}  {relerr:12.6e}  {Tl:8.5f}  {note}")

# Optional plot: T(theta0) exact vs series
ths = np.linspace(0.0, 3.05, 300)
Texs = np.array([T_exact(t) for t in ths])
Tss = np.array([T_series(t) for t in ths])

plt.figure(figsize=(8,4))
plt.plot(ths, Texs, label='Exact (elliptic)')
plt.plot(ths, Tss, '--', label='Series up to theta^8')
plt.xlabel('theta0 (rad)')
plt.ylabel('Period T (s)')
plt.legend()
plt.grid(True)
plt.show()