import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Fundamentals of physics - Newton's Laws of Motion
# =========================================================
# Kinematics and Dynamics of particles and rigid bodies
# =========================================================

# Displacement, Velocity, and Acceleration
def kinematics_example(x0=0, v0=0, a=9.81):
    #conditions
    t = np.linspace(0, 10, 100)  # time from 0 to 10 seconds
    x = x0 + v0*t + 0.5*a*t**2  # position as a function of time
    v = v0 + a*t  # velocity as a function of time
    a = np.full_like(t, a)  # constant acceleration
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, x, label='Position (m)')
    plt.title('Kinematics: Position, Velocity, and Acceleration')
    plt.ylabel('Position (m)')
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t, v, label='Velocity (m/s)', color='orange')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t, a, label='Acceleration (m/s²)', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return(x, v, a)
kinematics_example()

# Velocity at given displacement ( v^2 relation)
def velocity_at_displacement(v0, a, x0, x):
    v = np.sqrt(v0**2 + 2*a*(x - x0))
    print(f"Velocity at displacement {x} m: {v} m/s")
    return v

# free fall example
def free_fall_example(h=45, g=9.81):
    t_fall = np.sqrt(2*h/g)
    v_impact = g * t_fall
    print(f"Time to fall from {h} m: {t_fall:.2f} s")
    print(f"Impact velocity: {v_impact:.2f} m/s")
    plt.figure(figsize=(8, 5))
    plt.plot([0, t_fall], [h, 0], marker='o')
    plt.title('Free Fall from Height')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.ylim(0, h+5)
    plt.xlim(0, t_fall+1)
    plt.grid()
    plt.show()
    return t_fall, v_impact
free_fall_example()

# ========================================================
# Notes
# ========================================================

# Kinematics Equations:
# v = v0 + a*t
# x = x0 + v0*t + 0.5*a*t^2
# v^2 = v0^2 + 2*a*(x - x0)

# Free Fall:
# Special case of motion under constant acceleration due to gravity (g ≈ 9.81 m/s²)
# Time to fall from height h: t = sqrt(2*h/g)
# Impact velocity: v = g*t  

# average velocity = (initial velocity + final velocity) / 2
# average acceleration = (final velocity - initial velocity) / time
# 
# Dimensional Analysis:
# Ensures equations are dimensionally consistent (e.g., m/s² for acceleration)
# Vector Quantities:
# Displacement, velocity, and acceleration are vector quantities with both magnitude and direction
# Newton's Laws of Motion:
# 1. An object remains at rest or in uniform motion unless acted upon by a net external force.
# 2. The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass (F = m*a).
# 3. For every action, there is an equal and opposite reaction.
# Inertial Frames of Reference:
# Newton's laws hold true in inertial frames, which are either at rest or moving with constant velocity.
# Non-inertial Frames:
# In accelerating frames, fictitious forces (e.g., centrifugal force) may need to be considered.
# Rigid Body Dynamics:
# Extension of Newton's laws to systems of particles and rigid bodies, considering rotational motion and torque.
# Work-Energy Principle:
# The work done by forces on an object results in a change in its kinetic energy (Work-Energy Theorem).
# Conservation of Momentum:
# In the absence of external forces, the total momentum of a system remains constant.
# Impulse-Momentum Theorem:
# The change in momentum of an object is equal to the impulse applied to it (Impulse = Force * time).
# Friction and Air Resistance:
# Real-world applications often involve non-conservative forces like friction and drag, affecting motion.
# Projectile Motion:
# Motion under the influence of gravity, typically in a parabolic trajectory.
# Circular Motion:
# Motion along a circular path, involving centripetal acceleration directed towards the center of the circle.
# Harmonic Motion:
# Oscillatory motion, such as springs and pendulums, characterized by restoring forces proportional to displacement.
# D'Alembert's Principle:
# A reformulation of Newton's second law, useful for analyzing dynamic systems.
# Numerical Methods:
# Computational techniques (e.g., Euler's method, Runge-Kutta) for solving complex motion equations.
# Applications:
# Engineering, biomechanics, astrophysics, and various fields rely on Newtonian mechanics for analysis and design.
# Limitations:
# Newtonian mechanics is not applicable at very high speeds (relativity) or very small scales (quantum mechanics).