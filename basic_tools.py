import sympy as sp

# --- Define Symbols ---

t = sp.symbols('t', real=True)
x, y, z = sp.Function('x')(t), sp.Function('y')(t), sp.Function('z')(t)


#---- vector operations ----

def vectors(*components):
    """Create a vector from its components."""
    return sp.Matrix(components)

def magnitude(vector):
    """Calculate the magnitude of a vector."""
    return sp.sqrt(vector.dot(vector))

def unit_vector(vector):
    """Calculate the unit vector of a given vector."""
    mag = magnitude(vector)
    if mag == 0:
        raise ValueError("Cannot compute unit vector for zero vector.")
    return vector / mag

def derivative(vector):
    """Calculate the derivative of a vector."""
    return sp.Matrix([sp.diff(comp, t) for comp in vector])

r = vectors('x', 'y', 'z') # position vector
v = derivative(r) # velocity vector
a = derivative(v) # acceleration vector

print("r(t) =", r)
print("v(t) =", v)
print("a(t) =", a)
