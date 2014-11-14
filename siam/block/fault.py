from __future__ import division

from dolfin import *
from block import *

# Function spaces, elements

X0 = 0.0

N = int(cl_args.get('N', 10))

#eps = 3e-5
#r = Rectangle(-1, -1, 1, 1)
#r = Circle(0,0,1)
#c = Rectangle(-1, -eps, X0, eps)
#mesh = Mesh(r-c, 0)
mesh = RectangleMesh(-1.0, 0.0, 1.0, 1.0, 2*N, N)
while mesh.num_cells() < N :
    mesh = refine(mesh)

Nd = mesh.topology().dim()
x = SpatialCoordinate(mesh)

V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)

v, omega = TestFunction(V), TrialFunction(V)
q, phi   = TestFunction(Q), TrialFunction(Q)

u = Function(V)
p = Function(Q)

#========
# Define forms, material parameters, boundary conditions, etc.

### Material parameters

E = 12e9
nu = 0.4

P0    = Constant(0)
lmbda = Constant(E*nu/(1+nu)/(1-2*nu))
mu    = Constant(E/2/(1+nu))
dt    = Constant(1)
b     = Constant(1)
alpha = Constant(1)
Lambda = Constant(3e-8)

class overpressure(Expression):
    def eval(self, value, x):
        value[0] = 0
        if x[0] == -1:# and near(x[1],0.0):
            value[0] = -1e8
pressure = -P0*x[0]

p.vector()[:] = project(pressure, Q).vector()

t_n = Constant(0.0)
t_n = overpressure()

T = float(dt)*100

r = Constant(1)

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(Nd)
def v_D(q):
    return -Lambda*grad(q)
def coupling(w,r):
    return - alpha * r * div(w)

a00 = inner(grad(omega), sigma(v)) * dx
a01 = coupling(omega,q) * dx
a10 = coupling(v,phi) * dx
a11 = -(b*phi*q - dt*inner(grad(phi),v_D(q))) * dx

L0 = dot(t_n*FacetNormal(mesh), v) * ds
L1 = coupling(u, q) * dx - (r*dt + b*p)*q * dx

# Create boundary conditions.

bc_u = DirichletBC(V, [0.0]*Nd, lambda x,bdry: bdry and near(x[0], 1.0))
bc_p = DirichletBC(Q, 1e8, lambda x, bdry: bdry and near(x[0], -1.0))
bcs = [bc_u, bc_p]

# Assemble the matrices and vectors


AA = block_assemble([[a00,a10],[a01,a11]])
bb = block_assemble([L0,L1])
rhs_bc = block_bc(bcs, True).apply(AA).apply(bb)
