from __future__ import division

from dolfin import *
from block import *

# Function spaces, elements

X0 = 0.0

N = int(cl_args.get('N', 10))

eps = 3e-3
r = Rectangle(-1, -1, 1, 1)
c = Rectangle(-1, -eps, X0, eps)
mesh = Mesh(r-c, 0)
while mesh.num_cells() < N :
    mesh = refine(mesh)
# refinements: 342-1368-5472-21888
#plot(mesh, title="%d cells"%mesh.num_cells())
#interactive()

Nd = mesh.topology().dim()
x = mesh.ufl_cell().x

V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)

v, omega = TestFunction(V), TrialFunction(V)
q, phi   = TestFunction(Q), TrialFunction(Q)

u = Function(V)
p = Function(Q)

#========
# Define forms, material parameters, boundary conditions, etc.

### Material parameters

lmbda = Constant(1)
mu    = Constant(1)
dt    = Constant(1)
b     = Constant(1)
alpha = Constant(1)
Lambda = Constant(1)

t_n = Constant( [0.0]*Nd )

T = 0.1

r = Constant(0)

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

L0 = dot(t_n, omega) * ds
L1 = coupling(u,phi) * dx - (r*dt + b*p)*phi * dx

# Create boundary conditions.

bc_u = DirichletBC(V, [0.0]*Nd,  lambda x,bdry: near(x[0], 1.0))
bc_u1 = DirichletBC(V.sub(0), 0.0,  lambda x,bdry: near(x[0], 1.0))
bc_u2 = DirichletBC(V, [0.0]*Nd,  lambda x,bdry: near(x[0], 1.0) and abs(x[1]) <=2*eps, method='pointwise')
bc_u = [bc_u1, bc_u2]

bc_p = DirichletBC(Q, -x[0],     lambda x,bdry: bdry and abs(x[1]) > 2*eps)
bc_p_fault = DirichletBC(Q, 1.0, lambda x,bdry: abs(x[-1]) <= 2*eps and x[0] <= X0, method='pointwise')

bcs = [bc_u, [bc_p, bc_p_fault]]

# Assemble the matrices and vectors

AA, AAns = block_symmetric_assemble([[a00,a10],[a01,a11]], bcs=bcs)
bb = block_assemble([L0,L1], bcs=bcs, symmetric_mod=AAns)
