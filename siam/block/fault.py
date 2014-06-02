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
x = mesh.ufl_cell().x

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

v, omega = TestFunction(V), TrialFunction(V)
q, phi   = TestFunction(Q), TrialFunction(Q)

u = Function(V)
p = Function(Q)

#========
# Define forms, material parameters, boundary conditions, etc.

### Material parameters

P0    = Constant(1)
lmbda = Constant(1)
mu    = Constant(1)
dt    = Constant(1)
b     = Constant(1)
alpha = Constant(1)
Lambda = Constant(1)

p = project(-P0*x[0], Q)

t_n = Constant( [0.0]*Nd )

T = 1.0

r = Constant(0)

U = Expression([
    "-1.0/3*atan2(x[1],x[0])*sqrt(x[1]*x[1]+x[0]*x[0])*-sin(atan2(x[1],x[0]))",
    "-1.0/3*atan2(x[1],x[0])*sqrt(x[1]*x[1]+x[0]*x[0])*cos(atan2(x[1],x[0]))"
    ]);

U0=project(U,V)
interactive()

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

#bc_u = DirichletBC(V, [0.0]*Nd,  lambda x,bdry: near(x[0], 1.0))
#bc_u1 = DirichletBC(V.sub(0), 0.0,  lambda x,bdry: near(x[0], 1.0))
#bc_u2 = DirichletBC(V, [0.0]*Nd,  lambda x,bdry: near(x[0], 1.0) and abs(x[1]) <=2*eps, method='pointwise')
#bc_u3 = DirichletBC(V.sub(1), 0.0,  lambda x,bdry: near(x[0], X0) and abs(x[1]) <= 2*eps, method='pointwise')
#bc_u2 = DirichletBC(V, [0.0]*Nd,   lambda x,bdry: x[0]==1.0 and x[1]==0.0, method='pointwise')
bc_u2 = DirichletBC(V, [0.0]*Nd,   lambda x,bdry: near(x[0],0.0) and x[1]==0.0, method='pointwise')
bc_u3 = DirichletBC(V.sub(1), 0.0, lambda x,bdry: x[0] > X0 and near(x[1], 0.0), method='pointwise')
bc_u = [bc_u3, bc_u2]

#bc_p = DirichletBC(Q, -P0*x[0], lambda x,bdry: bdry and (abs(x[1]) > eps or x[0]>X0))
#bc_p_fault = DirichletBC(Q, P0, lambda x,bdry: bdry and abs(x[1]) <= eps and x[0] <= X0)
bc_p_bdry = DirichletBC(Q, -P0*x[0], lambda x,bdry: bdry and x[1] > 0.0)
bc_p_fault = DirichletBC(Q, P0, lambda x,bdry: bdry and near(x[1], 0.0) and x[0] <= X0)
#plot(bc_p)
#plot(bc_p_fault)
#interactive()
bc_p = [bc_p_bdry, bc_p_fault]

bcs = [bc_u, bc_p]

# Assemble the matrices and vectors


AA, AAns = block_symmetric_assemble([[a00,a10],[a01,a11]], bcs=bcs)
bb = block_assemble([L0,L1], bcs=bcs, symmetric_mod=AAns)
