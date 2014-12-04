from __future__ import division

from dolfin import *
from block import *

# Function spaces, elements

Nd = int(cl_args.get('dim', 2))
N  = int(cl_args.get('N', 10))
if Nd == 2:
    mesh = UnitSquareMesh(N,N)
else:
    mesh = UnitCubeMesh(N,N,N)

V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)

v, omega = TestFunction(V), TrialFunction(V)
q, phi   = TestFunction(Q), TrialFunction(Q)

u = Function(V)
p = Function(Q)

if False:
    # Perturb mesh
    mv = Function(V)
    mvv = mv.vector()
    block_vec([mvv]).randomize()
    mvv[:] = mvv*(1/3/N)
    mesh.move(mv)
    plot(mesh)

#========
# Define forms, material parameters, boundary conditions, etc.

### Material parameters

lmbda = Constant(1e4)
mu    = Constant(1e3)
dt    = Constant(.001)
b     = Constant(0)
alpha = Constant(1.0)

class Permeability(Expression):
    def value_shape(self):
        return (Nd,Nd)
    def eval(self, tensor, x):
        tensor.shape = self.value_shape()
        tensor[:] = 0.0
        for d in range(Nd):
            if 0.0 <= x[-1] < 0.499:
                tensor[d,d] = 1.0
            else:
                tensor[d,d] = delta
Lambda = Permeability()
#plot(mesh, interactive=True)

class t_n(Expression):
    def value_shape(self):
        return Nd,
    def eval(self, value, x):
        value[0] = x[0]
        value[1] = -x[1]
t_n = t_n()

T = 0.1

r = Constant(0)

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(Nd)
def v_D(q):
    return -Lambda*grad(q)
def coupling(w,r):
    return - alpha * r * div(w)

a00 = inner(grad(omega), sigma(v)) * dx
a01 = coupling(omega, q) * dx
a10 = coupling(v, phi) * dx
a11 = -(b*phi*q - dt*inner(grad(phi), v_D(q))) * dx

L0 = dot(t_n, v) * ds
L1 = coupling(u, q) * dx - (r*dt + b*p)*q * dx

# Create boundary conditions.

bc_u_bedrock        = DirichletBC(V, [0.0]*Nd, lambda x,bdry: bdry and x[-1] <= 1/N/3)
bc_p_drained_top    = DirichletBC(Q,  0.0,     lambda x,bdry: bdry and x[-1] >= 1-1/N/3)

bcs = [bc_u_bedrock, bc_p_drained_top]
#bcs = [bc_u_bedrock, None]

# Assemble the matrices and vectors

AA = block_assemble([[a00,a10],[a01,a11]])
bb = block_assemble([L0,L1])
print bb.norm()
#exit()

block_bc(bcs, True).apply(AA).apply(bb)
