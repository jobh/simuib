from __future__ import division

import scipy
import scipy.linalg
import numpy
from dolfin import *
from block import *
from block.algebraic.trilinos import *
from block.iterative import *
from matplotlib import pyplot

set_log_level(PROGRESS)
# Function spaces, elements

N=32
mesh = UnitSquareMesh(N,N)
dim = mesh.topology().dim()

n = FacetNormal(mesh)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

v, omega = TestFunction(V), TrialFunction(V)
q, phi   = TestFunction(Q), TrialFunction(Q)

u = Function(V)
p = Function(Q)

#========
# Define forms, material parameters, boundary conditions, etc.

### Material parameters

lmbda = Constant(1.0)
mu    = Constant(1.0)
lmbda = mu = Constant(1e5)

class Permeability(Expression):
    def value_shape(self):
        return (2,2)
    def eval(self, tensor, x):
        tensor.shape = self.value_shape()
        tensor[:] = 0.0
        if 0.2 < x[1] <= 0.8:
            tensor[0,0] = 1.0
            tensor[1,1] = tensor[0,0]
        else:
            tensor[0,0] = 1e-6
            tensor[1,1] = tensor[0,0]

S = Constant(0.000)
alpha = Constant(1.0)
Lambda = Permeability()

t_n = Constant( [0.0]*dim )

dt = Constant(.02)
T = 0.1

r = Constant(0)

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(dim)
def v_D(q):
    return -Lambda*grad(q)
def b(w,r):
    return - alpha * r * div(w)

a00 = inner(grad(omega), sigma(v)) * dx
a01 = b(omega,q) * dx
a10 = b(v,phi) * dx
a11 = -(S*phi*q - dt*inner(grad(phi),v_D(q))) * dx

L0 = dot(t_n, omega) * ds
L1 = b(u,phi) * dx - (r*dt + S*p)*phi * dx

# Create boundary conditions.

bc_u_bedrock        = DirichletBC(V, [0.0]*dim, lambda x,_: near(x[1],0))
bc_p_drained_edges  = DirichletBC(Q,  0.0,      lambda x,_: near(x[1],1))

bcs = [bc_u_bedrock, bc_p_drained_edges]

# Assemble the matrices and vectors

AA, AAns = block_symmetric_assemble([[a00,a10],[a01,a11]], bcs=bcs)
bb = block_assemble([L0,L1], bcs=bcs, symmetric_mod=AAns)

[[A, B],
 [C, D]] = AA

def exact_schur():
    Ai = AmesosSolver(A)
    Sp = ML(collapse(D-C*InvDiag(A)*B))
    Si = LGMRES(D-C*Ai*B, precond=Sp, iter=1)

    SS = [[Ai, B],
          [C,  Si]]
    return block_mat(SS).scheme('sgs')

def drained_split():
    Ai = AmesosSolver(A)
    Di = AmesosSolver(D)
    SS = [[Ai, B],
          [C, Di]]
    return block_mat(SS).scheme('tgs')

def undrained_split():
    SA = collapse(A-float(1/S)*B*C)
    SAi = AmesosSolver(SA)
    Di = AmesosSolver(D)
    SS = [[SAi, B],
          [C,  Di]]
    return block_mat(SS).scheme('tgs')

def fixed_strain():
    Ai = AmesosSolver(A)
    Di = AmesosSolver(D)
    SS = [[Ai, B],
          [C, Di]]
    return block_mat(SS).scheme('tgs', reverse=True)

    #SS1 = [[Ai, 0], [0, Di]]
    #SS2 = [[1, -B*Di], [0, 1]]
    #return block_mat(SS1)*block_mat(SS2)

def fixed_stress():
    beta = 2*mu + dim*lmbda
    SD = collapse(D+dim*float(1/beta))
    SDi = AmesosSolver(SD)
    Ai = AmesosSolver(A)
    SS = [[Ai, B],
          [C, SDi]]
    return block_mat(SS).scheme('tgs', reverse=True)

def run(prec, runs=[0]):
    AAinv = LGMRES(AA, precond=eval(prec)())

    x0 = AA.create_vec()
    x0.randomize()

    xx = AAinv(initial_guess=x0, maxiter=20, tolerance=1e-10, show=2)*bb
    pyplot.semilogy(AAinv.residuals, marker='o', color='bgrkcmy'[runs[0]], label=prec)
    runs[0] += 1

run('drained_split')
#run('undrained_split')
run('fixed_stress')
run('fixed_strain')
run('exact_schur')

pyplot.legend()
pyplot.show()

print "Finished normally"
