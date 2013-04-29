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
Nd = mesh.topology().dim()

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

b = Constant(1e-12)
alpha = Constant(1.0)
Lambda = Permeability()

t_n = Constant( [0.0]*Nd )

dt = Constant(.02)
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

bc_u_bedrock        = DirichletBC(V, [0.0]*Nd, lambda x,_: near(x[1],0))
bc_p_drained_edges  = DirichletBC(Q,  0.0,     lambda x,_: near(x[1],1))

bcs = [bc_u_bedrock, bc_p_drained_edges]

# Assemble the matrices and vectors

AA, AAns = block_symmetric_assemble([[a00,a10],[a01,a11]], bcs=bcs)
bb = block_assemble([L0,L1], bcs=bcs, symmetric_mod=AAns)

[[A,  B],
 [BT, C]] = AA

def exact_schur():
    Ai = AmesosSolver(A)
    Sp = ML(collapse(C-BT*InvDiag(A)*B))
    Si = LGMRES(C-BT*Ai*B, precond=Sp, iter=1)

    SS = [[Ai, B],
          [BT, Si]]
    return block_mat(SS).scheme('sgs')

def inexact_schur():
    Ai = ML(A)
    Sp = ML(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp]]
    return block_mat(SS).scheme('tgs')

def exact_A_inexact_schur():
    Ai = AmesosSolver(A)
    Sp = ML(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp]]
    return block_mat(SS).scheme('tgs')

def drained_split():
    Ai = AmesosSolver(A)
    Ci = AmesosSolver(C)
    SS = [[Ai, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs')

def undrained_split():
    SA  = collapse(A-1/float(b)*B*BT)
    SAi = AmesosSolver(SA)
    Ci  = AmesosSolver(C)
    SS  = [[SAi, B],
           [BT,  Ci]]
    return block_mat(SS).scheme('tgs')

def fixed_strain():
    Ai = AmesosSolver(A)
    Ci = AmesosSolver(C)
    SS = [[Ai, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs', reverse=True)

    #SS1 = [[Ai, 0], [0, Ci]]
    #SS2 = [[1, -B*Ci], [0, 1]]
    #return block_mat(SS1)*block_mat(SS2)

def fixed_stress():
    beta = 2*mu + Nd*lmbda
    SC   = collapse(C+Nd/float(beta))
    SCi  = AmesosSolver(SC)
    Ai   = AmesosSolver(A)
    SS   = [[Ai, B],
            [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)

x0 = AA.create_vec()
x0.randomize()

def run(prec, runs=[0]):
    from time import time
    try:
        t = time()
        precond = eval(prec)()
        AAinv = LGMRES(AA, precond=precond)
        xx = AAinv(initial_guess=x0, maxiter=15, tolerance=1e-10, show=2)*bb
        t = time()-t

        num_iter = AAinv.iterations
        residuals = AAinv.residuals

        AAinv = Richardson(AA, precond=precond, iter=1)
        xx = AAinv(initial_guess=x0, iter=1, show=0)*bb
        res = AAinv.residuals[1]/AAinv.residuals[0]

        pyplot.semilogy(residuals, marker='o', color='bgrkcmy'[runs[0]],
                        label='%-21s (#=%2d, e=%.2e, t=%.1f)'%(prec, num_iter, res, t))

    except Exception, e:
        print prec, e
    runs[0] += 1

run('drained_split')
run('undrained_split')
run('fixed_stress')
run('fixed_strain')
run('exact_schur')
run('inexact_schur')
run('exact_A_inexact_schur')

pyplot.legend(prop={'family':'monospace', 'size':'x-small'})
pyplot.show()

print "Finished normally"
