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

N=20
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

lmbda = Constant(1)
mu    = Constant(1)
delta = 1e0

class Permeability(Expression):
    def value_shape(self):
        return (2,2)
    def eval(self, tensor, x):
        tensor.shape = self.value_shape()
        tensor[:] = 0.0
        if 0.0 <= x[1] < 0.5:
            tensor[0,0] = 1.0
            tensor[1,1] = tensor[0,0]
        else:
            tensor[0,0] = delta
            tensor[1,1] = tensor[0,0]
b = Constant(1e-6)
alpha = Constant(1.0)
Lambda = Permeability()

t_n = Constant( [0.0]*Nd )

dt = Constant(0.02)
T = 0.1

r = Constant(0)

#===== Print some derived quantities ===
beta = 2*mu + Nd*lmbda
Kdr = beta/Nd
nu = lmbda/2/(lmbda+mu)
tau = alpha**2/Kdr/b
print 'Bulk modulus = %.2g, Poisson ratio = %.2g, coupling strength = %.2g' % (Kdr,nu,tau)

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

bc_u_bedrock        = DirichletBC(V, [0.0]*Nd, lambda x,bdry: bdry and x[1] <= 1/N/3)
bc_p_drained_edges  = DirichletBC(Q,  0.0,     lambda x,bdry: bdry and x[1] >= 1-1/N/3)

bcs = [bc_u_bedrock, bc_p_drained_edges]
#bcs = [bc_u_bedrock, None]

# Assemble the matrices and vectors

AA, AAns = block_symmetric_assemble([[a00,a10],[a01,a11]], bcs=bcs)
bb = block_assemble([L0,L1], bcs=bcs, symmetric_mod=AAns)

[[A,  B],
 [BT, C]] = AA

def exact_schur():
    Ai = AmesosSolver(A)
    Sp = ML(collapse(C-BT*InvDiag(A)*B))
    Si = LGMRES(C-BT*Ai*B, precond=Sp, tolerance=1e-14)
    SS = [[Ai, B],
          [BT, Si]]
    return block_mat(SS).scheme('sgs')

def exact_A_approx_schur():
    Ai = AmesosSolver(A)
    Sp = AmesosSolver(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp]]
    return block_mat(SS).scheme('sgs')

def exact_C_approx_schur():
    Ci = AmesosSolver(C)
    Sp = AmesosSolver(collapse(A-B*InvDiag(C)*BT))
    SS = [[Sp, B],
          [BT, Ci]]
    return block_mat(SS).scheme('sgs', reverse=True)

def drained_split():
    Ai = AmesosSolver(A)
    Ci = AmesosSolver(C)
    SS = [[Ai, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs')

def undrained_split():
    # Stable (note sign change)
    b_ = assemble(-b*q*phi*dx)
    b_i = AmesosSolver(b_)
    #SAp = AmesosSolver(A)
    SAp = AmesosSolver(collapse(A-B*InvDiag(b_)*BT))
    SAi = ConjGrad(A-B*b_i*BT, precond=SAp, tolerance=1e-14)
    Ci  = AmesosSolver(C)
    SS = [[SAi, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs')

def fixed_strain():
    Ai = AmesosSolver(A)
    Ci = AmesosSolver(C)
    SS = [[Ai, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs', reverse=True)

def fixed_stress():
    # Stable (note sign change)
    beta_inv = assemble(-1/beta*q*phi*dx)
    SC   = collapse(C+Nd*beta_inv)
    SCi  = AmesosSolver(SC)
    Ai   = AmesosSolver(A)
    SS = [[Ai, B],
          [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)

def optimized_fixed_stress():
    # Stable; Mikelic & Wheeler
    beta_inv = assemble(-1/beta*q*phi*dx)
    SC   = collapse(C+Nd/2*beta_inv)
    SCi  = AmesosSolver(SC)
    Ai   = AmesosSolver(A)
    SS = [[Ai, B],
          [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)

x0 = AA.create_vec()
x0.randomize()

def run(prec, runs=[0]):
    from time import time
    try:
        t = time()
        precond = eval(prec)()

        # GMRES

        AAinv = LGMRES(AA, precond=precond)
        xx = AAinv(initial_guess=x0, maxiter=10, tolerance=1e-10, show=2)*bb
        t = time()-t

        # Plot

        num_iter = AAinv.iterations
        residuals = AAinv.residuals
        for i in reversed(range(len(residuals))):
            residuals[i] /= residuals[0]

        pyplot.figure(1)
        pyplot.semilogy(residuals, marker='xo'[runs[0]//7], color='bgrkcmy'[runs[0]%7],
                        label='%-22s (#it=%2d, t=%.1f)'%(prec, num_iter, t))

        AAinv = Richardson(AA, precond=precond)
        xx = AAinv(initial_guess=x0, maxiter=10, tolerance=1e-10, show=2)*bb

        num_iter = AAinv.iterations
        residuals = AAinv.residuals
        for i in reversed(range(len(residuals))):
            residuals[i] /= residuals[0]

        pyplot.figure(2)
        pyplot.semilogy(residuals, marker='xo'[runs[0]//7], color='bgrkcmy'[runs[0]%7],
                        label='%-22s (#it=%2d, 1sr=%.2e)'%(prec, num_iter, residuals[1]))

    except Exception, e:
        raise
        print prec, e
    runs[0] += 1

run('drained_split')
run('undrained_split')
run('fixed_stress')
run('optimized_fixed_stress')
run('fixed_strain')
run('exact_schur')
#run('inexact_schur')
run('exact_A_approx_schur')
run('exact_C_approx_schur')

info = 'd=%.0e b=%.0e K=%.1e\ntau=%.1e nu=%.4f'%(delta,b,Kdr,tau,nu)

pyplot.figure(1)
pyplot.legend(prop={'family':'monospace', 'size':'x-small'})
pyplot.title('LGMRES; '+info)

pyplot.figure(2)
pyplot.legend(prop={'family':'monospace', 'size':'x-small'})
pyplot.title('Richardson; '+info)

pyplot.show()

print "Finished normally"
