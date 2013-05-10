from __future__ import division

from dolfin import *
from block import *
from block.algebraic.trilinos import *
from block.iterative import *
from matplotlib import pyplot

set_log_level(PROGRESS if MPI.process_number()==0 else ERROR)

def dx_times(form):
    return form*dx
#from simpleproblem import *
#from spinal_cord_2d import *
from spinal_cord_3d import *

#===== Print some derived quantities ===
beta = 2*mu + Nd*lmbda
Kdr = beta/Nd
nu = lmbda/2/(lmbda+mu)
tau = alpha**2/Kdr/b if float(b)>0 else float('inf')
try:
    print 'Bulk modulus = %.2g, Poisson ratio = %.2g, coupling strength = %.2g' % (Kdr,nu,tau)
except:
    pass

#solvers = [BiCGStab, LGMRES, Richardson]
solvers = [BiCGStab, Richardson]

# Assemble the matrices and vectors

[[A,  B],
 [BT, C]] = AA

def exact_schur():
    Sp = MumpsSolver(collapse(C-BT*InvDiag(A)*B))
    Si = BiCGStab(C-BT*Ai*B, precond=Sp, tolerance=1e-14, maxiter=200)
    SS = [[Ai, B],
          [BT, Si]]
    return block_mat(SS).scheme('sgs')

def exact_A_approx_schur():
    Sp = MumpsSolver(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp]]
    return block_mat(SS).scheme('sgs')

def inexact_schur():
    Ai = ML(A, pdes=Nd, nullspace=rigid_body_modes(V))
    Sp = DD_ILUT(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp ]]
    return block_mat(SS).scheme('tgs')

def exact_A_ml_schur():
    #Ai = ML(A, nullspace=rigid_body_modes(V))
    Sp = DD_ILUT(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp]]
    return block_mat(SS).scheme('sgs')

def exact_C_approx_schur():
    Ci = MumpsSolver(C)
    Sp = MumpsSolver(collapse(A-B*InvDiag(C)*BT))
    SS = [[Sp, B],
          [BT, Ci]]
    return block_mat(SS).scheme('sgs', reverse=True)

def drained_split():
    Ci = MumpsSolver(C)
    SS = [[Ai, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs')

def undrained_split():
    # Stable (note sign change)
    b_ = assemble(dx_times(-b/alpha*q*phi))
    b_i = MumpsSolver(b_)
    SAp = MumpsSolver(collapse(A-B*InvDiag(b_)*BT))
    SAp._deps = [A,B,BT,b_]
    SAi = ConjGrad(A-B*b_i*BT, precond=SAp, show=1, tolerance=1e-14)
    Ci  = MumpsSolver(C)
    SS = [[SAi, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs')

def fixed_strain():
    Ci = MumpsSolver(C)
    SS = [[Ai, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs', reverse=True)

def fixed_stress():
    # Stable (note sign change)
    beta_inv = assemble(dx_times(-alpha/beta*q*phi))
    SC   = collapse(C+Nd*beta_inv)
    SCi  = MumpsSolver(SC)
    SS = [[Ai, B],
          [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)

def inexact_fixed_stress():
    # Stable (note sign change)
    beta_inv = assemble(dx_times(-alpha/beta*q*phi))
    SC   = collapse(C+Nd*beta_inv)
    SCi  = DD_ILUT(SC)
    Ai   = ML(A, pdes=Nd, nullspace=rigid_body_modes(V))
    SS = [[Ai, B],
          [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)

def optimized_fixed_stress():
    # Stable; Mikelic & Wheeler
    beta_inv = assemble(dx_times(-alpha/beta*q*phi))
    SC   = collapse(C+Nd/2*beta_inv)
    SCi  = MumpsSolver(SC)
    SS = [[Ai, B],
          [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)

x0 = AA.create_vec()
x0.randomize()

def run(prec, runs=[0]):
    try:
        print '===', prec.__name__, '==='
        precond = prec()

        for solver in solvers:

            # Solve

            AAinv = solver(AA, precond=precond)
            xx = AAinv(initial_guess=x0, maxiter=20, tolerance=1e-10, show=2)*bb

            # Plot

            num_iter = AAinv.iterations
            residuals = AAinv.residuals
            for j in reversed(range(len(residuals))):
                residuals[j] /= residuals[0]

            pyplot.figure(solver.__name__)
            pyplot.semilogy(residuals, marker='xo'[runs[0]//7], color='bgrkcmy'[runs[0]%7],
                            label='%-22s (#it=%2d)'%(prec.__name__, num_iter))

    except Exception, e:
        print prec, e
    runs[0] += 1

    try:
        del precond
        del AAinv
    except:
        pass
    import gc
    gc.collect()


# Do not use Ai
run(undrained_split)
#run(exact_C_approx_schur
run(inexact_schur)
run(inexact_fixed_stress)

Ai = MumpsSolver(A)

run(drained_split)
run(fixed_stress)
run(optimized_fixed_stress)
#run(fixed_strain)
run(exact_schur)
run(exact_A_approx_schur)
run(exact_A_ml_schur)

del Ai

try:
    info = 'd=%.0e b=%.0e K=%.1e tau=%.1e nu=%.4f'%(delta,b,Kdr,tau,nu)
except:
    info = ''

for solver in solvers:
    f = pyplot.figure(solver.__name__)
    x = f.axes[0].get_xaxis().get_data_interval()
    pyplot.semilogy(x, [1.0, 1.0], 'k--')
    pyplot.grid()
    pyplot.legend(loc='lower left', ncol=2,
                  prop={'family':'monospace', 'size':'x-small'}).draggable()
    pyplot.title('%s\n%s'%(solver.__name__, info))

if MPI.process_number() == 0:
    pyplot.show()
    pass

print "Finished normally"
