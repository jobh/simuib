from __future__ import division

from dolfin import *
from block import *
from block.algebraic.trilinos import *
from block.iterative import *
from matplotlib import pyplot

set_log_level(PROGRESS if MPI.process_number()==0 else ERROR)

from spinal_cord_2d import *
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

rigid_body_modes(V)

def exact_schur():
    Ai = AmesosSolver(A)
    Sp = AmesosSolver(collapse(C-BT*InvDiag(A)*B))
    Si = LGMRES(C-BT*Ai*B, precond=Sp, tolerance=1e-14, maxiter=10)
    SS = [[Ai, B],
          [BT, Si]]
    return block_mat(SS).scheme('sgs')

def exact_A_approx_schur():
    Ai = AmesosSolver(A)
    Sp = AmesosSolver(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp]]
    return block_mat(SS).scheme('sgs')

def inexact_schur():
    Ai = ML(A, pdes=Nd, nullspace=rigid_body_modes(V))
    Sp = DD_ILUT(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp]]
    return block_mat(SS).scheme('tgs')

def exact_A_ml_schur():
    #Ai = ML(A, nullspace=rigid_body_modes(V))
    Ai = AmesosSolver(A)
    Sp = DD_ILUT(collapse(C-BT*InvDiag(A)*B))
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
    b_ = assemble(-b/alpha*q*phi*dx)
    b_i = AmesosSolver(b_)
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
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd*beta_inv)
    SCi  = AmesosSolver(SC)
    Ai   = AmesosSolver(A)
    SS = [[Ai, B],
          [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)

def optimized_fixed_stress():
    # Stable; Mikelic & Wheeler
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd/2*beta_inv)
    SCi  = AmesosSolver(SC)
    Ai   = AmesosSolver(A)
    SS = [[Ai, B],
          [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)

x0 = AA.create_vec()
x0.randomize()

def run(prec, runs=[0]):
    try:
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

run(drained_split)
run(undrained_split)
run(fixed_stress)
run(optimized_fixed_stress)
#run(fixed_strain)
run(exact_schur)
run(inexact_schur)
run(exact_A_approx_schur)
run(exact_A_ml_schur)

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
