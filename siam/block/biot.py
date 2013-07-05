from __future__ import division

from dolfin import *
from block import *
from block.algebraic.trilinos import *
from block.iterative import *
from matplotlib import pyplot
import numpy

set_log_level(PROGRESS if MPI.process_number()==0 else ERROR)

def get_command_line_arguments():
    import sys
    dict = {}
    if len(sys.argv) == 1: return dict 
    for a in sys.argv[1:]: 
        key, value = a.split('=')
        dict[key] = value
    return dict

cl_args = get_command_line_arguments()
problem = int(cl_args.get('problem', 2))
if problem == 1:
    execfile('simpleproblem.py')
elif problem == 2:
    execfile('spinal_cord_2d.py')
elif problem == 3:
    execfile('spinal_cord_3d.py')
elif problem == 4:
    execfile('fault.py')
elif problem == 5:
    execfile('voring-small.py')
else:
    raise RuntimeError("unknown problem")

plot_error = int(cl_args.get("plot_error", 0))
test = plot_error or int(cl_args.get("test", 0))
justsave = int(cl_args.get("justsave", 0))
inexact = int(cl_args.get("inexact", 0))

#===== Print some derived quantities ===
beta = 2*mu + Nd*lmbda
Kdr = beta/Nd
nu = lmbda/2/(lmbda+mu)
try:
    print 'nu = %g'%float(nu)
    E = mu/nu* ((1.0 + nu)*(1.0 - 2.0*nu))
    print 'E =%g'%float(E)
    print 'Kdr =%g'%float(Kdr)
    exit()
except:
    pass
try:
    tau = alpha**2/Kdr/b if float(b)>0 else float('inf')
    print 'Bulk modulus = %.2g, Poisson ratio = %.2g, coupling strength = %.2g' % (Kdr,nu,tau)
except:
    pass

#solvers = [BiCGStab, LGMRES, Richardson]
solvers = [BiCGStab, Richardson]
#solvers = [BiCGStab]
#solvers = [Richardson]
#solvers = [LGMRES]

# Assemble the matrices and vectors

[[A,  B],
 [BT, C]] = AA

def pressure_schur():
    Sp = MumpsSolver(collapse(C-BT*InvDiag(A)*B))
    Si = BiCGStab(C-B.T*Ai*B, precond=Sp, tolerance=1e-14,
                  nonconvergence_is_fatal=True)
    SS = [[Ai, B],
          [BT, Si]]
    return block_mat(SS).scheme('tgs', reverse=True)
pressure_schur.color = 'c'

def exact_A_approx_schur():
    Sp = MumpsSolver(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp]]
    return block_mat(SS).scheme('sgs')

def inexact_pressure_schur():
    Sp = DD_ILUT(collapse(C-BT*InvDiag(A)*B))
    SS = [[Aml, B],
          [BT,  Sp ]]
    return block_mat(SS).scheme('tgs', reverse=True)
inexact_pressure_schur.color = 'c'

def inexact_symm_schur():
    Sp = ML(collapse(C-BT*InvDiag(A)*B))
    SS = [[Aml, 0],
          [0,  Sp]]
    return block_mat(SS).scheme('sgs')
inexact_symm_schur.color='k'

def inexact_gs():
    Cp = DD_ILUT(C)
    SS = [[Aml, B],
          [BT, Cp]]
    return block_mat(SS).scheme('tgs')

def inexact_jacobi():
    Cp = DD_ILUT(C)
    #Sp = DD_ILUT(collapse(C-BT*InvDiag(A)*B))
    SS = [[Aml, B],
          [BT, Cp]]
    return block_mat(SS).scheme('jac')
inexact_jacobi.color='y'

def jacobi():
    SS = [[Ai, B],
          [BT, Ci]]
    return block_mat(SS).scheme('jac')
jacobi.color='y'

def exact_A_ml_schur():
    Sp = DD_ILUT(collapse(C-BT*InvDiag(A)*B))
    SS = [[Ai, B],
          [BT, Sp]]
    return block_mat(SS).scheme('sgs')

def inexact_drained_split():
    SS = [[Aml, B],
          [BT, DD_ILUT(C)]]
    return block_mat(SS).scheme('tgs')
inexact_drained_split.color = 'g'

def drained_split():
    SS = [[Ai, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs')
drained_split.color = 'g'

def undrained_split():
    # Stable (note sign change)
    try:
        if float(b) == 0.0:
            return
    except:
        pass
    b_ = assemble(-b/alpha*q*phi*dx)
    b_i = MumpsSolver(b_)
    SAi = ConjGrad(A-B*b_i*BT, precond=Ai, show=1, tolerance=1e-14,
                   nonconvergence_is_fatal=True)
    SS = [[SAi, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs')
undrained_split.color = 'b'

def inexact_undrained_split():
    # Stable (note sign change)
    try:
        if float(b) == 0.0:
            return
    except:
        pass
    b_ = assemble(-b/alpha*q*phi*dx)
    b_i = InvDiag(b_)
    SAi = ML(collapse(A-B*b_i*BT))

    SS = [[SAi, B],
          [BT, DD_ILUT(C)]]
    return block_mat(SS).scheme('tgs')
inexact_undrained_split.color = 'b'

def fixed_strain():
    SS = [[Ai, B],
          [BT, Ci]]
    return block_mat(SS).scheme('tgs', reverse=True)
fixed_strain.color = 'k'

def inexact_fixed_strain():
    SS = [[Aml, B],
          [BT, DD_ILUT(C)]]
    return block_mat(SS).scheme('tgs', reverse=True)
inexact_fixed_strain.color = 'k'

def fixed_stress():
    # Stable (note sign change)
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd*beta_inv)
    SCi  = MumpsSolver(SC)
    SS = [[Ai, B],
          [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)
fixed_stress.color='r'

def inexact_fixed_stress():
    # Stable (note sign change)
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd*beta_inv)
    SCp  = DD_ILUT(SC)
    SS = [[Aml, B],
          [BT, SCp]]
    return block_mat(SS).scheme('tgs', reverse=True)
inexact_fixed_stress.color='r'

def inexact_optimized_fixed_stress():
    # Stable (note sign change)
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd/2*beta_inv)
    SCp  = DD_ILUT(SC)
    SS = [[Aml, B],
          [BT, SCp]]
    return block_mat(SS).scheme('tgs', reverse=True)

def optimized_fixed_stress():
    # Stable; Mikelic & Wheeler
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd/2*beta_inv)
    SCi  = MumpsSolver(SC)
    SS = [[Ai, B],
          [BT, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)

def create_homogeneous():
    # Note: Probably only works with problem=2 (and maybe 3)
    area = assemble(Constant(1)*dx, mesh=mesh)
    mu_avg = assemble(mu*dx, mesh=mesh)/area
    lmbda_avg = assemble(lmbda*dx, mesh=mesh)/area
    K_avg = assemble(K*dx, mesh=mesh)/area
    b_avg = assemble(b*dx, mesh=mesh)/area
    lmbdamuInv_avg = assemble(lmbdamuInv*dx, mesh=mesh)/area

    def sigma(v):
        return 2.0*mu_avg*sym(grad(v)) + lmbda_avg*tr(grad(v))*Identity(Nd)
    def v_D(q):
        return -K_avg*grad(q)
    def coupling(w,r):
        return - alpha * r * div(w)
    def corr(test, trial):
        return 1*lmbdamuInv_avg*h**2*inner(grad(test), grad(trial))

    a00 = inner(grad(omega), sigma(v)) * dx
    a01 = coupling(omega,q) * dx
    a10 = coupling(v,phi) * dx
    a11 = -(b_avg*phi*q - dt*inner(grad(phi),v_D(q))) * dx - corr(phi, q)*dx

    AAhom, _ = block_symmetric_assemble([[a00, a01], [a10, a11]], bcs=bcs)
    return AAhom

def homogeneous():
    [[Ahom, Bhom],
     [_, Chom]]  = create_homogeneous()

    Ainv = MumpsSolver(Ahom)
    Cinv = MumpsSolver(Chom)
    SS = block_mat([[Ainv, Bhom],
                    [Bhom.T, Cinv]])
    return SS.scheme('tgs')
homogeneous.color = 'y'

def inexact_homogeneous():
    [[Ahom, Bhom],
     [_, Chom]]  = create_homogeneous()

    Ainv = ML(Ahom, pdes=Nd, nullspace=rbm)
    Cinv = DD_ILUT(Chom)
    SS = block_mat([[Ainv, Bhom],
                    [Bhom.T, Cinv]])
    return SS.scheme('tgs')
inexact_homogeneous.color = 'y'

#==================

x0 = AA.create_vec()
x0.randomize()

print x0[0].size()+x0[1].size()
x0s = []

def run1(prec, runs=[0]):
    try:
        print '===', prec.__name__, '==='
        precond = prec()
        if precond is None:
            print '(skip)'
            return

        for solver in solvers:

            t = Timer('%s %s'%(solver.__name__, prec.__name__))

            if test:
                bb.zero()
                x0[0] *= 1/x0[0].norm('l2')
                x0[1] *= 1/x0[1].norm('l2')

            # Solve
            res0 = (AA*x0-bb).norm()
            err0U = x0[0].norm('l2')
            err0P = x0[1].norm('l2')
            #res0 = 1.0
            #err0U = 1.0
            #err0P = 1.0

            residuals = [(AA*x0-bb).norm()/res0]
            errorsU = [x0[0].norm('l2')/err0U]
            errorsP = [x0[1].norm('l2')/err0P]
            if plot_error:
                pv = p.vector()
                uv = u.vector()
                uv[:] = numpy.abs(x0[0])/err0U
                pv[:] = numpy.abs(x0[1])/err0P
                plot(u, mode='color', title='%s %s'%(prec.__name__, solver.__name__))
                plot(p, mode='color', title='%s %s'%(prec.__name__, solver.__name__))
                interactive(True)
            def cb(k, x, r):
                residuals.append((AA*x-bb).norm()/res0)
                errorsU.append(x[0].norm('l2')/err0U)
                errorsP.append(x[1].norm('l2')/err0P)
                if plot_error:
                    uv[:] = numpy.abs(x[0])/err0U
                    pv[:] = numpy.abs(x[1])/err0P
                    plot(u, mode='color', title='%s %s'%(prec.__name__, solver.__name__))
                    plot(p, mode='color')
                    interactive()
            numiter = 10 if solver == LGMRES else 50
            if problem==5:
                numiter *= 3

            AAinv = solver(AA, precond=precond, iter=numiter, tolerance=1e-10)

            if False:
                try:
                    AAinv.compute_fixed_iterations(show=3)
                except Exception, e:
                    print e
                finally:
                    exit()

            xx = AAinv(initial_guess=x0, callback=cb, show=2)*bb

            # Plot

            num_iter = AAinv.iterations

            pyplot.figure(solver.__name__)
            pyplot.semilogy(residuals, marker='xo'[runs[0]//7], color=prec.color,
                            label='%-22s'%(prec.__name__))
            if test:
                pyplot.semilogy(errorsU, linestyle='--', color=prec.color)
                pyplot.semilogy(errorsP, linestyle=':', color=prec.color)

            del t

    except Exception, e:
        print prec, e
        #raise
    runs[0] += 1

    try:
        del precond
        del AAinv
    except:
        pass
    import gc
    gc.collect()

def run3(prec):
    if not x0s:
        for i in range(10):
            x0s.append(AA.create_vec())
            x0s[-1].randomize()

    try:
        print '===', prec.__name__, '==='
        precond = prec()
        if precond is None:
            print '(skip)'
            return

        solver = BiCGStab
        num_iter = []
        for x in x0s:
            # Solve

            AAinv = solver(AA, precond=precond)
            xx = AAinv(initial_guess=x, maxiter=100, tolerance=-1e-8, show=2)*bb

            # Plot

            num_iter.append(AAinv.iterations)

        print 'X %s %.1f %d'%(prec.__name__,
                              numpy.mean(num_iter),
                              numpy.max(num_iter))

    except Exception, e:
        print prec, e

    try:
        del precond
        del AAinv
    except:
        pass
    import gc
    gc.collect()

def run2end(prec):
    global bb
    precond = prec()

    # Solve

    AAinv = BiCGStab(AA, precond=precond)
    pv = p.vector()
    uv = u.vector()
    for i in range(1):
        xx = AAinv(tolerance=1e-10, show=2)*bb
        uv[:], pv[:] = xx

        # Plot
        plot(u, key='1', mode='displacement')
        plot(p, key='2', mode='color')
        #plot(tr(sigma(u)), title='tr sigma', key='3', mode='color')

    interactive()

run=run1

if inexact:
    rbm = rigid_body_modes(V)
    Aml = ML(A, pdes=Nd, nullspace=rbm)

    #run(exact_C_approx_schur
    #run(inexact_symm_schur)
#    run(inexact_undrained_split)
    run(inexact_drained_split)
    run(inexact_fixed_stress)
    #run(inexact_fixed_strain)
    #run(inexact_optimized_fixed_stress)
    run(inexact_pressure_schur)
    #run(inexact_gs)
    #run(inexact_jacobi)
    run(inexact_homogeneous)

    del Aml

else:
    Ai = MumpsSolver(A)
    Ci = MumpsSolver(C)

#    run(undrained_split)
    run(drained_split)
    run(fixed_stress)
    #run(fixed_strain)
    #run(optimized_fixed_stress)
    run(pressure_schur)
    #run(exact_A_approx_schur)
    #run(exact_A_ml_schur)
    #run(jacobi)
    run(homogeneous)

    #if problem == 4:
    #    run2end(fixed_stress)

    del Ai
    del Ci

try:
    info = 'd=%.0e b=%.0e K=%.1e tau=%.1e nu=%.4f'%(delta,b,Kdr,tau,nu)
except:
    info = ''

try:
    for solver in solvers:
        f = pyplot.figure(solver.__name__)
        pyplot.ylim(1e-14,1e6)
        #x = f.axes[0].get_xaxis().get_data_interval()
        #pyplot.semilogy(x, [1.0, 1.0], 'k--')
        pyplot.grid()
        pyplot.xlabel('Iterations')
        pyplot.ylabel('Residual')
        pyplot.legend(loc='upper right', ncol=2,
                      prop={'family':'monospace', 'size':'x-small'}).draggable()
        pyplot.title('%s\n%s'%(solver.__name__, info))

    if MPI.process_number() == 0:
        for solver in solvers:
            f = pyplot.figure(solver.__name__)
            pyplot.savefig(solver.__name__+'.pdf')
        if not justsave:
            pyplot.show()
        pass

    list_timings()
except:
    pass

print "Finished normally"
