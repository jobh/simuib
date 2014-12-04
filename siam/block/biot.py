from __future__ import division

from dolfin import *
from block import *
from block.algebraic.trilinos import *
from block.iterative import *
from block.dolfin_util import rigid_body_modes
from block.block_util import isequal
from matplotlib import pyplot, rcParams
import numpy, random

rcParams.update({'font.size': 14})

set_log_level(PROGRESS if MPI.rank(None)==0 else ERROR)

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
    delta=1e-8
    execfile('simpleproblem.py')
elif problem == 14:
    delta=1e-4
    execfile('simpleproblem.py')
elif problem == 12:
    delta=1e-12
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

if not issymmetric(AA):
    raise RuntimeError("not symmetric")

plot_error = int(cl_args.get("plot_error", 0))
test = plot_error or int(cl_args.get("test", 1))
justsave = int(cl_args.get("justsave", 0))
inexact = int(cl_args.get("inexact", 0))
ml_cycles = int(cl_args.get("ml_cycles", 1))

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
exit()

#solvers = [BiCGStab, LGMRES, Richardson]
#solvers = [BiCGStab, Richardson]
#solvers = [BiCGStab]
#solvers = [Richardson]
solvers = [LGMRES]

# Assemble the matrices and vectors

[[A,  B],
 [BT, C]] = AA
del BT # use B.T instead, save the memory...

def pressure_schur():
    Sp = MumpsSolver(collapse(C-B.T*InvDiag(A)*B))
    Si = BiCGStab(C-B.T*Ai*B, precond=Sp, tolerance=1e-14,
                  nonconvergence_is_fatal=True)
    SS = [[Ai, B],
          [B.T, Si]]
    return block_mat(SS).scheme('tgs', reverse=True)
pressure_schur.color = 'c'

def exact_A_approx_schur():
    Sp = MumpsSolver(collapse(C-B.T*InvDiag(A)*B))
    SS = [[Ai, B],
          [B.T, Sp]]
    return block_mat(SS).scheme('sgs')

def inexact_pressure_schur():
    Sp = ML(collapse(C-B.T*InvDiag(A)*B), cycles=ml_cycles)
    SS = [[Aml, B],
          [B.T,  Sp ]]
    return block_mat(SS).scheme('tgs', reverse=True)
inexact_pressure_schur.color = 'c'

def inexact_symm_schur():
    Sp = ML(collapse(C-B.T*InvDiag(A)*B), cycles=ml_cycles)
    SS = [[Aml, 0],
          [0,  Sp]]
    return block_mat(SS).scheme('sgs')
inexact_symm_schur.color='k'

def inexact_gs():
    Cp = ML(C, cycles=ml_cycles)
    SS = [[Aml, B],
          [B.T, Cp]]
    return block_mat(SS).scheme('tgs')

def inexact_jacobi():
    Cp = ML(C, cycles=ml_cycles)
    #Sp = ML(collapse(C-B.T*InvDiag(A)*B))
    SS = [[Aml, B],
          [B.T, Cp]]
    return block_mat(SS).scheme('jac')
inexact_jacobi.color='k'

def jacobi():
    SS = [[Ai, B],
          [B.T, Ci]]
    return block_mat(SS).scheme('jac')
jacobi.color='k'

def exact_A_ml_schur():
    Sp = ML(collapse(C-B.T*InvDiag(A)*B), cycles=ml_cycles)
    SS = [[Ai, B],
          [B.T, Sp]]
    return block_mat(SS).scheme('sgs')

def inexact_drained_split():
    SS = [[Aml, B],
          [B.T, ML(C, cycles=ml_cycles)]]
    return block_mat(SS).scheme('tgs')
inexact_drained_split.color = 'y'

def drained_split():
    SS = [[Ai, B],
          [B.T, Ci]]
    return block_mat(SS).scheme('tgs')
drained_split.color = 'y'

def undrained_split():
    # Stable (note sign change)
    try:
        if float(b) == 0.0:
            return
    except:
        pass
    b_ = assemble(-b/alpha*q*phi*dx)
    b_i = MumpsSolver(b_)
    SAi = ConjGrad(A-B*b_i*B.T, precond=Ai, show=1, tolerance=1e-15,
                   nonconvergence_is_fatal=True)
    SS = [[SAi, B],
          [B.T, Ci]]
    return block_mat(SS).scheme('tgs')
undrained_split.color = 'b'


SAi_ = None

def inexact_undrained_split():
    # Stable (note sign change)
    try:
        if float(b) == 0.0:
            return
    except:
        pass
    b_ = assemble(-b/alpha*q*phi*dx)
    b_i = InvDiag(b_)
    global SAi_
    if SAi_ is None:
        SAi_ = ML(collapse(A-B*b_i*B.T), cycles=ml_cycles)
    SAi = SAi_

    SS = [[SAi, B],
          [B.T, ML(C, cycles=ml_cycles)]]
    return block_mat(SS).scheme('tgs')
inexact_undrained_split.color = 'b'

def fixed_strain():
    SS = [[Ai, B],
          [B.T, Ci]]
    return block_mat(SS).scheme('tgs', reverse=True)
fixed_strain.color = 'g'

def inexact_fixed_strain():
    SS = [[Aml, B],
          [B.T, ML(C, cycles=ml_cycles)]]
    return block_mat(SS).scheme('tgs', reverse=True)
inexact_fixed_strain.color = 'g'

def fixed_stress():
    # Stable (note sign change)
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd*beta_inv)
    SCi  = MumpsSolver(SC)
    SS = [[Ai, B],
          [B.T, SCi]]
    return block_mat(SS).scheme('tgs', reverse=True)
fixed_stress.color='r'

def inexact_fixed_stress():
    # Stable (note sign change)
    global SC,SCp
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd*beta_inv)
    SCp  = ML(SC, cycles=ml_cycles)
    SS = [[Aml, B],
          [B.T, SCp]]
    return block_mat(SS).scheme('tgs', reverse=True)
inexact_fixed_stress.color='r'

def inexact_optimized_fixed_stress():
    # Stable (note sign change)
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd/2*beta_inv)
    SCp  = ML(SC, cycles=ml_cycles)
    SS = [[Aml, B],
          [B.T, SCp]]
    return block_mat(SS).scheme('tgs', reverse=True)

def optimized_fixed_stress():
    # Stable; Mikelic & Wheeler
    beta_inv = assemble(-alpha/beta*q*phi*dx)
    SC   = collapse(C+Nd/2*beta_inv)
    SCi  = MumpsSolver(SC)
    SS = [[Ai, B],
          [B.T, SCi]]
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

    Ainv = ML(Ahom, pdes=Nd, nullspace=rbm, cycles=ml_cycles)
    Cinv = ML(Chom, cycles=ml_cycles)
    SS = block_mat([[Ainv, Bhom],
                    [Bhom.T, Cinv]])
    return SS.scheme('tgs')
inexact_homogeneous.color = 'y'

def homogeneous_pressure_schur():
    [[Ahom, Bhom],
     [_, Chom]]  = create_homogeneous()

    Ai = MumpsSolver(Ahom)
    Sp = MumpsSolver(collapse(Chom-Bhom.T*InvDiag(Ahom)*Bhom))
    Si = BiCGStab(Chom-Bhom.T*Ai*Bhom, precond=Sp, tolerance=1e-14,
                  nonconvergence_is_fatal=True)
    SS = [[Ai, Bhom],
          [Bhom.T, Si]]
    return block_mat(SS).scheme('tgs', reverse=True)
homogeneous_pressure_schur.color = 'g'

def inexact_homogeneous_pressure_schur():
    [[Ahom, Bhom],
     [_, Chom]]  = create_homogeneous()

    Sinv = MumpsSolver(collapse(Chom-Bhom.T*InvDiag(Ahom)*Bhom))
    Ainv = ML(Ahom, pdes=Nd, nullspace=rbm, cycles=ml_cycles)
    SS = block_mat([[Ainv, Bhom],
                    [Bhom.T, Sinv]])
    return SS.scheme('tgs')
inexact_homogeneous_pressure_schur.color = 'g'

#==================

x0 = AA.create_vec()
#numpy.random.seed()
#x0.randomize()
#x0[0][:] = 1
#x0[1][:] = 1

x0s = []

def norm(x,M):
    return sqrt(abs(x.inner(M*x)))

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
                x0.randomize()
                x0[0] *= 1/norm(x0[0], A)
                x0[1] *= 1/norm(x0[1], C)
                err0U = norm(x0[0], A)
                err0P = norm(x0[1], C)

            # Solve
            res0 = (AA*x0-bb).norm()

            #residuals = [(AA*x0-bb).norm()/res0]
            residuals = [1.0]
            errorsU = [1.0]
            errorsP = [1.0]
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
                if test:
                    errorsU.append(norm(x[0],A)/err0U)
                    errorsP.append(norm(x[1],C)/err0P)
                if plot_error:
                    uv[:] = numpy.abs(x[0])/err0U
                    pv[:] = numpy.abs(x[1])/err0P
                    plot(u, mode='color', title='%s %s'%(prec.__name__, solver.__name__))
                    plot(p, mode='color')
                    interactive()
            numiter = 15 if solver == LGMRES else 50
            if problem==5:
                numiter *= 3

            AAinv = solver(AA, precond=precond, iter=numiter, tolerance=1e-12)

            if False:
                try:
                    AAinv.compute_fixed_iterations(show=3)
                except Exception, e:
                    print e
                finally:
                    exit()

            xx = AAinv(initial_guess=x0.copy(), callback=cb, show=2)*bb

            # Plot

            num_iter = AAinv.iterations

            pyplot.figure(solver.__name__)
            marker = 'o+o'[runs[0]//2]
            markersize = [4,6,0][runs[0]//2]
            linestyle = '-' if 'jacobi' in prec.__name__ else '-'
            linewidth = 1.5 if 'jacobi' in prec.__name__ else 1.5
            pyplot.semilogy(residuals, color=prec.color,
                            #marker='o', markersize=4,
                            marker=marker, markersize=markersize,
                            ls=linestyle, lw=linewidth,
                            label='%-22s'%(prec.__name__), drawstyle='steps-post')
            if test:
                pyplot.semilogy(numpy.sqrt(numpy.array(errorsU)**2+numpy.array(errorsP)**2)/numpy.sqrt(2),
                                linestyle=':',
                                color=prec.color,
                                marker=marker,
                                markersize=markersize/2,
                                drawstyle='steps-post')
                #pyplot.semilogy(errorsP, linestyle=':', color=prec.color)

            del t

    except Exception, e:
        print prec, e
        raise
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
    t = 0
    while t < T:
        t += float(dt)
        xx = AAinv(tolerance=1e-10, show=2)*bb
        uv[:], pv[:] = xx

        #v = rigid_body_modes(V)
        #orthogonalize(uv, v)

        # Plot
        #plot(u)
        plot(u, key='1', title='u', mode='displacement')
        plot(p, key='2', title='p', mode='color')
        #plot(div(u), key='3', title='div(u)', mode='color')
        #plot(p-(2*nu/Nd+lmbda)*div(u), title='p-tr(sigma)/D', key='4', mode='color')
        #plot((2*mu/Nd+lmbda)*div(u), title='p-z div(u)', key='4', mode='color')
        #plot(p-tr(sigma(u))/Nd, title='p-tr(sigma)/D', key='4.1', mode='color')
        bb = block_assemble([L0,L1]);
        rhs_bc.apply(bb)

        from util import PlotLine
        #pl = PlotLine(mesh, lambda x: [2*x-1, 0])
        #pl(project(p-tr(sigma(u))/Nd, Q), title="p-tr(sigma)/D")
    interactive()

run=run1

if inexact:
    rbm = rigid_body_modes(V)
    Aml = ML(A, pdes=Nd, nullspace=rbm, cycles=ml_cycles)

    if problem == 4:
        #run2end(inexact_pressure_schur)
        pass

    #run(exact_C_approx_schur
    #run(inexact_symm_schur)
#    run(inexact_undrained_split)
    run(inexact_drained_split)
    run(inexact_fixed_stress)
    run(inexact_fixed_strain)
    #run(inexact_optimized_fixed_stress)
    #run(inexact_pressure_schur)
    #run(inexact_gs)
    run(inexact_jacobi)
    #run(inexact_homogeneous)
    #run(inexact_homogeneous_pressure_schur)

    def ev_est(M, prec, name):
        cg = ConjGrad(M, precond=prec, tolerance=1e-20)
        x = M.create_vec()
        block_vec([x]).randomize()
        cg*x
        ev = cg.eigenvalue_estimates()

        sign,name2 = ('-', name[1:]) if name[0]=='-' else ('', name)
        pyplot.figure(name)
        pyplot.grid()
        pyplot.title(r'Eigenvalues of $%s\hat{P}%s$'%(sign,name2))
        pyplot.xlabel('EV#')
        pyplot.ylabel(r'Value')
        pyplot.semilogy(ev, '-o', drawstyle='steps-post')
        pyplot.savefig('EV[%s],problem=%d,exact=%d,N=%d,cycles=%d.pdf' % (name, problem, not inexact, N, ml_cycles))
        with open('ev.log', 'a') as f:
            print >>f, 'EV[%s],problem=%d,N=%d\t%g,cycles=%d'%(name,problem,N,ev[-1]/ev[0], ml_cycles)
        print ev[-1]/ev[0]

    ev_est(A, Aml, "A")
    ev_est(-C, ML(collapse(-C), cycles=ml_cycles), "-C")
    ev_est(-SC, -SCp, "-S_c")

    del Aml

else:
    Ai = MumpsSolver(A)
    Ci = MumpsSolver(C)

    if problem == 4:
        #run2end(pressure_schur)
        pass
        
#    run(undrained_split)
    run(drained_split)
    run(fixed_stress)
    run(fixed_strain)
    #run(optimized_fixed_stress)
    #run(pressure_schur)
    #run(exact_A_approx_schur)
    #run(exact_A_ml_schur)
    run(jacobi)
    #run(homogeneous)
    #run(homogeneous_pressure_schur)

    del Ai
    del Ci

try:
    info = '\nd=%.0e b=%.0e K=%.1e tau=%.1e nu=%.4f'%(delta,b,Kdr,tau,nu)
    info=''
except:
    info = ''

try:
    for solver in solvers:
        f = pyplot.figure(solver.__name__)
        pyplot.ylim(1e-12,1e6)
        #x = f.axes[0].get_xaxis().get_data_interval()
        #pyplot.semilogy(x, [1.0, 1.0], 'k--')
        pyplot.grid()
        pyplot.xlabel('Iterations')
        pyplot.ylabel(r'Normalized $\ell^2$ residual')
        #pyplot.legend(loc='upper right', ncol=2,
        #              prop={'family':'monospace', 'size':'x-small'}).draggable()
        pyplot.legend(loc='upper right', ncol=1, framealpha=0.5).draggable()

        pyplot.title(solver.__name__ + info)

    if MPI.rank(None) == 0:
        for solver in solvers:
            f = pyplot.figure(solver.__name__)
            pyplot.tight_layout()
            pyplot.savefig('%s,problem=%d,exact=%d,N=%d,cycles=%d.pdf' % (solver.__name__, problem, not inexact, N, ml_cycles))
        if not justsave:
            pyplot.show()

    list_timings()
except e:
    raise

print "Finished normally"
