from __future__ import division
"""A solver using blockwise preconditioning via cbc.block (bzr branch lp:cbc.block/1.0)"""

import scipy
from util import *
from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.trilinos import *
from common import *

##
# Mesh
##

# Attributes of the global mesh
dim = mesh.topology().dim()
hmin = MPI.min(mesh.hmin())
hmax = MPI.max(mesh.hmax())

# Attributes of the cell/facets/points
h = CellSize(mesh)
n = FacetNormal(mesh)
x = mesh.ufl_cell().x

if dim == 2:
    # Smooth the circle mesh
    for i in range(10):
        mesh.smooth()
        if 'm' in do_plot:
            plot(mesh)

    # Create a line-plotter along y=0
    plotline = PlotLine(mesh, lambda r:[r,0])
else:
    # Line coincides with domain; plot only one
    do_plot.replace('s', 'l')
    do_plot.replace('S', 'L')
    plotline = PlotLine(mesh, lambda r:r)

##
# Constitutive relations
##

lmbda = 2

def k(s):
    return 1 + (lmbda-1)*s

def kinv(s):
    return 1/k(s)

def g_(s): # f(s) = g_(s)s
    return lmbda*kinv(s)

def f(s):
    return g_(s)*s

def f_upwind_flux(s,u):
    un = (dot(u, n) + abs(dot(u, n))) / 2  # max(dot(u,n), 0)

    #Ren upwind:
    #   f_h(s) = f(s_+) max(u.n,0) + f(s_-) min(u.n,0)
    return (f(s('+'))*un('+') - f(s('-'))*un('-'))

    #Vektet upwind (la f(s) = g'(s)s), s* kontinuerlig :
    #   f_h(s) = g'(s*) ( s_+ max(u.n,0) + s_- min(u.n,0) )
    #return g_(avg(s)) * (s('+')*un('+') - s('-')*un('-'))

##
# Functions and spaces
##

# General definitions (aliases)
P0 = FunctionSpace(mesh, "DG", 0)
P1 = FunctionSpace(mesh, "CG", 1)
RT0 = FunctionSpace(mesh, "RT", 1) if dim>1 else P1

# Spaces in use
V = RT0
Q = P0
S = P0

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
w = TestFunction(Q)
s = TrialFunction(S)
r = TestFunction(S)

# Functions (with degrees of freedom vectors)
u_soln = Function(V)
p_soln = Function(Q)
s_soln = Function(S)
s_anal = Function(S)

# Derived functions
s_diff = s_soln - s_anal

##
# Initial and boundary conditions
##

s_soln.vector()[:] = 0.0

if dim == 1:
    def _bc_u_dom(x, on_boundary): return on_boundary and x[0]<0.5
    _bc_u_val = Constant(0)
else:
    # Scale outflow to 1. The pressure solution is sensitive to this value, and
    # the boundary is not an exact circle.
    def _bc_u_dom(x, on_boundary): return on_boundary
    _bc_u_val = Expression(["x[0]*scale", "x[1]*scale"], scale=1)
    _bc_u_val.scale = 1/assemble(dot(_bc_u_val, n)*ds, mesh=mesh)
bc_u = DirichletBC(V, _bc_u_val, _bc_u_dom)

##
# Parameters and sources
##

# Strength of the pressure sources
source_strength = 1;

delta = DeltaFunction(mesh)
if dim == 1:
    q_u = source_strength * delta(Point(0.0)) - source_strength * delta(Point(1.0))
    q_s = delta(Point(0.0)) - f(s_soln)*delta(Point(1.0))
else:
    q_u = delta(Point(0.0,0.0))
    q_s = delta(Point(0.0,0.0))

# Maximal admissible time step
# The maxmimal cell volume should be bounded above by hmax^dim / dim 
# at least for simplices in 1D and 2D
dt = Constant(hmax**dim / (1.1 * dim  * source_strength * lmbda))

t = 0
xx = None
while t < T-float(dt)/2:
    t += float(dt)      # float(...) to extract the value of a Constant

    ##
    # Solve and plot transport equation
    ##

    eq3 = (s-s_soln)/dt*r*dx - dot(f(s_soln)*u_soln, grad(r))*dx + dot(f(s_soln)*u_soln, n)*r*ds - q_s*r*dx
    eq3 += f_upwind_flux(s_soln, u_soln)*jump(r)*dS
    A = assemble(lhs(eq3))
    b = assemble(rhs(eq3))
    Ainv = LGMRES(A, precond=ML(A), initial_guess=s_soln.vector())
    s_soln.vector()[:] = Ainv*b

    if 's' in do_plot or (dim==1 and 'l' in do_plot):
        plot(s_soln, title="s [t=%.2f]"%t)
    if 'l' in do_plot and dim>1:
        plotline(s_soln, title="s [t=%.2f]"%t)

    ##
    # Solve and plot conservation equations (coupled)
    ##

    eq1_u = inner(kinv(s_soln)*u,v)*dx
    eq1_p = p*div(v)*dx
    eq2_u = div(u)*w*dx + q_u*w*dx
    A = assemble(eq1_u)
    B = assemble(eq1_p)
    C = assemble(lhs(eq2_u))
    c = assemble(rhs(eq2_u))

    #   [A B] * [u] = [b]
    #   [C 0]   [p]   [c]

    AA = block_mat([[A, B],
                    [C, 0]])
    bb = block_vec([0, c])

    bc = block_bc([bc_u, None])
    bc.apply(AA, bb, symmetric=False)

    Schur = collapse(-C*InvDiag(A)*B)
    AAprec = block_mat([[A,   B    ],
                        [C,   Schur]])
    AAinv = BiCGStab(AA, initial_guess=xx,
                     precond=AAprec.scheme('symmetric gauss-seidel', inverse=ML))

    xx = AAinv*bb

    u_soln.vector()[:] = xx[0]
    p_soln.vector()[:] = xx[1]

    if 'u' in do_plot:
        plot(u_soln, title="u [t=%.2f]"%t)
    if 'p' in do_plot:
        plot(p_soln, title="p [t=%.2f]"%t)

    ##
    # Calculate and plot analytical solution, print error
    ##

    if dim == 1:
        s_anal.assign(project((1.0/(lmbda-1)*(sqrt(lmbda*Constant(t)/x)-1)), P1))
    else:
        s_anal.assign(project((1.0/(lmbda-1)*(sqrt(lmbda*Constant(t)/pi/dot(x,x))-1)), P1))
    vec = s_anal.vector()
    vec[vec>1.0] = 1.0
    vec[vec<0.0] = 0.0

    if 'S' in do_plot:
        plot(s_anal, title="s analytical [t=%.2f]"%t)
    if 'L' in do_plot:
        plotline(s_anal, title="s analytical [t=%.2f]"%t)
    if 'm' in do_plot:
        plot(mesh)
    if 'd' in do_plot:
        plotline(s_diff, title="s_soln-s_anal [t=%.2f]"%t)
    err = assemble(abs(s_soln-s_anal)*dx)
    print "t=%.3f |e|=%.3g"%(t, err)

if do_plot:
    # wait for user interaction (press 'q' to exit)
    interactive()
