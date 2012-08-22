from __future__ import division
from util import *
from dolfin import *
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
W = V*Q

# Trial and test functions
u, p = TrialFunctions(W)
v, w = TestFunctions(W)
s = TrialFunction(S)
r = TestFunction(S)

# Functions (with degrees of freedom vectors)
up_soln = Function(W) # contains u_soln and p_soln
u_plot = Function(V)
p_plot = Function(Q)
s_soln = Function(S)
s_anal = Function(S)

# Derived functions
u_soln, p_soln = up_soln.split()
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
bc_u = DirichletBC(W.sub(0), _bc_u_val, _bc_u_dom)

##
# Parameters and sources
##

dt = Constant(hmin/dim/(dim+1)/(hmax/hmin))

delta = DeltaFunction(mesh)
if dim == 1:
    q_u = delta(Point(0.0)) - delta(Point(1.0))
    q_s = delta(Point(0.0)) - f(s_soln)*delta(Point(1.0))
else:
    q_u = delta(Point(0.0,0.0))
    q_s = delta(Point(0.0,0.0))

##
# Time loop
##

t = 0
while t < T-float(dt)/2:
    t += float(dt)      # float(...) to extract the value of a Constant

    ##
    # Solve and plot transport equation
    ##

    eq3 = (s-s_soln)/dt*r*dx - dot(f(s_soln)*u_soln, grad(r))*dx + dot(f(s_soln)*u_soln, n)*r*ds - q_s*r*dx
    eq3 += f_upwind_flux(s_soln, u_soln)*jump(r)*dS
    solve(lhs(eq3)==rhs(eq3), s_soln)

    if 's' in do_plot or (dim==1 and 'l' in do_plot):
        plot(s_soln, title="s [t=%.2f]"%t)
    if 'l' in do_plot and dim>1:
        plotline(s_soln, title="s [t=%.2f]"%t)

    ##
    # Solve and plot conservation equations (coupled)
    ##

    eq1 = inner(kinv(s_soln)*u,v)*dx + p*div(v)*dx
    eq2 = div(u)*w*dx - q_u*w*dx
    solve(lhs(eq1+eq2)==rhs(eq1+eq2), up_soln, bcs=bc_u)

    u_soln, p_soln = up_soln.split()
    if 'u' in do_plot:
        # plotting doesn't work correctly for u_soln, p_soln -- workaround:
        u_plot.assign(u_soln); plot(u_plot, title="u [t=%.2f]"%t)
    if 'p' in do_plot:
        p_plot.assign(p_soln); plot(p_plot, title="p [t=%.2f]"%t)

    ##
    # Calculate and plot analytical solution, print error
    ##

    if dim == 1:
        s_anal.assign(project((1.0/(lmbda-1)*sqrt(lmbda*Constant(t)/x)-1), P1))
    else:
        s_anal.assign(project((1.0/(lmbda-1)*sqrt(lmbda*Constant(t)/pi/dot(x,x))-1), P1))
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
