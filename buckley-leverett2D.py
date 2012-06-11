from __future__ import division
from dolfin import *

set_log_level(WARNING)
do_plot = 'mlL'
#parameters["form_compiler"]["optimize"] = True
#parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["representation"] = 'quadrature'
#parameters["form_compiler"]["quadrature_degree"] = 1

##
# Mesh
##

mesh = UnitCircle(32)
if 'm' in do_plot:
    plot(mesh)
dim = mesh.topology().dim()
hmin = MPI.min(mesh.hmin())
hmax = MPI.max(mesh.hmax())
h = CellSize(mesh)
n = FacetNormal(mesh)
x = mesh.ufl_cell().x

class plotline(object):
    mesh = UnitInterval(int(1.0/hmin))
    V = FunctionSpace(mesh, "CG", 1)
    F = {}

    def __call__(self, expr, title):
        if not expr in self.F:
            self.F[expr] = Function(self.V)
        v = self.F[expr].vector()
        for i,x in enumerate(self.mesh.coordinates()):
            v[i] = expr([x,0.0])
        plot(self.F[expr], title=title)
plotline = plotline()

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

P0 = FunctionSpace(mesh, "DG", 0)
P1 = FunctionSpace(mesh, "CG", 1)
RT0 = FunctionSpace(mesh, "RT", 1) if dim>1 else P1

V = RT0
Q = P0
S = P0
W = V*Q

u, p = TrialFunctions(W)
v, w = TestFunctions(W)
s = TrialFunction(S)
r = TestFunction(S)

up_soln = Function(W) # contains u_soln and p_soln
u_plot = Function(V)
p_plot = Function(Q)
s_soln = Function(S)
s_anal = Function(S)

##
# Initial and boundary conditions
##

s_soln.vector()[:] = 0.0

def _bc_u_dom(x, on_boundary):
    return on_boundary if dim>1 else near(x[0],0.0)
_bc_u_val = Expression(["x[0]*scale", "x[1]*scale"], scale=1.0)
_bc_u_val.scale = 4/assemble(dot(_bc_u_val, n)*ds, mesh=mesh)
bc_u = DirichletBC(W.sub(0), _bc_u_val, _bc_u_dom)

##
# Parameters and sources
##

def delta(V, pt):
    """Unit area delta function in discrete space V"""
    q = Function(V)
    PointSource(V, pt).apply(q.vector())
    a = q.vector()
    a[:] = a.array()/assemble(q*dx)
    return q

dt = Constant(hmin/dim/(dim+1)/(hmax/hmin)/3)
T = 0.15

q_u = 4*delta(P1, Point(0.0,0.0))
q_s = 4*delta(P1, Point(0.0,0.0))

##
# Time loop
##

t = 0
while t < T-float(dt)/2:
    t += float(dt)

    ##
    # Solve and plot conservation equations
    ##

    eq1 = inner(kinv(s_soln)*u,v)*dx + p*div(v)*dx
    eq2 = div(u)*w*dx - q_u*w*dx
    solve(lhs(eq1+eq2)==rhs(eq1+eq2), up_soln, bcs=bc_u)

    eq=(eq1==eq2)

    u_soln, p_soln = up_soln.split()
    if 'u' in do_plot:
        # plotting doesn't work correctly for u_soln, p_soln -- workaround:
        u_plot.assign(u_soln); plot(u_plot, title="u [t=%.2f]"%t)
    if 'p' in do_plot:
        p_plot.assign(p_soln); plot(p_plot, title="p [t=%.2f]"%t)

    ##
    # Solve and plot transport equation
    ##

    eq3 = (s-s_soln)/dt*r*dx - dot(f(s_soln)*u_soln, grad(r))*dx + dot(f(s_soln)*u_soln, n)*r*ds - q_s*r*dx
    eq3 += f_upwind_flux(s_soln, u_soln)*jump(r)*dS
    solve(lhs(eq3)==rhs(eq3), s_soln)

    if 's' in do_plot:
        plot(s_soln, title="s [t=%.2f]"%t)

    ##
    # Calculate and plot analytical solution, print error
    ##

    s_anal.assign(project((1.0/(lmbda-1)*sqrt(lmbda*4*Constant(t)/pi/dot(x,x))-1), P1))
    vec = s_anal.vector()
    vec[vec>1.0] = 1.0
    vec[vec<0.0] = 0.0

    s_anal.t = t
    if 'l' in do_plot:
        plotline(s_soln, title="s [t=%.2f]"%t)
    if 'L' in do_plot:
        plotline(s_anal, title="s analytical [t=%.2f]"%t)

    if 'S' in do_plot:
        plot(s_anal, mesh=mesh, title="s analytical [t=%.2f]"%t)
    err = assemble(abs(s_soln-s_anal)*dx)
    print "t=%.3f |e|=%.3g"%(t, err)

if do_plot:
    interactive()
