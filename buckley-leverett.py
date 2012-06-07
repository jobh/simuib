from __future__ import division
from dolfin import *

set_log_level(WARNING)
do_plot = True
#parameters["form_compiler"]["optimize"] = True
#parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["representation"] = 'quadrature'
#parameters["form_compiler"]["quadrature_degree"] = 1

##
# Mesh
##

N = 64
mesh = UnitInterval(N)
dim = mesh.topology().dim()
hmin = MPI.min(mesh.hmin())
h = CellSize(mesh)
n = FacetNormal(mesh)

##
# Constitutive relations
##

lmbda = 2

def k(s):
    return 1 + (lmbda-1)*s

def kinv(s):
    return 1/k(s)

def f(s):
    return lmbda*s*kinv(s)

def f_upwind_flux(s,u):
    un = (dot(u, n) + abs(dot(u, n))) / 2  # max(dot(u,n), 0)

    #Ren upwind:
    #   f_h(s) = f(s_+) max(u.n,0) + f(s_-) min(u.n,0)
    return (f(s('+'))*un('+') - f(s('-'))*un('-'))

    #Vektet upwind (la f(s) = g'(s)s), s* kontinuerlig :
    #   f_h(s) = g'(s*) ( s_+ max(u.n,0) + s_- min(u.n,0) )
    #return lmbda*kinv(avg(s)) * (s('+')*un('+') - s('-')*un('-'))

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

##
# Initial and boundary conditions
##

s_soln.vector()[:] = 0.0

def u_boundary(x, on_boundary):
    return on_boundary if dim>1 else near(x[0],0.0)
u_bval = Constant([0]*dim) if dim>1 else Constant(0)
bc_u = DirichletBC(W.sub(0), u_bval, u_boundary)

##
# Parameters and sources
##

dt = Constant(hmin/2.1)
T = 0.3

q_u = Expression("(near(x[0],0.0) ? 1.0 : near(x[0],1.0) ? -1.0 : 0.0)") * 4/h
q_s = Expression("(near(x[0],0.0) ? 1.0 : 0.0)") * 4/h

##
# Analytical solution
##

s_anal = Expression("max(0.0, min(1.0, 1.0/(lmbda-1)*(sqrt(lmbda*t/x[0])-1)))",
                    lmbda=lmbda, t=0)

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

    u_soln, p_soln = up_soln.split()
    if do_plot:
        # plotting doesn't work correctly for u_soln, p_soln -- workaround:
        u_plot.assign(u_soln); plot(u_plot, title="u [t=%.2f]"%t)
        p_plot.assign(p_soln); plot(p_plot, title="p [t=%.2f]"%t)

    ##
    # Solve and plot transport equation
    ##

    eq3 = (s-s_soln)/dt*r*dx - dot(f(s_soln)*u_soln, grad(r))*dx + dot(f(s_soln)*u_soln, n)*r*ds - q_s*r*dx
    eq3 += f_upwind_flux(s_soln, u_soln)*jump(r)*dS
    solve(lhs(eq3)==rhs(eq3), s_soln)

    if do_plot:
        plot(s_soln, title="s [t=%.2f]"%t)

    ##
    # Plot analytical solution, print error
    ##

    s_anal.t = t
    if do_plot:
        plot(s_anal, mesh=mesh, title="s analytical [t=%.2f]"%t)
    err = assemble(abs(s_soln-s_anal)*dx)
    print "t=%.3f |e|=%.3g"%(t, err)

if do_plot:
    interactive()
