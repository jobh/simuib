from dolfin import *

mesh = UnitSquare(32,32)
dim = mesh.topology().dim()

P0 = FunctionSpace(mesh, "DG", 0)
P1 = FunctionSpace(mesh, "CG", 1)
RT0 = FunctionSpace(mesh, "RT", 1) if dim>1 else P1
V = RT0
Q = P0
S = P0

W = V*Q

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

g = Expression("sin(2*3.14*x[0])")

a = inner(u,v)*dx + div(u)*q*dx + div(v)*p*dx
L = g*q*dx

solution = Function(W)

solve(a ==L, solution)

u, p = split(solution)
plot(u)
plot(project(p,P1))

S0 = Constant(0.5)
dt = Constant(0.01)

s = TrialFunction(S)
r = TestFunction(S)

a1 = s*r*dx
L1 = S0*r*dx - dt*(div(u*S0))*r*dx

solution1 = Function(S)
solve(a1 == L1, solution1)

plot(project(solution1,P1))

interactive()
