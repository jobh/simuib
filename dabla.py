

from dolfin import *

mesh = UnitSquare(32,32)

V = FunctionSpace(mesh, "RT", 1)
Q = FunctionSpace(mesh, "DG", 0)
S = FunctionSpace(mesh, "DG", 0)

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
plot(p)

S0 = Constant(0.5) 
dt = Constant(0.01)

s = TrialFunction(S)
r = TestFunction(S)

a1 = s*r*dx 
L1 = S0*r*dx - dt*(div(u*S0))*r*dx

solution1 = Function(S)
solve(a1 == L1, solution1)

plot(solution1)


interactive()




 

