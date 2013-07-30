
print " not done yet with this "  

from dolfin import *

from block import *
from block.iterative import *
from block.algebraic.trilinos import *
from block.dolfin_util import *

import pytave

def boundary(x, on_boundary): 
  return on_boundary

Ns = [4, 8] 
def epsilon(u): 
  return 0.5*(grad(u) + grad(u).T) 

for N in Ns:  
  mesh = UnitSquare(N, N)
#  V = VectorFunctionSpace(mesh, "Lagrange", 2)
#  Q = FunctionSpace(mesh, "Lagrange", 1)

  V = FunctionSpace(mesh, "RT", 1)
  Q = FunctionSpace(mesh, "DG", 0)
#  bc = [DirichletBC(V, Constant((0,0)), boundary), None]
  bc = [None, None]
  u, v = TrialFunction(V), TestFunction(V)
  p, q = TrialFunction(Q), TestFunction(Q)


  p00 = inner(u,v)*dx   
  a00 = div(u)*div(v)*dx + inner(u,v)*dx   
  a01 = div(v)*p*dx  
  a10 = div(u)*q*dx  
  a11 = p*q*dx  
  P00 = assemble(p00)
  A00 = assemble(a00)
  A10 = assemble(a10)
  A01 = assemble(a01)
  A11 = assemble(a11)

  
  e = pytave.feval(1, "check_eigs_vs", A11.array(), A01.array(), A00.array(), P00.array())
  e = pytave.feval(1, "abs", e[0])
  e = pytave.feval(1, "sort", e[0])
  e = e[0]

  print " condition number 1 - div RT ", e[-1]/e[0], e[-1]/e[1]   

  p00 = inner(u,v)*dx   
  a00 = inner(u,v)*dx   
  a01 = div(v)*p*dx  
  a10 = div(u)*q*dx  
#  a11 = p*q*dx + inner(grad(p),grad(q))*dx  
  alpha = 1.0 
  n = FacetNormal(mesh)
  h = CellSize(mesh)
  h_avg = (h('+') + h('-'))/2

  a11 = p*q*dx +  alpha/h_avg*dot(jump(p, n), jump(q, n))*dS() + (alpha/h)*p*q*ds  
  P00 = assemble(p00)
  A00 = assemble(a00)
  A10 = assemble(a10)
  A01 = assemble(a01)
  A11 = assemble(a11)

  
  e = pytave.feval(1, "check_eigs_vs", A11.array(), A01.array(), A00.array(), P00.array())
  e = pytave.feval(1, "abs", e[0])
  e = pytave.feval(1, "sort", e[0])
  e = e[0]

  print " condition number 2 - I RT ", e[-1]/e[0], e[-1]/e[1]   


  V = VectorFunctionSpace(mesh, "Lagrange", 2)
  Q = FunctionSpace(mesh, "Lagrange", 1)

#  bc = [DirichletBC(V, Constant((0,0)), boundary), None]
  bc = [None, None]
  u, v = TrialFunction(V), TestFunction(V)
  p, q = TrialFunction(Q), TestFunction(Q)


  p00 = inner(u,v)*dx   
  a00 = div(u)*div(v)*dx + inner(u,v)*dx   
  a01 = div(v)*p*dx  
  a10 = div(u)*q*dx  
  a11 = p*q*dx  
  P00 = assemble(p00)
  A00 = assemble(a00)
  A10 = assemble(a10)
  A01 = assemble(a01)
  A11 = assemble(a11)

  
  e = pytave.feval(1, "check_eigs_vs", A11.array(), A01.array(), A00.array(), P00.array())
  e = pytave.feval(1, "abs", e[0])
  e = pytave.feval(1, "sort", e[0])
  e = e[0]

  print " condition number 1 - div TH", e[-1]/e[0], e[-1]/e[1]   

  p00 = inner(u,v)*dx   
  a00 = inner(u,v)*dx   
  a01 = div(v)*p*dx  
  a10 = div(u)*q*dx  
  a11 = p*q*dx + inner(grad(p),grad(q))*dx  
  P00 = assemble(p00)
  A00 = assemble(a00)
  A10 = assemble(a10)
  A01 = assemble(a01)
  A11 = assemble(a11)

  
  e = pytave.feval(1, "check_eigs_vs", A11.array(), A01.array(), A00.array(), P00.array())
  e = pytave.feval(1, "abs", e[0])
  e = pytave.feval(1, "sort", e[0])
  e = e[0]

  print " condition number 2 - I TH", e[-1]/e[0], e[-1]/e[1]   






