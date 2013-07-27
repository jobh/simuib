
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
  V = VectorFunctionSpace(mesh, "Lagrange", 2)
  Q = FunctionSpace(mesh, "Lagrange", 1)
  bc = [DirichletBC(V, Constant((0,0)), boundary), None]
  u, v = TrialFunction(V), TestFunction(V)
  p, q = TrialFunction(Q), TestFunction(Q)

  a00 = inner(epsilon(u), epsilon(v))*dx  
  a01 = div(v)*p*dx  
  a10 = div(u)*q*dx  
  a11 = p*q*dx  
  AA, AArhs = block_symmetric_assemble([[a00, a01], [a10, a11]], bcs=bc)

  
  e = pytave.feval(1, "check_eigs", AA[0,0].array(), AA[1,0].array(), AA[1,1].array())
  e = pytave.feval(1, "abs", e[0])
  e = pytave.feval(1, "sort", e[0])
  e = e[0]

  print " condition number 1 - eps ", e[-1]/e[0], e[-1]/e[1]  



  a00 = div(u)*div(v)*dx + inner(u,v)*dx   
  a01 = div(v)*p*dx  
  a10 = div(u)*q*dx  
  a11 = p*q*dx  
  AA, AArhs = block_symmetric_assemble([[a00, a01], [a10, a11]], bcs=bc)

  
  e = pytave.feval(1, "check_eigs", AA[0,0].array(), AA[1,0].array(), AA[1,1].array())
  e = pytave.feval(1, "abs", e[0])
  e = pytave.feval(1, "sort", e[0])
  e = e[0]

  print " condition number 2 - div ", e[-1]/e[0], e[-1]/e[1]   

  a00 = inner(grad(u), grad(v))*dx  
  a01 = div(v)*p*dx  
  a10 = div(u)*q*dx  
  a11 = p*q*dx  
  AA, AArhs = block_symmetric_assemble([[a00, a01], [a10, a11]], bcs=bc)

  
  e = pytave.feval(1, "check_eigs", AA[0,0].array(), AA[1,0].array(), AA[1,1].array())
  e = pytave.feval(1, "abs", e[0])
  e = pytave.feval(1, "sort", e[0])
  e = e[0]

  print " condition number 3 - grad ", e[-1]/e[0], e[-1]/e[1]    


  c = Constant(1000000)
  a00 = c*div(u)*div(v)*dx + inner(epsilon(u),epsilon(v))*dx   
  a01 = div(v)*p*dx  
  a10 = div(u)*q*dx  
  a11 = (1/c)*p*q*dx  
  AA, AArhs = block_symmetric_assemble([[a00, a01], [a10, a11]], bcs=bc)

  e = pytave.feval(1, "check_eigs", AA[0,0].array(), AA[1,0].array(), AA[1,1].array())
  e = pytave.feval(1, "abs", e[0])
  e = pytave.feval(1, "sort", e[0])
  e = e[0]

  print " condition number 4 - div + eps ", e[-1]/e[0], e[-1]/e[1]
  print "eigenvalues " , e     



