
import sys
from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.trilinos import *

from block.dolfin_util import *



def get_command_line_arguments():
    dict = {}
    if len(sys.argv) == 1: return dict
    for a in sys.argv[1:]:
        key, value = a.split('=')
        dict[key] = value
    return dict


def dump_matrix(filename, name, AA):
    f = open(filename, 'w')
    AA = AA.array()
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
           if abs(AA[i,j]) > 10e-10:
               f.write("%s (%d, %d) = %e;\n " % (name,i+1,j+1,AA[i,j]))



cl_args = get_command_line_arguments()

N = 4 
if cl_args.has_key("N"):
    N = int(cl_args["N"])   

mu_val = 1 
if cl_args.has_key("mu"):
    mu_val = float(cl_args["mu"])   

lambda_val = 1 
if cl_args.has_key("lambda"):
    lambda_val = float(cl_args["lambda"])   

true_eps = True 
if cl_args.has_key("eps"):
    true_eps = bool(cl_args["eps"])   

K_val = 1 
if cl_args.has_key("K"):
    K_val = float(cl_args["K"])   

mesh = UnitSquare(N, N)
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
VQ = MixedFunctionSpace([V,Q])
boundary = BoxBoundary(mesh)

mu = Constant(mu_val)
lmbda = Constant(lambda_val)
K = Constant(K_val)

u, p = TrialFunction(V), TrialFunction(Q)
v, q = TestFunction(V), TestFunction(Q)

def eps(u): 
    if true_eps: return sym(grad(u))
    else : grad(u)

a00 = mu*inner(eps(u), eps(v))*dx  + lmbda*div(u)*div(v)*dx 
a10 = div(u)*q*dx 
a01 = div(v)*p*dx 
a11 = K*inner(grad(p), grad(q))*dx 

bc_u = DirichletBC(VQ.sub(0), Constant((0,0)), boundary.all) 
bc_p = DirichletBC(VQ.sub(1), Constant(0), boundary.all) 

bcs = [bc_u, bc_p]

AA, AArhs = block_symmetric_assemble([[a00, a01],
                                      [a10, a11]], bcs=bcs)


[[A, B],
 [C, D]] = AA

dump_matrix("A.m", "A", A)
dump_matrix("B.m", "B", B)
dump_matrix("C.m", "C", C)
dump_matrix("D.m", "D", D)


ofile_str = """

A; 
B; 
C; 
D; 

AA = [A, B; C, -D]; 
BB1 = [A, 0*B; 0*C, D + C*inv(A)*B];
BB2 = [A, 0*B; 0*C, D];
BB3 = [A, 0*B; C, D]*[inv(A), 0*B; 0*C, inv(D)]*[A, B; 0*C, D];
 
e1 = sort(abs(qz(AA, BB1))); 
e2 = sort(abs(qz(AA, BB2))); 
e3 = sort(abs(qz(AA, BB3))); 

plot(e1); 
hold on; 
plot(e2); 
plot(e3); 
sleep(4);

e1(1)/e1(size(e1)(1))
e2(1)/e2(size(e2)(1))
e3(1)/e3(size(e3)(1))

"""

ofile = open("ofile.m", "w")
ofile.write(ofile_str)
ofile.close()
os.system("octave ofile.m")








