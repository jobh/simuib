
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

N          = int  (cl_args.get("N",        4))
mu_val     = float(cl_args.get("mu",       1))
lambda_val = float(cl_args.get("lambda",   1))
true_eps   = bool (cl_args.get("true_eps", True))
K_val      = float(cl_args.get("K",        1))

mesh = UnitSquareMesh(N, N)
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
 [_, C]] = AA

dump_matrix("A.m", "A", A)
dump_matrix("B.m", "B", B)
dump_matrix("C.m", "C", C)


ofile_str = """
A; B; C;
BT = transpose(B);

S = C - BT*inv(A)*B;

AA = [A, B; BT, C];
BB1 = [A, 0*B; 0*BT, S];
BB2 = [A, 0*B; 0*BT, C];
BB3 = [A, 0*B; BT, C]*[inv(A), 0*B; 0*BT, inv(C)]*[A, B; 0*BT, C];

hold on;
e1 = sort(abs(qz(AA, BB1))); plot(e1, 'b;1;'); e1(end)/e1(1), drawnow;
e2 = sort(abs(qz(AA, BB2))); plot(e2, 'r;2;'); e2(end)/e2(1), drawnow;
e3 = sort(abs(qz(AA, BB3))); plot(e3, 'g;3;'); e3(end)/e3(1), drawnow;

sleep(4);
"""

ofile = open("ofile.m", "w")
ofile.write(ofile_str)
ofile.close()
os.system("octave -q ofile.m")
