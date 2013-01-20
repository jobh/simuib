
import sys
from dolfin import *
from block import *
from block.dolfin_util import *
import numpy

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
    f.write("%s = sparse(%d,%d);"%(name,AA.shape[0],AA.shape[1]))
    for (i,j) in zip(*numpy.where(AA)):
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

dump_matrix("Am.m", "A", A)
dump_matrix("Bm.m", "B", B)
dump_matrix("Cm.m", "C", C)


ofile_str = """
format compact;
Am; Bm; Cm;
BT = transpose(B);

IA = eye(size(A));
IC = eye(size(C));
B0 = zeros(size(B));
BT0 = zeros(size(BT));

Ai = full(inv(A)); Ci = full(inv(C));

Aii = inv(diag(diag(A)));
S = C - BT*Aii*B;
Si = full(inv(S));

AA = [A, B; BT, C];
BB1 = [Ai, B0; BT0, Ci];
BB2 = [Ai, B0; BT0, Si];
BB3 = [IA, -Ai*B; BT0, IC]*[Ai, B0; BT0, Si]*[IA, B0; -BT*Ai, IC];

semilogy(1);
hold on;

s=sort(svd(BB1*AA)); k1=s(end)/s(1), semilogy(s, 'b'), drawnow;
s=sort(svd(BB2*AA)); k2=s(end)/s(1), semilogy(s, 'r'), drawnow;
s=sort(svd(BB3*AA)); k3=s(end)/s(1), semilogy(s, 'g'), drawnow;
"""

ofile = open("ofile.m", "w")
ofile.write(ofile_str)
ofile.close()
success = (0 == os.system("matlab -nodesktop -nosplash -r 'try, ofile, pause(4), end, exit' 2>/dev/null"))
if not success:
    os.system("octave -q --eval 'ofile, pause(4)'")
