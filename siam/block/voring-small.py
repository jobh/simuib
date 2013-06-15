import numpy
from dolfin import *
import ufl
parameters['reorder_dofs_serial'] = False

dir = 'diffpack/voring-small/'
mesh = Mesh(dir+'small-geom.xml.gz')
print 'Loaded mesh'
P0 = FunctionSpace(mesh, 'DG', 0)
P1 = FunctionSpace(mesh, 'CG', 1)

isSubd = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]

#mf = MeshFunction("size_t", mesh, 3, 0)
#mesh.domains().markers(3).assign(mf)

#markers = mesh.domain().markers(3)
#domains = CellFunction("size_t", mesh, 0)
facies = MeshFunction("size_t", mesh, mesh.domains().markers(3)).array()
#dx = dx[mf]
#plot(Function(P1), assemble(TestFunction(P1)*Constant(1)*dx(1)))
#plot(mf)
#interactive()
#exit()

#print 'Loaded fields'

V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)

v, omega = TestFunction(V), TrialFunction(V)
q, phi   = TestFunction(Q), TrialFunction(Q)

def map_over_facies(arg):
    return tuple(arg(i) for i in range(len(isSubd)) if isSubd[i])

dx_all = ufl.integral.MeasureSum(*map_over_facies(dx))
M_inv = LinearSolver('cg', 'amg')
M_inv.set_operator(assemble(inner(q,phi)*dx_all))

for i in [1,2,3,4]:
    v = assemble(q*dx(i))
    M_inv.solve(v, v.copy())
    print v.array().max()
    plot(Function(Q, v), hide_below=0.1, interactive=True)
exit()

u = Function(V)
p = Function(Q)

### Material parameters

# set tmp0 = MATERIAL_CONSTANTS = 2.5 2.5 1.52 2.5 2.5 2.5 2.3 2.5 2.5 2.5 2.1 2 2.3 0 0 1.3 2.3 ;
# set tmp2 = ELEMENT_FILE = DIR/tn16.field.gz
# set permeabilityY = RPN = 10 tmp2 pow
# set permeabilityX = RPN = 10 tmp0 pow permeabilityY *
# set porosity = ELEMENT_FILE = DIR/tn2.field.gz
# set youngmodule  = MATERIAL_CONSTANTS = .5e9 .5e9 .5e9 .5e9 .5e9 .5e9 .5e9 .5e9 .5e9 .5e9 .5e9 .8e9 .5e9 1e9 0 .9e9 .5e9 ;
# set poissonratio = MATERIAL_CONSTANTS = 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.25 0.35 0.4 0 0.20 0.35 ;
# set bulkmoduleS = MATERIAL_CONSTANTS = 10e9 10e9 10e9 10e9 10e9 10e9 10e9 10e9 10e9 10e9 10e9 5e9 10e9 20e9 0 4e9 10e9 ;
# set bulkmoduleW = CONSTANT = 4.3e+9
# set densityS = MATERIAL_CONSTANTS = 2680 2680 2680 2680 2680 2680 2667.5 2680 2680 2680 2655 2669 2667 2750 0 2666 2667.5 ;
# set densityW = CONSTANT = 1020
# set viscosity = CONSTANT = 0.002
# set biot = CONSTANT = 1.0

permratio = \
    numpy.power(10, [0, 2.5, 2.5, 1.5, 2.5, 2.5, 2.5, 2.3, 2.5, 2.5,
                   2.5, 2.1, 2.0, 2.3, 0.0, 0,   1.3, 2.3])

heatcapS        = [0,   891, 891, 854, 891, 891, 891, 887, 891, 891,
                   891, 887, 824, 879, 787, 0,   795, 887]
youngmodule     = [0,   5e8, 5e8, 5e8, 5e8, 5e8, 5e8, 5e8, 5e8, 5e8,
                   5e8, 5e8, 8e8, 5e8, 1e9, 0,   9e8, 5e8]
poissonratio    = [0,   .35, .35, .35, .35, .35, .35, .35, .35, .35,
                   .35, .35, .25, .35, .40, 0,   .20, .35]

porosity = Function(P0); porosity.vector()[:] =     numpy.loadtxt(dir+'small-tn2.raw.gz')
permZ    = Function(P0); permZ   .vector()[:] = 10**numpy.loadtxt(dir+'small-tn16.raw.gz')

class ByFacies(Expression):
    def __init__(self, lst):
        self.lst = lst
    def eval_cell(self, values, x, cell):
        facie = facies[cell.index]
        values[0] = self.lst[facie]

nu = ByFacies(poissonratio)
G  = ByFacies(youngmodule)
permRatio = ByFacies(permratio)

LambdaZ = permZ / viscosity
LambdaXY = permZ * permRatio / viscosity

lmbda = Constant(1e4)
mu    = Constant(1e3)
dt    = Constant(1)
b     = Constant(1e-6)
alpha = Constant(1.0)

class Permeability(Expression):
    def value_shape(self):
        return (Nd,Nd)
    def eval(self, tensor, x):
        tensor.shape = self.value_shape()
        tensor[:] = 0.0
        for d in range(Nd):
            if 0.0 <= x[-1] < 0.5:
                tensor[d,d] = 1.0
            else:
                tensor[d,d] = delta
Lambda = Permeability()

t_n = Constant( [0.0]*Nd )

T = 0.1

r = Constant(0)

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(Nd)
def v_D(q):
    return -Lambda*grad(q)
def coupling(w,r):
    return - alpha * r * div(w)

a00 = inner(grad(omega), sigma(v)) * dx
a01 = coupling(omega,q) * dx
a10 = coupling(v,phi) * dx
a11 = -(b*phi*q - dt*inner(grad(phi),v_D(q))) * dx

L0 = dot(t_n, omega) * ds
L1 = coupling(u,phi) * dx - (r*dt + b*p)*phi * dx

# Create boundary conditions.

bc_u_bedrock        = DirichletBC(V, [0.0]*Nd, lambda x,bdry: bdry and x[-1] <= 1/N/3)
bc_p_drained_top    = DirichletBC(Q,  0.0,     lambda x,bdry: bdry and x[-1] >= 1-1/N/3)

bcs = [bc_u_bedrock, bc_p_drained_top]
#bcs = [bc_u_bedrock, None]

# Assemble the matrices and vectors

AA, AAns = block_symmetric_assemble([[a00,a10],[a01,a11]], bcs=bcs)
bb = block_assemble([L0,L1], bcs=bcs, symmetric_mod=AAns)
