from block import *
from dolfin import *
from scipy.interpolate import splrep, splev
import numpy
import os

#Read in from file
def read_file(file_name):
    file_pres = open(file_name)
    lines_pres = file_pres.readlines()
    file_pres.close()
    len_file = len(lines_pres)
    print "length pressure file:", len_file
    
    start = 840 - 170
    stop = 1010
    
    pres = numpy.zeros(len_file) 
    for i in range(len_file):
        pres[i] = float(lines_pres[i])
    pres_heartbeat = numpy.array(pres[start:stop])
    pres_heartbeat += 2 # make sure we have positive pressure values
    pres_heartbeat *= 1.3332 # convert from mmHg to Pa/100
    print pres_heartbeat
    print len(pres_heartbeat)
    
    tt = numpy.linspace(0, (stop-start), num=(stop-start))
    tt = tt/200.#sampling frequence
    print len(tt)
    return tt, pres_heartbeat

file_name = "filtered_interval_1440000_to_1442000_WAVE_ICP-PAR_06072011_155906_16AE7DCB5-7C1_200.txt"

tt, pres_heartbeat = read_file(file_name)
pres_spline = splrep(tt, pres_heartbeat)

# Function spaces, elements

L = 0.5
mesh_name = "mesh_scaled.xml.gz"
if not os.path.exists(mesh_name):
    import urllib
    print 'Downloading mesh %s'%mesh_name
    urlbase='http://simula.no/~jobh/siam2013/'
    urllib.FancyURLopener().retrieve(urlbase+mesh_name, mesh_name)

mesh = Mesh(mesh_name)
mesh.order()

Nd = mesh.topology().dim()

V = VectorFunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 1)

v, omega = TrialFunction(V), TestFunction(V)
q, phi   = TrialFunction(W), TestFunction(W)

u_prev = Function(V)
p_prev = Function(W)

#========
# Define forms, material parameters, boundary conditions, etc.
boundary_markers = mesh.domains().facet_domains(mesh)
marker_file_boundary = File("boundary_markers.pvd")
marker_file_boundary << boundary_markers

domain_markers = mesh.domains().cell_domains(mesh)
marker_file_domain = File("domain_markers.pvd")
marker_file_domain << domain_markers

"""
boundary_markers = FacetFunction("uint", mesh)
boundary_markers.set_all(0)

class NeumanBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]-1) < DOLFIN_EPS
neuman_boundary = NeumanBoundary()

neuman_boundary.mark(boundary_markers, 1)

marker_file1 = File("boundary_markers_dolfin.pvd")
marker_file1 << boundary_markers
"""
ds = ds[boundary_markers]
dx = dx[domain_markers]
pmax = 8.0#800 Pa
v_wave = 200.0 # cm/s (for a period of 1 sec we get wavelength equal wavevelocity
u0 = Constant([0.0]*Nd)
#p0 = Expression("pmax*x[2]*sin(2*pi*t)", t=0, pmax=pmax)
#p0 = Expression("pmax*x[2]", pmax=pmax)
#p0 = Expression("pmax*sin((2*pi/v_w)*(x[2]+v_w*t))", v_w = v_wave, pmax=pmax, t=0)

def dx_times(form):
    return form*dx(0)+form*dx(1)+form*dx(2)

### Material parameters

#lmbda = Constant(1e5)
#mu    = Constant(1e5)
E = 50#00
nu = 0.35
mu    = E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))


b = Constant(0)
alpha = Constant(1.0)
#Lambda = Constant(1e-5)#Constant( numpy.diag([.02]*(Nd-1)+[.001]) )
K0 = Constant(1e-2)
K1 = Constant(1e-6)
K2 = Constant(1e-5)

t_n = Constant([0.0]*Nd)
f_n = Constant([0.0]*Nd)
n = FacetNormal(mesh)
h = mesh.hmax()
dt = Constant(.00125)
T = 0.85#3#0.02
on = 1
Q = Constant(0)

class applied_pres(Expression):
    def __init__(self):
        self.a = 0.0
    def set_time(self, t):
        self.a = t#splev(t, pres_spline)#fluid_p/upm * sin(2*pi*t)
    def eval(self, value, x):
        y = x[2]  # scale to [0,1]
        #print type(y), type(self.a), type(v_wave)
        value[0] = splev(1./v_wave*(y + v_wave*self.a) % 2.0, pres_spline)
        #value[0] = sin((2*pi)/v_wave*(y+v_wave*self.a))
        
p0 = applied_pres()

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(Nd)
def v_D(Lambda, q):
    return -Lambda*grad(q)
def coupling(w,r):
    return - alpha * r * div(w)

def corr(q):
    C = h**2/(4*(lmbda+2*mu))
    return C*grad(q)


a00 = dx_times(inner(grad(omega), sigma(v)))
a01 = dx_times(coupling(omega,q))
a10 = dx_times(coupling(v,phi))
a11 = dx_times(-(b*phi*q - dt*inner(grad(phi),v_D(K0, q)) + on*inner(corr(q), grad(phi))))

L0 = inner(omega, -p0*n)*ds(7)
L1 =  dx_times(coupling(u_prev,phi) - (Q*dt + b*p_prev)*phi - on*inner(corr(p_prev), grad(phi)))

# Create boundary conditions.
bcu0 = DirichletBC(V, u0, 3)
bcu1 = DirichletBC(V, u0, 4)
bcu_z5 = DirichletBC(V.sub(2), Constant(0), 5)
bcu_z6 = DirichletBC(V.sub(2), Constant(0), 6)
bcu_z8 = DirichletBC(V.sub(2), Constant(0), 8)
bcu_z9 = DirichletBC(V.sub(2), Constant(0), 9)

bcp = DirichletBC(W, p0, 7)
bcs = [[bcu0, bcu1, bcu_z5,bcu_z6,bcu_z8,bcu_z9], [bcp]]

# Assemble the matrices
# Insert the matrices into blocks

AA, AA_ = block_symmetric_assemble([[a00, a01],
                                    [a10, a11]], bcs=bcs)
bb = block_assemble([L0, L1], symmetric_mod=AA_, bcs=bcs)
