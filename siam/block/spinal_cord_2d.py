from __future__ import division

from block import *
from dolfin import *
import numpy

from scipy.interpolate import splrep, splev

#number of elements in x-dir -remember should equal the one in creating_parameters.py:
E_pia = 2.3e6 #Pa 
if cl_args.has_key("E_pia"):
    E_pia = float(cl_args["E_pia"])   
print "E_pia = ", E_pia 

E_tissue = 5e3 #Pa 
if cl_args.has_key("E_tissue"):
    E_tissue = float(cl_args["E_tissue"])   
print "E_tissue = ", E_tissue 

_nu = 0.35 
if cl_args.has_key("nu"):
    _nu = float(cl_args["nu"])   
print "nu = ", _nu

pia_perm = 1
if cl_args.has_key("pia_perm"):
    pia_perm = int(cl_args["pia_perm"])   
print "pia_perm = ", pia_perm 


caps_perm = 1
if cl_args.has_key("caps_perm"):
    caps_perm = int(cl_args["caps_perm"])   
print "caps_perm = ", caps_perm 

l_cc = 1
if cl_args.has_key("l_cc"):
    l_cc = float(cl_args["l_cc"])   
print "l_cc = ", l_cc
 
# homogen permeability:
K_pia = 3.1e-15#1.82e-14
if cl_args.has_key("K_pia"):
    K_pia = float(cl_args["K_pia"])   
print "K_pia = ", K_pia 

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
    
    tt = numpy.linspace(0, (stop-start), num=(stop-start))
    tt = tt/200.#sampling frequence
    return tt, pres_heartbeat

file_name = "filtered_interval_1440000_to_1442000_WAVE_ICP-PAR_06072011_155906_16AE7DCB5-7C1_200.txt"

tt, pres_heartbeat = read_file(file_name)
pres_spline = splrep(tt, pres_heartbeat)

# length units per meter (normally cm)
upm = 100

v_wave = 200

# size of grid in cm
m = 1
n = 2

# cells per cm
N  = int(cl_args.get('N', 32)) # should be even

# scale to length units
m *= upm/100; n *= upm/100; N /= upm/100

# Function spaces, elements
mesh = RectangleMesh(0, 0, m, n, int(N*m), int(N*n))
Nd = mesh.topology().dim()

V = VectorFunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 1)

v, omega = TrialFunction(V), TestFunction(V)
q, phi   = TrialFunction(W), TestFunction(W)

u_prev = Function(V)
p_prev = Function(W)


#========
# Define forms, material parameters, boundary conditions, etc.

central_canal = 1
grey_matter   = 2
white_matter  = 3
pia           = 4
def where(x):
    #radial distance from center of mesh
    r = abs(1 - 2*x[0]/m) - DOLFIN_EPS
    y = abs(1 - 2*x[1]/n) - DOLFIN_EPS
    if r < 0.1 and y < l_cc:
        return central_canal
    elif r < 0.5:
        return grey_matter
    elif r < 0.94:
        return white_matter
    elif r <= 1.0:
        return pia #assumed pia to be 300 micrometer -Elliott 150 micrometer
    else:
        raise RuntimeError()

### Material parameters -note mesh in cm -rest SI units! 1 funny unit = 100 Pa 

class nu(Expression):
    def eval(self, value, x):
        w = where(x)
        value[0] = _nu
        #value[0] = 0.25 if where(x)==central_canal else 0.45
        #value[0] = 0.25 if where(x)==central_canal else 0.35
nu = nu()

class E(Expression):
    def eval(self, value, x):
        #value[0] = 421 / upm
        value[0] = E_pia / upm if where(x)==pia else E_tissue/upm
E = E()

class K(Expression):
    def eval(self, value, x):
        #viscosity of water [Pa*s=kg/m/s^2 -> kg/cm/s^2]:
        mu_w = 9.11e-4 / upm
        scale = upm**2    # [m*m -> cm*cm]
        w = where(x)
        if w == central_canal:
            value[0] = (1e-12)*scale/mu_w     # ~ 1e-3 cm^2/s
        elif w == grey_matter:
            value[0] = (1.82e-15)*scale/mu_w  # ~ 2e-6 cm^2/s
        elif w == white_matter:
            value[0] = (1.82e-14)*scale/mu_w  # ~ 2e-5 cm^2/s
        elif w == pia:
            value[0] = K_pia*scale/mu_w

K = K()

lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))
mu =  E / (2.0*(1.0 + nu))
lmbdamu = 4*(lmbda + 2*mu)
lmbdamuInv = 1.0/lmbdamu
h = mesh.hmax()
t_n = Constant([0.0]*Nd)
f_n = Constant([0.0]*Nd)

#dt = Constant(.005)
dt = Constant(0.00125)
T = 0.00125#1

Q = Constant(0)

beta = 2*mu + Nd*lmbda
Kdr = beta/Nd
Kf = Constant(2.2e9/upm)
b = 0.5/Kf + 0.5/Kdr
alpha = Constant(1.0)

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(Nd)
def v_D(q):
    return -K*grad(q)
def coupling(w,r):
    return - alpha * r * div(w)
def corr(test, trial):
    return 0*lmbdamuInv*h**2*inner(grad(test), grad(trial))

a00 = inner(grad(omega), sigma(v)) * dx
a01 = coupling(omega,q) * dx
a10 = coupling(v,phi) * dx
a11 = -(b*phi*q - dt*inner(grad(phi),v_D(q))) * dx - corr(phi, q)*dx

# How much the pressure boundary affects the fluid and the solid

solid_p = 1.0#0.4
fluid_p = 1.0

# Create boundary conditions.

def left_right_boundary(x=True):
    return near(x[0], 0) or near(x[0], m)
def top_bottom_boundary(x=True):
    return near(x[1], 0) or near(x[1], n)
class left_right_boundary_subdomain(SubDomain):
    def inside(self, x, inside):
        return left_right_boundary(x)
left_right_boundary_subdomain().mark_facets(mesh, 1)

def boundary_zero_y(x):
    return top_bottom_boundary(x)
def boundary_zero_x(x):
    return top_bottom_boundary(x) and near(x[0], m/2)
"""
class applied_pres(Expression):
    def __init__(self):
        self.a = 0.0
    def set_time(self, t):
        self.a = fluid_p/upm * sin(2*pi*t)
    def eval(self, value, x):
        y = x[1]/n  # scale to [0,1]
        if   y <  0.4:  p =   3.12*y - 81.50
        elif y <= 0.6:  p = 401.25*y -180.25
        else:           p =   3.12*y + 80.25
        value[0] = self.a*p
"""
class applied_pres(Expression):
    def __init__(self):
        self.a = 0.0
    def set_time(self, t):
        self.a = t#splev(t, pres_spline)#fluid_p/upm * sin(2*pi*t)
    def eval(self, value, x):
        y = x[1]/n  # scale to [0,1]
        #print type(y), type(self.a), type(v_wave)
        value[0] = splev(1./v_wave*(y + v_wave*self.a) % 2.0, pres_spline)
        #value[0] = sin((2*pi)/v_wave*(y+v_wave*self.a))

applied_pres = applied_pres()

normal = FacetNormal(mesh)
L0 = inner(omega, t_n) * ds \
    + Constant(-solid_p) * applied_pres * inner(omega, normal) * ds(1)
L1 = coupling(u_prev,phi) * dx - (Q*dt + b*p_prev)*phi * dx - corr(phi,p_prev)*dx

bc_disp_x  = DirichletBC(V.sub(0), 0.0, boundary_zero_x, method="pointwise")
bc_disp_y  = DirichletBC(V.sub(1), 0.0, boundary_zero_y)
bc_pres_pia  = DirichletBC(W, applied_pres, left_right_boundary)
bc_pres_caps = DirichletBC(W, applied_pres, top_bottom_boundary)

if pia_perm == 1 and caps_perm == 1:
    bcs = [[bc_disp_x, bc_disp_y], [bc_pres_caps, bc_pres_pia]]
elif pia_perm == 0 and caps_perm == 1:
     bcs = [[bc_disp_x, bc_disp_y], [bc_pres_caps]]
elif pia_perm == 1 and caps_perm == 0:
    bcs = [[bc_disp_x, bc_disp_y], [bc_pres_pia]]
elif pia_perm == 0 and caps_perm == 0:
     bcs = [[bc_disp_x, bc_disp_y], None]

#bcs = [[bc_disp_x, bc_disp_y], []]

#h = mesh.hmin()
# Assemble the matrices and vectors

AA = block_assemble([[a00, a01],
                     [a10, a11]], bcs=bcs)

bb = block_assemble([L0, L1], bcs=bcs)
