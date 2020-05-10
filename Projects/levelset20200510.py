from fenics import*
import time
import numpy as np
from dolfin import *; 
from mshr import *
import dolfin.common.plotting as fenicsplot
from matplotlib import pyplot as plt


# Needed for level set problem:
#   1. phi(x) = min|x-x_i|, domain1
#               -min|x-x_i|, domain2
#   2. Advection eq d_phi/d_t + grad(v * phi)
#   3. normal n = grad(ph)/|grad(phi)| -> n = grad(phi)
#   4. epsilon = thickness of the interface, associated with mesh size



##########
## Mesh ##
##########

L = 2
H = 2
resolution = 32

mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), L*resolution, H*resolution)

plt.figure()
plot(mesh)
plt.show()


#######################
## Define subdomains ##
#######################


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) 

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L)

class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H)
      
left = Left()
right = Right()
lower = Lower()
upper = Upper()

## Define mesh functions (for boundary conditions)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
lower.mark(boundaries, 3)
upper.mark(boundaries, 4)


################################
## Define boundary conditions ##
################################

# Define boundary conditions 
class DirichletBoundaryLower(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

class DirichletBoundaryUpper(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H)

class DirichletBoundaryLeft(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0) 

class DirichletBoundaryRight(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)

class DirichletBoundaryObjects(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (not near(x[0], 0.0)) and (not near(x[0], L)) and (not near(x[1], 0.0)) and (not near(x[1], H))

dbc_lower = DirichletBoundaryLower()
dbc_upper = DirichletBoundaryUpper()
dbc_left = DirichletBoundaryLeft()
dbc_right = DirichletBoundaryRight()
dbc_objects = DirichletBoundaryObjects()

# Velocity boundary conditions
uin = 1.0  #Expression('5.0*fabs(sin(t))', element = V.ufl_element(), t=0.0)
bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)
bcu_upp0 = DirichletBC(V.sub(0), 0.0, dbc_upper)
bcu_upp1 = DirichletBC(V.sub(1), 0.0, dbc_upper)
bcu_low0 = DirichletBC(V.sub(0), 0.0, dbc_lower)
bcu_low1 = DirichletBC(V.sub(1), 0.0, dbc_lower)
bcu_obj0 = DirichletBC(V.sub(0), 0.0, dbc_objects)
bcu_obj1 = DirichletBC(V.sub(1), 0.0, dbc_objects)

bcu = [bcu_in0, bcu_in1, bcu_upp0, bcu_upp1, bcu_low0, bcu_low1, bcu_obj0, bcu_obj1]

# Pressure boundary conditions
pin = Expression('5.0*fabs(sin(t))', element = Q.ufl_element(), t=0.0)
pout = 0.0
bcp0 = DirichletBC(Q, pin, dbc_left) 
bcp1 = DirichletBC(Q, pout, dbc_right)

bcp = [bcp0, bcp1]



###########################
## Finite element spaces ##
###########################


## Generate finite element spaces 
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)
C = FunctionSpace(mesh, "Lagrange", 1)


## Define trial and test functions 
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)


## Define iteration functions
## (u0,p0) solution from previous time step
## (u1,p1) linearized solution at present time step  
u0 = Function(V)
u1 = Function(V)
p0 = Function(Q)
p1 = Function(Q)


## Mean velocities for trapozoidal time stepping
um = 0.5*(u + u0)
um1 = 0.5*(u1 + u0)


###################
## Navier-Stokes ##
###################

rho1 = 1
rho2 = 0.1
mu1 = 1
mu2 = 0.1


def rho(phi0):
    return(rho1 * 0.5* (1.0+ sign(phi0)) + rho2 * 0.5*(1.0 - sign(phi0)))

def mu(phi0):
   return(mu1 * 0.5* (1.0+ sign(phi0)) + mu2 * 0.5*(1.0 - sign(phi0)))


Fu = rho(phi0) * inner((u - u0)/dt + grad(um)*um1, v)*dx \
    - p1*div(v)*dx + mu(phi0)*inner(grad(um), grad(v))*dx \
    + d1*inner((u - u0)/dt \
    + grad(um)*um1 + grad(p1), grad(v)*um1)*dx \
    + d2*div(um)*div(v)*dx 
au = lhs(Fu)
Lu = rhs(Fu)

Fp = d1*inner((u1 - u0)/dt + grad(um1)*um1 + grad(p), grad(q))*dx + div(um1)*q*dx 
ap = lhs(Fp)
Lp = rhs(Fp)


######################################
## Level set function and advection ##
######################################

radius = 0.2
center = Point(1, 1)


# The initial state of the two domains, the signed function to be advected
# levelsetfunction = distance(x) - radius 
ls = Expression('sqrt((x[0]-X) * (x[0]-X) + (x[1]-Y) * (x[1]-Y)) - r', degree=2, X=center[0], Y=center[1], r=radius)

# Time step
dt =  0.5*mesh.hmin()           
k = Constant (dt)

# Initial signed function phi
phi0 = interpolate(ls,C)               

# Interface thickness
eps = Constant (1.0 / dx)        

# Numerical diffusion parameter 
alpha = Constant (0.0625 / dx)   

# Define measure for boundary integration  
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Set parameters for nonlinear and lienar solvers 
num_nnlin_iter = 5 
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default" 



def convection(u, phi)
    """Convection equation""" 
    return phi



###################################################
## Numerical integration of the re-initialization ##
####################################################


def reinitialize(ls, mesh, dx):

    # Space definition
    PHI = FunctionSpace(mesh, "CG", 2)

    # Set the initial value
    phi = TrialFunction(PHI)
    phi0 = TrialFunction(PHI)
    w = TestFunction(PHI)

    # Normal of ls
    def normgrad(b):
        return (sqrt(b.dx(0)**2 + b.dx(1)**2))

    signp = ls / sqrt(ls*ls + eps*eps * normgrad(ls)*normgrad(ls))

    # FEM linearization of reinitialization equation
    a = (phi / k) * w * dx
    L = (phi0 / k) * w * dx \
        + signp * (1.0 - sqrt(dot(grad(phi0), grad(phi0)))) * w * dx \
        - alpha * inner(grad(phi0), grad(w))* dx

    solve (a == L, phi , bc)
    return phi