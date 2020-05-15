from fenics import*
import time
import numpy as np
from dolfin import *; 
from mshr import *
import dolfin.common.plotting as fenicsplot
from matplotlib import pyplot as plt
from ufl import sign
import copy

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

#plt.figure()
#plot(mesh)
#plt.show()


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

###########################
## Finite element spaces ##
###########################


## Generate finite element spaces 
# V for velocity, Q for presssure, PHI for the level set
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)
PHI = FunctionSpace(mesh, "Lagrange", 2)


## Define trial and test functions 
u = TrialFunction(V)
p = TrialFunction(Q)
phi = TrialFunction(PHI)


v = TestFunction(V)
q = TestFunction(Q)
w = TestFunction(PHI)


## Define iteration functions
## (u0,p0) solution from previous time step
## (u1,p1) linearized solution at present time step  
u0 = Function(V)
u1 = Function(V)
p0 = Function(Q)
p1 = Function(Q)
phi0 = Function(PHI)
phi00 = Function(PHI)
phi1 = Function(PHI)
phi11 = Function(PHI)




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
pin = 1.0 #Expression('5.0*fabs(sin(t))', element = Q.ufl_element(), t=0.0)
pout = 0.0
bcp0 = DirichletBC(Q, pin, dbc_left) 
bcp1 = DirichletBC(Q, pout, dbc_right)

bcp = [bcp0, bcp1]


# Physical properties
rho1 = 100
rho2 = 0.1
mu1 = 100
mu2 = 0.1

# The initial state of the two domains, the signed function to be advected
# levelsetfunction = distance(x) - radius 
radius = 0.2
center = Point(1, 1)
ls = Expression('sqrt((x[0]-X) * (x[0]-X) + (x[1]-Y) * (x[1]-Y)) - r', degree=2, X=center[0], Y=center[1], r=radius)

# Initial signed function phi
phi0 = interpolate(ls,PHI)               

#plt.figure()
#plot(sign(phi0), interactive=True) 
#plt.show()

# Define measure for boundary integration  
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Set parameters for nonlinear and lienar solvers 
num_nnlin_iter = 5 
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default" 

# Set plot frequency
plot_time = 0
plot_freq = 10

# Time stepping 
T = 3
dt =  0.5*mesh.hmin()           
k = Constant (dt)
t = dt



###################
## Navier-Stokes ##
###################


def navierstokes(phi0):

    def rho(phi0):
        return (rho1 * 0.5* (1.0+ sign(phi0)) + rho2 * 0.5*(1.0 - sign(phi0)))

    def mu(phi0):
        return (mu1 * 0.5* (1.0+ sign(phi0)) + mu2 * 0.5*(1.0 - sign(phi0)))

    ## Mean velocities for trapozoidal time stepping
    um = 0.5*(u + u0)
    um1 = 0.5*(u1 + u0)

    # Stabilization parameters
    h = CellDiameter(mesh);
    u_mag = sqrt(dot(u1,u1))
    d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0))) 
    d2 = h*u_mag 

    Fu = rho(phi0) * inner((u - u0)/dt + grad(um)*um1, v)*dx \
        - p1*div(v)*dx + mu(phi0)*inner(grad(um), grad(v))*dx \
        + d1*inner((u - u0)/dt \
        + grad(um)*um1 + grad(p1), grad(v)*um1)*dx \
        + d2*div(um)*div(v)*dx 
    au = lhs(Fu)
    Lu = rhs(Fu)

    Fp = d1 * inner((u1 - u0)/dt + grad(um1)*um1 + grad(p), grad(q))*dx + div(um1)*q*dx 
    ap = lhs(Fp)
    Lp = rhs(Fp)

    # Assemble matrix and vector
    Au = assemble(au)
    bu = assemble(Lu)

    # Compute solution 
    [bc.apply(Au, bu) for bc in bcu]
    [bc.apply(u1.vector()) for bc in bcu]
    solve(Au, u1.vector(), bu, "bicgstab", "default")

    # Assemble matrix and vector
    Ap = assemble(ap) 
    bp = assemble(Lp)

    # Compute solution 
    [bc.apply(Ap, bp) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(Ap, p1.vector(), bp, "bicgstab", prec)

    u0.assign(u1)
    return (u1,p1)



#######################################
## Convection of the signed function ##
#######################################


def convection(uc, phi0):

    # Stabilization 
    d3 = 0.5*mesh.hmin() 
    
    Fconv = inner((phi1-phi0)/dt,w)*dx \
    + inner(dot(uc,grad(phi1)),w)*dx \
    + d3*dot(uc,grad(phi1))*dot(uc,grad(w))*dx

    solve(Fconv == 0, phi1)
    phi0.assign(phi1)
    return phi0



###################################################
## Numerical integration of the re-initialization ##
####################################################



def reinitialize(phi0):
 
    phi00.assign(phi0)

    tau=0
    dtau = 0.5*mesh.hmin()
    num_steps = int(5/dt)

    # Interface thickness
    xh =  mesh.hmin()
    eps = Constant(2 * xh)    

    # Numerical diffusion parameter 
    alpha = Constant(xh)

    # Smoothed signed function
    signphi = phi0 / sqrt(phi0*phi0 + eps*eps) 

    for _ in range(num_steps):
        tau += dtau
        
        # FEM linearization of reinitialization equation
        reina = (phi11 / k) * w * dx
        reinb = (phi00 / k) * w * dx \
            + signphi * (1.0 - sqrt(dot(grad(phi00), grad(phi00)))) * w * dx \
            #- alpha * inner(grad(phi0), grad(w))* dx

        solve(reina == reinb, phi11)

        phi00.assign(phi11)


    return phi00



###################
## Time stepping ##
###################


# Open files to export solution to Paraview
velocity_solution_export = File("levelsetresults/levelset_velocity_solution.pvd")
pressure_solution_export = File("levelsetresults/levelset_pressure_solution.pvd")
phi_solution_export = File("levelsetresults/levelset_phi_solution.pvd")
# signed_phi_solution_export = File("levelsetresults/levelset_signed_phi_solution.pvd")

while t < T + DOLFIN_EPS:

    ## Navier-Stokes
    velocity, pressure = navierstokes(phi0) 

    u1.assign(velocity)
    p1.assign(pressure)

    ## Convection
    phiconv = convection(velocity, phi0)

    #phi0.assign(phiconv)

    ## Reinitialization
    phire = reinitialize(phiconv)

    phi0.assign(phire)
  
    

    if t > plot_time:     
            
        s = 'Time t = ' + repr(t) 
        print(s)
    
        # Save solution to file
        # velocity_solution_export << u1
        # pressure_solution_export << p1
        # phi_solution_export << phi0
        # signed_phi = sign(phi0)
        # signed_phi_solution_export << signed_phi

        # Plot solution
        # plt.figure()
        # plot(u1, title="Velocity")
        # plt.show()

        # plt.figure()
        # plot(p1, title="Pressure")
        # plt.show()

        plt.figure()
        plot(sign(phi0), title="Phi", interactive=True)
        plt.show()

        plot_time += T/plot_freq
        
    
    u0.assign(u1)
    t += dt


