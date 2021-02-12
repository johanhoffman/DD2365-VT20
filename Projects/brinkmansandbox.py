# Load neccessary modules.
#from google.colab import files

import numpy as np
import time

from dolfin import *; 
from mshr import *
    
import dolfin.common.plotting as fenicsplot

from matplotlib import pyplot as plt

## Physical parameters
# Set viscosity and density
mu = 0.4
mu_air = 0.4
mu_ink = 0.4
rho = 1
rho_air = 1
rho_ink = 1
mu_brink = 1
mu_brink_air = 1
mu_brink_ink = 1
K_perm = 0.1
eps_por = 0.1

# Define rectangular domain 
L = 4
H = 2

# Define circle
xc = 0.5*L
yc = 0.5*H
rc = 0.2

# Define subdomains (for boundary conditions)
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


###################
## Generate mesh ##
###################

resolution = 16

mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(xc,yc),rc) - Circle(Point(xc,yc+1),rc) - Circle(Point(xc,yc-1),rc), resolution)

#plt.figure()
#plot(mesh)
#plt.show()

# Define mesh functions (for boundary conditions)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
lower.mark(boundaries, 3)
upper.mark(boundaries, 4)



# Generate finite element spaces (for velocity and pressure)
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions 
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Define iteration functions
# (u0,p0) solution from previous time step
# (u1,p1) linearized solution at present time step  
u0 = Function(V)
u1 = Function(V)
p0 = Function(Q)
p1 = Function(Q)

# Mean velocities for trapozoidal time stepping
um = 0.5*(u + u0)
um1 = 0.5*(u1 + u0)


################################
## Define boundary conditions ##
################################

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

# Examples of time dependent and stationary inflow conditions
#uin = Expression('4.0*x[1]*(1-x[1])', element = V.sub(0).ufl_element())
#uin = Expression('1.0 + 1.0*fabs(sin(t))', element = V.sub(0).ufl_element(), t=0.0)

uin = 1.0
bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)
bcu_upp0 = DirichletBC(V.sub(0), 0.0, dbc_upper)
bcu_upp1 = DirichletBC(V.sub(1), 0.0, dbc_upper)
bcu_low0 = DirichletBC(V.sub(0), 0.0, dbc_lower)
bcu_low1 = DirichletBC(V.sub(1), 0.0, dbc_lower)
bcu_obj0 = DirichletBC(V.sub(0), 0.0, dbc_objects)
bcu_obj1 = DirichletBC(V.sub(1), 0.0, dbc_objects)

pin = Expression('5.0*fabs(sin(t))', element = Q.ufl_element(), t=0.0)
pout = 0.0
#bcp0 = DirichletBC(Q, pin, dbc_left) 
bcp1 = DirichletBC(Q, pout, dbc_right)

#bcu = [bcu_in0, bcu_in1, bcu_upp0, bcu_upp1, bcu_low0, bcu_low1, bcu_obj0, bcu_obj1]
bcu = [bcu_in0, bcu_in1, bcu_upp1, bcu_low1, bcu_obj0, bcu_obj1]
bcp = [bcp1]

# Define measure for boundary integration  
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


# Set parameters for nonlinear and lienar solvers 
num_nnlin_iter = 5 
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default" 

# Time step length 
dt = 0.5*mesh.hmin() 



################################
## Define variational problem ##
################################


# Stabilization parameters
h = CellDiameter(mesh);
u_mag = sqrt(dot(u1,u1))
d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
d2 = h*u_mag

# terms of Navier-Stokes-Brinkman to be solved
brinkmanterm = mu_brink / eps_por * um 
materialderivative = rho * ((u - u0)/dt + grad(um)*um1)

# Momentum variational equation on residual form
Fu = inner(materialderivative, v) * dx \
    - p1 * div(v) * dx \
    + mu * inner(grad(um), grad(v)) * dx \
    + d1 * inner((u - u0)/dt + grad(um)*um1 + grad(p1), grad(v)*um1) * dx \
    + d2 * div(um)*div(v)*dx 
au = lhs(Fu)
Lu = rhs(Fu)

# Continuity variational equation on residual form
Fp = d1 * inner((u1 - u0)/dt + grad(um1)*um1 + grad(p), grad(q))*dx \
    + div(um1) * q * dx 
ap = lhs(Fp)
Lp = rhs(Fp)


# Open files to export solution to Paraview
velocity_solution_export = File("brinkmansandboxresultat/u_NS_test.pvd")
pressure_solution_export = File("brinkmansandboxresultat/p_NS_test.pvd")



###################
## Time stepping ##
###################

T = 30
t = dt

# Set plot frequency
plot_time = 0
plot_freq = 5

start_sample_time = 1.0


while t < T + DOLFIN_EPS:

    #s = 'Time t = ' + repr(t) 
    #print(s)

    pin.t = t
    #uin.t = t

    # Solve non-linear problem 
    k = 0
    while k < num_nnlin_iter: 
        
        # Assemble momentum matrix and vector 
        Au = assemble(au)
        bu = assemble(Lu)

        # Compute velocity solution 
        [bc.apply(Au, bu) for bc in bcu]
        [bc.apply(u1.vector()) for bc in bcu]
        solve(Au, u1.vector(), bu, "bicgstab", "default")

        # Assemble continuity matrix and vector
        Ap = assemble(ap) 
        bp = assemble(Lp)

        # Compute pressure solution 
        [bc.apply(Ap, bp) for bc in bcp]
        [bc.apply(p1.vector()) for bc in bcp]
        solve(Ap, p1.vector(), bp, "bicgstab", prec)

        # Save solution to file
        # velocity_solution_export << (u1,t)
        # pressure_solution_export << (p1,t)

        k += 1



    #if t > plot_time:     
        
        #s = 'Time t = ' + repr(t) 
        #print(s)
    
        # Save solution to file
        #velocity_solution_export << (u1,t)
        #pressure_solution_export << (p1,t)

        # Plot solution
        #plt.figure()
        #plot(u1, title="Velocity")

        #plt.figure()
        #plot(p1, title="Pressure")

        #plot_time += T/plot_freq
        
        #plt.show()


    # Update time step
    u0.assign(u1)
    t += dt


# Plot solution
plt.figure()
plot(u1, title="Velocity")

plt.figure()
plot(p1, title="Pressure")
plt.show()
