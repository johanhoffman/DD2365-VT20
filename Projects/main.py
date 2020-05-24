from fenics import *
import mesh as msh
import numpy as np
import matplotlib.pyplot as plt
import time
import sys



T = 10 			# final time
# num_steps = 5000   	# number of time steps
# dt = T / num_steps 	# time step size
mu = 10**(-5)		# dynamic viscosity
rho = 1.4			# density
uin = 1.0			# inflow velocity

aspect_ratio = 1/5
angle = 10
resolution = 64+32
double = False
_round = False


ii = 0
while ii < len(sys.argv):
	if("-a" in sys.argv[ii]):
		angle = int(sys.argv[ii+1])
		ii += 1
	elif("-r" in sys.argv[ii]):
		aspect_ratio = 1/int(sys.argv[ii+1])
		ii += 1
	elif("-q" in sys.argv[ii]):
		resolution = int(sys.argv[ii+1])
		ii += 1
	elif("-d" in sys.argv[ii]):
		double = True
	elif("-c" in sys.argv[ii]):
		_round = True

	ii += 1


mesh, (xmax, ymax), (l0,l1,h0,h1) = msh.basicDomain(angle, aspect_ratio, resolution, double=double, round=_round);
# plt.figure()
# plot(mesh)
# plt.show()

print("Re: ", int(round(rho*uin*ymax/mu,0)))
print("Aspect ratio:", round(aspect_ratio, 4))
print("Angle:", angle)
print("Resolution:", resolution)
print("Round:", _round)

# print("Aspect ratio:", round(aspect_ratio, 4), "Angle:", angle, "Resolution:", resolution, file=sys.stderr)
print("Mesh size:", mesh.num_cells())

root = "data"
if(double):
	root = "double"
if(_round):
	angle = "_round"
folder = "{}/res{}/ang{}/asp{}/".format(root, resolution, angle, round(aspect_ratio, 3))

## ============================================================= ##
# Define function spaces

V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define boundaries
inflowBoundary = CompiledSubDomain("on_boundary && near(x[0], 0)")
outflowBoundary = CompiledSubDomain("on_boundary && near(x[0], "+str(xmax)+")")
# slipBoundary = CompiledSubDomain("on_boundary && !(near(x[0], 0) || near(x[0], "+str(xmax)+"))")
wallBoundary = CompiledSubDomain("on_boundary && (near(x[1], 0) || near(x[1], "+str(ymax)+"))");
objectBoundary = CompiledSubDomain("on_boundary && !(near(x[0], 0) || near(x[1], 0) || near(x[0], "+str(xmax)+") || near(x[1],"+str(ymax)+"))");

## ============================================================= ##
# Define Dirichlet boundary conditions

# uinflow = Expression(("uin*6*x[1]*(YMAX-x[1])/(YMAX*YMAX)", "0.0"), element=V.ufl_element(), YMAX=ymax, uin=uin);
# uinflow = Expression(("t < uin ? t : uin", "0.0"), element=V.ufl_element(), uin=uin, t=0.0)

uinflow = Constant((1.0, 0.0))

# bcu_inflow = DirichletBC(V, Constant((uin, 0.0)), inflowBoundary)
bcu_inflow = DirichletBC(V, uinflow, inflowBoundary)
bcu_outflow = DirichletBC(V, uinflow, outflowBoundary)
# bcu_walls = DirichletBC(V, Constant((0.0, 0.0)), slipBoundary)
bcp_outflow = DirichletBC(Q, Constant(0), outflowBoundary)
bcu = [bcu_inflow, bcu_outflow] #, bcu_walls]
bcp = [bcp_outflow]

## ============================================================= ##
# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u0 = Function(V)
u1  = Function(V)
p0 = Function(Q)
p1  = Function(Q)

# Mark boundaries for integration.
ib = 1
ob = 2
sb = 3
wb = 4
# objb = 5
 
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
inflowBoundary.mark(boundaries, ib)
outflowBoundary.mark(boundaries, ob)
# slipBoundary.mark(boundaries, sb)
objectBoundary.mark(boundaries, sb)
wallBoundary.mark(boundaries, wb)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

## ============================================================= ##
## Define variational problem

# Define expressions used in variational forms
dt = 0.1*aspect_ratio*mesh.hmin()
f  = Constant((0, 0))
k  = Constant(dt)
nu = Constant(mu/rho);
h = CellDiameter(mesh);
n  = FacetNormal(mesh)
u_mag = sqrt(dot(u1,u1))
d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
d2 = h*u_mag
gamma = 100/h
theta = 1/(500*h)
e = Constant(5)

# Mean velocities for trapozoidal time stepping
um = 0.5*(u + u0)
um1 = 0.5*(u1 + u0)

# project u on vector parallel to facet
def tau(u):
	global n
	return u - dot(u,n)*n

# Conbined slip boundarys (walls and objects)
ds1 = (ds(sb)+ds(wb))
ds2 = (ds(sb)+ds(wb)+ds(ob))
ds3 = (ds(sb)+ds(wb)+ds(ib))

# Momentum variational equation on residual form
Fu = (
	## Navier stokes on weak residual form
	inner((u - u0)/dt + grad(um)*um1, v)*dx 
	+ nu*inner(grad(um), grad(v))*dx
	- p1*div(v)*dx
	## Source term
	# - inner(f, v);

	## Stabilization terms
	+ d1*inner((u - u0)/dt + grad(um)*um1 + grad(p1), grad(v)*um1)*dx 
	+ d2*div(um)*div(v)*dx

	## Partial integration boundary terms
	- nu*inner(nabla_grad(um)*n, v)*ds2
	+ inner(p1*n, v)*ds3

	## Slip boundary (no skin penetration) penalty term
	+ gamma*inner(dot(um,n)*n, v)*ds1

	## Skin friction (for slip boundary) penalty term
	# object boundary
	+ theta*inner(tau(um), v)*ds(sb)
	+ theta*e*inner(tau(nabla_grad(um)*n), v)*ds(sb)
	# wall boundary
	+ theta*inner(tau(um - uinflow), v)*ds(wb)
	+ theta*e*inner(tau(nabla_grad(um - uinflow)*n), v)*ds(wb)
)

au = lhs(Fu)
Lu = rhs(Fu)

# Continuity variational equation on residual form
Fp = (
	## Navier stokes continuity equation on weak form
	div(um1)*q*dx
	## Stabilization terms
	+ d1*inner((u1 - u0)/k + grad(um1)*um1 + grad(p), grad(q))*dx 
	)
ap = lhs(Fp)
Lp = rhs(Fp)

# Define force measurement

eps = 10**(-4)
psiExp = Expression(("L0 <= x[0] && x[0] <= L1 && H0 <= x[1] && x[1] <= H1 ? 1.0 : 0.0", "0.0"), 
	L0=l0-eps ,L1=l1+eps, H0=h0-eps, H1=h1+eps, element = V.ufl_element())
psi = interpolate(psiExp, V)

force = (
	inner((u1 - u0)/k 
	+ grad(um1)*um1, psi)*dx 
	- p1*div(psi)*dx 
	+ nu*inner(grad(um1), grad(psi))*dx
	)


## ============================================================= ##

# Create XDMF files for visualization output
file_u = File(folder+"pvd/velocity.pvd")
file_p = File(folder+"pvd/pressure.pvd")

# xdmffile_u = XDMFFile(folder+'velocity.xdmf')
# xdmffile_p = XDMFFile(folder+'pressure.xdmf')
# xdmffile_u = XDMFFile('navier_stokes_cylinder/velocity.xdmf')
# xdmffile_p = XDMFFile('navier_stokes_cylinder/pressure.xdmf')

# Time-stepping
t = 0
time0 = time.time()

t_array = []
f_array = []

prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
print(int(T/dt), "iterations total")

with open(folder+"data.txt", "w") as file:

	file.write("Re: " + str(int(round(rho*uin*ymax/mu,0)))+"\n")
	file.write("Drag: \n")
	file.close()

while t < T:
	## Assemble matrices
	Au = assemble(au)
	Ap = assemble(ap) 
	[bc.apply(Au) for bc in bcu]
	[bc.apply(Ap) for bc in bcp]

	for i in range(10):

		for j in range(5):

			# Compute velocity solution
			bu = assemble(Lu)
			[bc.apply(bu) for bc in bcu]
			[bc.apply(u1.vector()) for bc in bcu]
			solve(Au, u1.vector(), bu, "bicgstab", "default")

			# Compute pressure solution 
			bp = assemble(Lp)
			[bc.apply(bp) for bc in bcp]
			[bc.apply(p1.vector()) for bc in bcp]
			solve(Ap, p1.vector(), bp, "bicgstab", prec)

		# Update current time and solution
		t += dt
		u0.assign(u1)

	if(t > T/5):

		# Calculate drag force
		f_ = assemble(force);
		t_array.append(t)
		f_array.append(f_)
		with open(folder+"data.txt", "a") as file:
			file.write(str(t)+", "+str(f_)+"\n")
			file.close()

		# Save solution to file (XDMF/HDF5)
		# xdmffile_u.write(u1, t)
		# xdmffile_p.write(p1, t)
		file_u << u1;
		file_p << p1;

	# Update progress bar
	timeleft = (time.time()-time0)*(1-t/T)/(t/T)
	print(int(t / T*100), "%", round(timeleft,1), "s", end="   \r", file=sys.stderr)


# xdmffile_u.close()
# xdmffile_p.close()

print(str(round(time.time()-time0,1)) + "s")
print("Drag:", np.average(f_array[len(f_array)//2:]), np.std(f_array[len(f_array)//2:]))

plt.figure(figsize=(40,6))
plot(u1)
plt.savefig(folder+"t1.png")
plt.close()



