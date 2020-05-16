from fenics import *
import mesh as msh
import numpy as np
import matplotlib.pyplot as plt
import time
import sys



def buildMesh(headway, length, aspect_ratio, angle, radius=1/10, rear=False, full=False, double=False, resolution=8):

	h = 1
	if(double):
		h = 2

	mesh, (l,h) = msh.createMesh(length, resolution*(length+headway), 
		angle, aspect_ratio, headway, length+headway, rear=rear, full=full, 
		double=double, radius=radius, segments=int(24*5*radius))
	
	x0 = headway*0.9
	x1 = l
	if(full):
		x1 = l-headway*0.9
	y1 = h

	if(double):
		y1 = h/2

	if(rear):
		x0 = 0
		x1 = length*2

	mesh = msh.refineMesh(mesh, x0, x1, 0, y1)
	# mesh = msh.refineMesh(mesh, headway, l, 0, ys)

	plt.figure()
	plot(mesh)
	plt.show()

	return mesh, (l,h)




T = 5.0 			# final time
num_steps = 5000   	# number of time steps
dt = T / num_steps 	# time step size
mu = 10**(-5)		# dynamic viscosity
rho = 1.4			# density
uin = 1.0			# inflow velocity

aspect_ratio = 1/10
headway = 1
length = 0.5
angle = 10
double = False
full = False
rear = False


if(len(sys.argv) > 1):
	aspect_ratio = 1/int(sys.argv[1])
if(len(sys.argv) > 2):
	angle = int(sys.argv[2])
if(len(sys.argv) > 3):
	if("double" in sys.argv[3]):
		double = True
if(len(sys.argv) > 4):
	if("full" in sys.argv[4]):
		full = True
	if("rear" in sys.argv[4]):
		rear = True

print("Re: ", int(round(rho*uin*0.1/mu,0)))
print("Aspect ratio:", round(aspect_ratio, 4))
print("Angle:", angle)
print("Double track:", double, ", Full length:", full, "Rear view:", rear)

print("Aspect ratio:", round(aspect_ratio, 4), "Angle:", angle, file=sys.stderr)

if(full):
	length = 5.0
	# headway = 2.0

if(double):
	headway = 2.0

# mesh, (xmax, ymax) = buildMesh(headway, length, aspect_ratio, angle, rear=rear,
# 	full=full, double=double, resolution=24)

mesh, (xmax, ymax) = msh.testMesh(res=32);
print("Mesh size:", mesh.num_cells())

# Define function spaces
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define boundaries
inflowBoundary = CompiledSubDomain("on_boundary && near(x[0], 0)")
outflowBoundary = CompiledSubDomain("on_boundary && near(x[0], "+str(xmax)+")")
slipBoundary = CompiledSubDomain("on_boundary && (near(x[1], 0) || near(x[1], "+str(ymax)+") || !(near(x[0], 0) || near(x[0], "+str(xmax)+")))")

uinflow = Expression(("uin*6*x[1]*(YMAX-x[1])/(YMAX*YMAX)", "0.0"), element=V.ufl_element(), YMAX=ymax, uin=uin);
# uinflow = Expression(("t < uin ? t : uin", "0.0"), element=V.ufl_element(), uin=uin, t=0.0)

# Define Dirichlet boundary conditions
# bcu_inflow = DirichletBC(V, Constant((uin, 0.0)), inflowBoundary)
bcu_inflow = DirichletBC(V, uinflow, inflowBoundary)
# bcu_walls = DirichletBC(V, Constant((0.0, 0.0)), slipBoundary)
# bcp_inflow = DirichletBC(Q, Constant(1.0), inflowBoundary)
bcp_outflow = DirichletBC(Q, Constant(0), outflowBoundary)
bcu = [bcu_inflow] #, bcu_walls]
bcp = [bcp_outflow]

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
 
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
inflowBoundary.mark(boundaries, ib)
outflowBoundary.mark(boundaries, ob)
slipBoundary.mark(boundaries, sb)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

## ============================================================= ##
## Define variational problem

# Define expressions used in variational forms
dt = 0.1*mesh.hmin()
f  = Constant((0, 0))
k  = Constant(dt)
nu = Constant(mu/rho);
h = CellDiameter(mesh);
n  = FacetNormal(mesh)
u_mag = sqrt(dot(u1,u1))
d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
d2 = h*u_mag
gamma = 100/h

# Mean velocities for trapozoidal time stepping
um = 0.5*(u + u0)
um1 = 0.5*(u1 + u0)

# Momentum variational equation on residual form
Fu = (
	inner((u - u0)/dt + grad(um)*um1, v)*dx 
	+ nu*inner(grad(um), grad(v))*dx
	- p1*div(v)*dx
	## Source term
	# - inner(f, v);
	## Stabilization terms
	+ d1*inner((u - u0)/dt + grad(um)*um1 + grad(p1), grad(v)*um1)*dx 
	+ d2*div(um)*div(v)*dx
	## Partial integration boundary term
	- nu*inner(nabla_grad(um)*n, v)*ds(sb)
	+ inner(p1*n, v)*ds(sb) 
	## Slip boundary (no skin penetration) penalty term
	+ gamma*inner(dot(um,n)*n, v)*ds(sb)
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
psiExp = Expression(("near(x[1], 0) || near(x[1], H) ? 0.0 : 1.0", "0.0"), H=ymax, element = V.ufl_element())
psi = interpolate(psiExp, V)
force = (
	inner((u1 - u0)/k 
	+ grad(um1)*um1, psi)*ds(sb) 
	- p1*div(psi)*ds(sb) 
	+ nu*inner(grad(um1), grad(psi))*ds(sb)
	)

## ============================================================= ##
## Assemble matrices
Au = assemble(au)
Ap = assemble(ap) 

[bc.apply(Au) for bc in bcu]
[bc.apply(Ap) for bc in bcp]

# Create XDMF files for visualization output
# xdmffile_u = XDMFFile('navier_stokes_cylinder/{}/{}/velocity.xdmf'.format(angle, aspect_ratio))
# xdmffile_p = XDMFFile('navier_stokes_cylinder/{}/{}/pressure.xdmf'.format(angle, aspect_ratio))
xdmffile_u = XDMFFile('navier_stokes_cylinder/velocity.xdmf')
xdmffile_p = XDMFFile('navier_stokes_cylinder/pressure.xdmf')

# Time-stepping
t = 0
time0 = time.time()

t_array = []
f_array = []

prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
print(int(T/dt), "iterations total")

while t < T:
	for i in range(5):
		for j in range(5):
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

		# Update current time
		t += dt
		u0.assign(u1)

	if(t > T/5):
		t_array.append(t)
		f_array.append(assemble(force))

	# Save solution to file (XDMF/HDF5)
	xdmffile_u.write(u1, t)
	xdmffile_p.write(p1, t)

	# Update progress bar
	timeleft = (time.time()-time0)*(1-t/T)/(t/T)
	# sys.err.write(str(int(t / T*100)) + "%" + str(round(timeleft,1))+ "s    \r");
	print(int(t / T*100), "%", round(timeleft,1), "s", end="   \r", file=sys.stderr)
	
# print()
print(str(round(time.time()-time0,1)) + "s")
print("Drag:", np.average(f_array[len(f_array)//2:]), np.std(f_array[len(f_array)//2:]))

plt.figure(figsize=(8,6))
plot(u1)
plt.show()
# plt.savefig(str(aspect_ratio)+":"+str(angle)+".png")
plt.close()

