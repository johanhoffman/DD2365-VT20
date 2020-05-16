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
mu = 10**(-6)		# dynamic viscosity
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

print()
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

mesh, (xmax, ymax) = msh.testMesh(res=48);
print("Mesh size:", mesh.num_cells())

# Define function spaces
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define boundaries
inflowBoundary = CompiledSubDomain("on_boundary && near(x[0], 0)")
outflowBoundary = CompiledSubDomain("on_boundary && near(x[0], "+str(xmax)+")")
slipBoundary = CompiledSubDomain("on_boundary && !(near(x[0], 0) || near(x[0], "+str(xmax)+"))")

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

# Define expressions used in variational forms

dt = 0.5*mesh.hmin()
U  = 0.5*(u0+u)
um = 0.5*(u0+u1)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Mark boundaries for integration.
ib = 1
ob = 2
sb = 3
 
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
inflowBoundary.mark(boundaries, ib)
outflowBoundary.mark(boundaries, ob)
slipBoundary.mark(boundaries, sb)

ds = Measure("ds", subdomain_data=boundaries)

beta = 100
h = CellDiameter(mesh)
gamma = beta/h;
e = 1/h;
f = 1/10;

# u_mag = sqrt(dot(u1,u1))
# d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
# d2 = h*u_mag

# Define symmetric gradient
def epsilon(u):
	return (nabla_grad(u) + nabla_grad(u).T)/2

# Define stress tensor
def sigma(u, p):
	return 2*mu*epsilon(u) - p*Identity(len(u))

def tau(u, n):
	return u - dot(u, n)*n;


## Formulate the variational problem in Incremental pressure correction scheme (IPCS)
## Good description/derivation here:
## http://www.diva-portal.org/smash/get/diva2:1242050/FULLTEXT01.pdf

# Define variational problem for step 1 (Tentative velocity solution)
F1 = (
	rho*dot((u - u0) / k, v)*dx
	+ rho*dot(dot(u0, nabla_grad(u0)), v)*dx
	# Stress tensor
	+ inner(sigma(U, p0), epsilon(v))*dx
	# + mu*inner(grad(u), grad(v))*dx
	# Source term
	# - dot(f, v)*dx
	# Partial integration boundary term
	+ dot(p0*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds
	# No skin penetration boundary condition (slip boundary)
	+ gamma*inner(dot(u, n)*n, v)*ds(sb)
	# Skin friction for slip boundary
	# + e*inner(tau(u,n), v)*ds(sb)
	# + e*f*inner(tau(dot(epsilon(u),n),n), v)*ds(sb)
	)

# F1 = rho*inner((u-u0)/k, v)*dx + inner(mu*grad(u), grad(v))*dx \
# 	+ rho*inner(dot(u0,nabla_grad(u0)), v)*dx + inner(nabla_grad(p0), v)*dx \
# 	+ inner(mu*dot(nabla_grad(u) + nabla_grad(u).T, n) - p0*n, v)*ds(sb) \
# 	+ inner(f, v)*dx \
# 	+ gamma*inner(dot(u, n), dot(v, n))*ds(sb)

a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2 (Pressure update)
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p0), nabla_grad(q))*dx - (1/k)*div(um)*q*dx

# Define variational problem for step 3 (Velocity correction)
a3 = dot(u, v)*dx + gamma*inner(dot(u, n), dot(v, n))*ds(sb)
L3 = dot(u0, v)*dx - k*dot(nabla_grad(p1 - p0), v)*dx

# Define force measurement
psiExp = Expression(("near(x[1], 0) || near(x[1], H) ? 0.0 : 1.0", "0.0"), H=ymax, element = V.ufl_element())
psi = interpolate(psiExp, V)
force = inner((u1 - u0)/k + grad(um)*um, psi)*ds(sb) - p1*div(psi)*ds(sb) + mu/rho*inner(grad(um), grad(psi))*ds(sb)


# Assemble matrices
# A1 = assemble(a1)
# A2 = assemble(a2)
# A3 = assemble(a3)

# Apply boundary conditions to matrices
# [bc.apply(A1) for bc in bcu]
# [bc.apply(A2) for bc in bcp]

# Create XDMF files for visualization output
# xdmffile_u = XDMFFile('navier_stokes_cylinder/{}/{}/velocity.xdmf'.format(angle, aspect_ratio))
# xdmffile_p = XDMFFile('navier_stokes_cylinder/{}/{}/pressure.xdmf'.format(angle, aspect_ratio))
xdmffile_u = XDMFFile('navier_stokes_cylinder/velocity.xdmf')
xdmffile_p = XDMFFile('navier_stokes_cylinder/pressure.xdmf')

# Time-stepping
t = 0
time0 = time.time()
n = 0

t_array = []
f_array = []

print(int(T/dt), "iterations total")
while t < T:

	for i in range(10):
		# Update current time
		# uinflow.t = t;
		t += dt

		try:
			# Step 1: Tentative velocity step
			A1 = assemble(a1)
			b1 = assemble(L1)
			[bc.apply(A1) for bc in bcu]
			[bc.apply(b1) for bc in bcu]
			solve(A1, u1.vector(), b1, 'bicgstab', 'hypre_amg')

			# Step 2: Pressure correction step
			A2 = assemble(a2)
			b2 = assemble(L2)
			[bc.apply(A2) for bc in bcp]
			[bc.apply(b2) for bc in bcp]
			solve(A2, p1.vector(), b2, 'bicgstab', 'hypre_amg')

			# Step 3: Velocity correction step
			A3 = assemble(a3)
			b3 = assemble(L3)
			[bc.apply(A3) for bc in bcu]
			[bc.apply(b3) for bc in bcu]
			solve(A3, u1.vector(), b3, 'cg', 'sor')

			# Update previous solution
			u0.assign(u1)
			p0.assign(p1)

		except RuntimeError:
			plt.figure()
			plot(mesh)
			plot(p0)
			plot(u0)
			plt.show()
			plt.close()

			print("Divergent solution")

			raise

		n+=1

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

