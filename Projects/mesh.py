from fenics import *
import mshr
import matplotlib.pyplot as plt
import numpy as np


def testMesh(res=32, segments=32):
	l = 3
	h = 0.41;
	h1 = h-0.01
	r = 0.15
	domain = mshr.Rectangle(Point(0,0), Point(l,h))
	obj = mshr.Circle(Point(0.5, h1/2), r, res);
	# obj += mshr.Rectangle(Point(0.5, h1/2-r), Point(0.9+r, h1/2+r))
	# obj += mshr.Circle(Point(0.9+r, h1/2), r, res);

	mesh = mshr.generate_mesh(domain-obj, res)
	# mesh = refineMesh(mesh, 0.5-r, 0.9+2*r, 0.0, h);

	plt.figure()
	plot(mesh)
	plt.show()

	return mesh, (l,h)


def basicDomain(angle, aspect_ratio, resolution):

	L = 22;
	H = 1 / (1-aspect_ratio);
	l = 4

	domain = tunnel(L, H)
	obj = simpleTrain(l, 1.0, np.pi*angle/180)
	obj = mshr.CSGTranslation(obj, Point(3, (H-1)/2))
	mesh = mshr.generate_mesh(domain-obj, resolution)
	mesh = refineMesh(mesh, 2.8, 3+l+0.2, 0, H)
	mesh = refineMesh(mesh, 2.9, 3+l+1.5, 0, H)
	mesh = refineMesh(mesh, L-0.3, L, 0, H)

	return mesh, (L, H);

def simpleTrain(length, width, angle):
	beta = (width/2)*np.tan(angle)
	points = [
		Point(length, 0),
		Point(length+beta, width/2),
		Point(length, width),
		Point(0, width),
		Point(-beta, width/2),
		Point(0,0)
	]
	return mshr.Polygon(points)

def wedge(width, radius, angle, segments):
	a = width/2-radius
	b = a*np.tan(angle)
	c = np.sqrt(a**2+b**2)

	w1 = roundRect(radius, c+2*radius, segments)
	w2 = roundRect(radius, c+2*radius, segments)

	if(radius < 1/4 and angle > 0):
		w1 += mshr.Rectangle(Point(radius,radius), Point(c, -c+radius))
		w2 += mshr.Rectangle(Point(radius,radius), Point(c, c-radius)) 

	w1 = mshr.CSGRotation(w1, Point(radius, radius), np.pi/2-angle)
	w2 = mshr.CSGRotation(w2, Point(radius, radius), -np.pi/2+angle)

	w3 = roundRect(radius, width, segments)
	w3 = mshr.CSGRotation(w3, Point(radius, radius), np.pi/2)
	w3 = mshr.CSGTranslation(w3, Point(b, 0))

	w5 = mshr.Rectangle(Point(b+2*radius,0), Point(b+width, width))

	return mshr.CSGTranslation(w1+w2, Point(0,width/2-radius)) + w3 - w5


def roundRect(radius, length, segments):
	shape = mshr.Circle(Point(radius, radius), radius, segments)
	shape += mshr.Rectangle(Point(radius,0), Point(length-radius, radius*2))
	shape += mshr.Circle(Point(length-radius, radius), radius, segments)

	return shape;


def train(angle, length, width, radius=1/10, segments=20):
	body = mshr.Rectangle(Point(0,0), Point(length, width))
	front = wedge(width, radius, np.pi*angle/180, segments);
	back = mshr.CSGRotation(front, Point(0, width/2), np.pi)

	delta = (width/2-radius)*np.tan(np.pi*angle/180)+radius;
	front = mshr.CSGTranslation(front, Point(-delta, 0))
	back = mshr.CSGTranslation(back, Point(length+delta,0))

	return front + body + back;


def tunnel(length, height):
	return mshr.Rectangle(Point(0,0), Point(length, height))


def refineMesh(mesh, x0, x1, y0, y1):
	cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
	for cell in cells(mesh):
		cell_marker[cell] = False
		p = cell.midpoint()
		if x0 <= p.x() <= x1 and y0 <= p.y() <= y1:
			cell_marker[cell] = True
	return refine(mesh, cell_marker)


def createMesh(length, resolution, angle, aspect_ratio, dx, domain, rear=False, full=True, double=False, radius=1/10, segments=24):
	H = 1 + aspect_ratio;
	L = domain

	if(double):
		H*=2;
	if(full):
		L = length + 3*dx

	domain = tunnel(L, H);
	obj = simpleTrain(length, 1.0, angle)
	# obj = train(angle, length, 1.0, radius=radius, segments=segments)
	if(rear):
		obj = mshr.CSGTranslation(obj, Point(0, aspect_ratio/2))
	else: 
		obj = mshr.CSGTranslation(obj, Point(dx, aspect_ratio/2))

	mesh = mshr.generate_mesh(domain-obj, resolution)
	return mesh, (L, H);



if __name__ == "__main__":

	mesh, (l,h) = createMesh(3, 16, 30, 1/5, 2, 3, full=False)
	mesh = refineMesh(mesh, 1.5, l, 0, h)
	plt.figure()
	plot(mesh)
	plt.show()

	# angle = 0
	# t = wedge(1, 1/5, np.pi*angle/180, 20)
	# mesh = mshr.generate_mesh(t, 16)

	# plt.figure()
	# plot(mesh)
	# plt.show()

	# t = train(30, 3, 1, radius=1/3)

	# for angle in range(0,45,10):
	# 	for r in range(3,10):
	# 		print(angle, r)
	# 		t = wedge(1, 1/r, np.pi*angle/180, 20)
	# 		mesh = mshr.generate_mesh(t, 16)

	# 		plt.figure()
	# 		plot(mesh)
	# 		plt.savefig(str(angle)+":"+str(r)+".png")
	# 		plt.close()