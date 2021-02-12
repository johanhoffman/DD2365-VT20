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


def basicDomain(angle, aspect_ratio, resolution, double=False, length=22, round=False, rounded=False):

	m = 1
	if(double):
		m=2

	L = length;
	H = 1 / (1-aspect_ratio);
	l = 4*m

	h0 = (H-1)/2
	h1 = h0+1
	domain = tunnel(L, H*m)

	if(round):
		obj = roundTrain(l, 1.0)
		l0 = 3-0.5
		l1 = 3+l+0.5
	else:
		if(rounded):
			segments = int(360/(90-angle) * 8)
			obj = roundedTrain(l, 1.0, np.pi*angle/180, segments=segments)
		else:
			obj = simpleTrain(l, 1.0, np.pi*angle/180)
		b = (1/2)*np.tan(np.pi*angle/180)
		l0 = 3-b
		l1 = 3+l+b

	obj = mshr.CSGTranslation(obj, Point(3, h0))

	mesh = mshr.generate_mesh(domain-obj, resolution)
	mesh = refineMesh(mesh, l0, l1, 0, H)
	mesh = refineMesh(mesh, 2.9, 3+l+1.5, 0, H)
	mesh = refineMesh(mesh, L-0.2, L, 0, H*m)

	return mesh, (L, H), (l0,l1,h0,h1);

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

def roundedTrain(length, width, angle, radius=1/10, segments=32):
	dx = radius*np.sin(np.pi/2-angle)
	dy = radius*(1-np.cos(np.pi/2-angle))

	s1 = simpleTrain(length-2*dx, width, 0)
	s1 = mshr.CSGTranslation(s1, Point(dx, 0))
	s2 = simpleTrain(length, width-2*dy, angle)
	s2 = mshr.CSGTranslation(s2, Point(0, dy))

	obj = s1+s2
	for i in range(2):
		for j in range(2):
			obj += mshr.Circle(Point(dx+i*(length-2*dx), radius+j*(width-2*radius)), radius, segments)

	return obj

def testDomain(angle):
	# t = tunnel(10,2)
	segments = 360//angle * 8
	v = roundedTrain(5, 1, np.pi*angle/180, segments=segments)

	return mshr.generate_mesh(v, 32)


def roundTrain(length, width, segments=32):
	rect = mshr.Rectangle(Point(0,0), Point(length, width))
	c1 = mshr.Circle(Point(0,width/2),width/2, segments)
	c2 = mshr.Circle(Point(length, width/2), width/2, segments)

	return rect + c1 + c2;


def tunnel(length, height):
	return mshr.Rectangle(Point(0,0), Point(length,height))


def refineMesh(mesh, x0, x1, y0, y1):
	cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
	for cell in cells(mesh):
		cell_marker[cell] = False
		p = cell.midpoint()
		if x0 <= p.x() <= x1 and y0 <= p.y() <= y1:
			cell_marker[cell] = True
	return refine(mesh, cell_marker)


if __name__ == "__main__":

	# mesh, (l,h) = basicDomain(10, 1/5, 32)
	mesh = testDomain(20);
	# mesh = refineMesh(mesh, 1.5, l, 0, h)
	plt.figure()
	plot(mesh)
	plt.show()

