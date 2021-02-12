from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def makepaths(res, ang, asp, root="data"):
	if(type(res) != list):
		res = [res]
	if(type(ang) != list):
		ang = [ang]
	if(type(asp) != list):
		asp = [asp];
		
	paths = [];
	for q in res:
		for a in ang:
			for r in asp:
				path = root+"/res"+str(q)+"/ang"+str(a)+"/asp"+str(r)+"/data.txt";
				paths.append(path)
	return paths;

def getFile(path):
	with open(path) as file:
		lines = file.readlines()[10:]
		file.close()
	
	time = []
	drag = [];

	for line in lines:
		t = float(line.split(", ")[0].strip())
		d = float(line.split(", ")[1].strip())
		time.append(t)
		drag.append(d)

	# l = len(time)
	return np.array(time), np.array(drag)


def readfile(path):
	
	time, drag = getFile(path)
	l = len(time)
	time = time[l//2:]
	drag = drag[l//2:]

	dv = np.sqrt(np.mean(np.power(drag, 2)))
	de = np.sqrt(np.mean(np.power(np.abs(drag)-dv,2)))

	return dv, de


def plotRes(res):
	paths = makepaths(res, 10, 0.333)
	drag = []
	ddrag = []

	for path in paths:
		dv, de = readfile(path)
		drag.append(dv)
		ddrag.append(de)

	plt.figure(figsize=(6.4, 4))
	plt.errorbar(res, np.divide(drag,drag[0]), yerr=np.divide(ddrag, drag[0]), 
		fmt=".-", capsize=3, capthick=1)
	plt.xlabel("Mesh diagonal resolution")
	plt.ylabel("Relative drag force")
	plt.yscale("log")
	plt.title("Mesh resolution")
	plt.show()

def plotdouble(r0, ang, asp):
	plt.figure()
	drag = []
	ddrag = []
	paths = makepaths(r0, ang, asp, root="double")
	for path in paths:
		dv, de = readfile(path)
		drag.append(dv)
		ddrag.append(de)

	plt.errorbar(ang, np.divide(drag,drag[-1]), yerr=np.divide(ddrag, drag[-1]), 
		fmt=".-", capsize=3, capthick=1)

	# plt.ylim((round(min(drag)/drag[-1],1)-0.1, round(max(drag)/drag[-1],1)+0.1))
	plt.title("Relative drag force in double track tunnel")
	plt.xlabel("Angle [degrees]")
	plt.ylabel("Relative drag force [a.u.]")
	# plt.ylim(0,2)
	plt.show()

def plotdrag(r0, ang, asp):
	plt.figure()
	for a in ang:
		paths = makepaths(r0, a, asp)
		for path in paths:
			t, d = getFile(path)
			l = len(d)
			# t = t[l//2:]
			# d = d[l//2:]
			dv = np.sqrt(np.mean(np.power(d[l//2:], 2)))
			# de = np.sqrt(np.mean(np.power(np.abs(dragd[l//2:])-dv,2)))
			label = str(a) + " deg"
			if(a is "_round"):
				label = "Round"
			bl, = plt.plot(t,-d, '-', label=label)
			plt.plot([t[0],t[-1]], [dv,dv], '--', color=bl.get_color(), label=label+" average")

	plt.title("Variation of drag force over time")
	plt.xlabel("Time [s]")
	plt.ylabel("Relative drag force [a.u.]")
	plt.legend()
	plt.grid()
	plt.show()


def plotDragVary(r0, ang, asp):
	base_line = -1;
	plt.figure()
	for a in ang:
		drag = []
		ddrag = []
		paths = makepaths(r0, a, asp)
		for path in paths:
			dv, de = readfile(path)
			drag.append(dv)
			ddrag.append(de)

		if(base_line == -1):
			base_line = min(drag);

		label=str(a)+" deg"
		if(a == "_round"):
			label="Round"
		plt.errorbar(1/np.subtract(1,asp), np.divide(drag,base_line), yerr=np.divide(ddrag,base_line), fmt=".-",
			label=label, capsize=3, capthick=1)

	fitfunc, params = fitCurve(r0, ang, asp)
	print(params)
	args = np.linspace(1.1,2,50)
	plt.plot(args, np.divide(fitfunc(args),base_line), '--', label=r"Best fit $a/(x-1)^p+c$")

	plt.legend()
	plt.grid()
	plt.title("Relative drag force in single track tunnel")
	plt.xlabel("Aspect ratio")
	plt.ylabel("Relative drag force [a.u.]")
	plt.yscale("log")
	plt.show()


def fitCurve(r0, ang, asp):
	base_line = -1;
	dragg = [0 for a in asp]
	div = [0 for a in asp]
	
	for a in ang:
		drag = []
		ddrag = []
		paths = makepaths(r0, a, asp)
		for path in paths:
			dv, de = readfile(path)
			drag.append(dv)
			ddrag.append(de)

		if(base_line == -1):
			base_line = min(drag);

		for i in range(len(drag)):
			if(not np.isnan(drag[i])):
				dragg[i] += drag[i]
				div[i] += 1

	dragg = np.divide(dragg, div)
	aa = 1/np.subtract(1,asp)
	f = lambda x, a, p, y0: a/np.power(x-1, p)+y0
	popt, pcov = curve_fit(f, aa, dragg, p0=[1,2,1])
	return lambda x: f(x, popt[0], popt[1], popt[2]), popt


def costCompare(r0, ang, asp):
	f, params = fitCurve(r0, ang, asp)
	asp = np.linspace(1.1,2,50)

	fracs = [0.001, 0.01, 0.1]
	c = f(asp)
	c = c/c[-1];
	plt.figure()
	plt.subplot(2,1,1)
	plt.title("Projection of increase in operating costs with increased drag")
	for frac in fracs:
		c0 = c*frac
		plt.plot(asp, 1-frac+c0, label="Cost fraction at 2:1: "+str(frac))
	plt.ylabel("Increase in operating cost")
	plt.yscale("log")
	plt.grid()
	plt.legend()

	plt.subplot(2,1,2)
	for frac in fracs:
		c0 = c*frac
		plt.plot(asp, c0/(1-frac+c0), label="Cost fraction at 2:1: "+str(frac))

	plt.xlabel("Aspect ratio")
	plt.ylabel("Energy cost as fraction of total cost")
	plt.grid()
	plt.ylim((0,1))
	plt.legend()
	plt.show()



if __name__ == "__main__":

	r0 = 100

	res = [32, 48, 64, 100, 140, 192, 230]
	ang = ["_round", 0, 10, 20, 30, 40, 60]
	asp = [0.1, 0.143, 0.2, 0.333, 0.5]
	# asp = [0.143, 0.2, 0.333, 0.5]

	# fitCurve(r0, ang, asp);

	plotRes(res)
	plotDragVary(r0, ang, asp)
	costCompare(r0, ang, asp)
	plotdouble(120, [0,10,20,30,40,60], 0.5)
	plotdrag(180, [0,40,"_round"], 0.333)

