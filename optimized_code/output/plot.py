import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sb
import struct
from os import sys

fin = open("resultat.raw", "rb")
fout = open("mire.raw", "rb")

fin_width = int(struct.unpack('i', fin.read(4))[0])
fin_height = int(struct.unpack('i', fin.read(4))[0])
fin_lines = int(struct.unpack('i', fin.read(4))[0])

fout_width = int(struct.unpack('i', fout.read(4))[0])
fout_height = int(struct.unpack('i', fout.read(4))[0])
fout_lines = int(struct.unpack('i', fout.read(4))[0])

if fin_width != fout_width and fin_height != fout_height:
	print(fin_width, fin_height, fout_height, fout_width)
	exit()

for it in range(320):
	array1 = np.zeros((fin_height, fin_width))
	array2 = np.zeros((fin_height, fin_width))

	y_step = int(fin_height/fin_lines)
	for l in range(fin_lines):
		for i in range(fin_width):
			for j in range(y_step):
				array1[j+y_step*l,i] = abs((struct.unpack('f', fin.read(4))[0] + struct.unpack('f', fin.read(4))[0]))
	
	y_step = int(fout_height/fout_lines)
	for l in range(fout_lines):
		for i in range(fout_width):
			for j in range(y_step):
				array2[j+y_step*l,i] = abs((struct.unpack('f', fout.read(4))[0] + struct.unpack('f', fout.read(4))[0]))

	delta = np.abs(array2 - array1)
	plt.clf()
	plt.title("frame n°{0}".format(it))
	sb.heatmap(array1, cmap="YlOrRd")
	plt.pause(0.00001)

fin.close()
fout.close()