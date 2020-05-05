import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sb
import struct
import sys

if len(sys.argv) < 4:
	print("analyze [filename 1] [filename 2] [number of frame]")
	exit()

frame = int(sys.argv[3])
file1 = sys.argv[1]
file2 = sys.argv[2]

fin = open(file1, "rb")
fout = open(file2, "rb")

fin_magick = int(struct.unpack('i', fin.read(4))[0]) 
fin_width = int(struct.unpack('i', fin.read(4))[0])
fin_height = int(struct.unpack('i', fin.read(4))[0])
fin_lines = int(struct.unpack('i', fin.read(4))[0])

fout_magick = int(struct.unpack('i', fout.read(4))[0])
fout_width = int(struct.unpack('i', fout.read(4))[0])
fout_height = int(struct.unpack('i', fout.read(4))[0])
fout_lines = int(struct.unpack('i', fout.read(4))[0])

if fin_width != fout_width and fin_height != fout_height:
	print(fin_width, fin_height, fout_height, fout_width)
	exit()

for it in range(frame):
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

plt.title("FERMER POUR QUITTER".format(it))
plt.show()

fin.close()
fout.close()