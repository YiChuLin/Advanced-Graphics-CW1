import sys
import numpy as np
from PNM import *

import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Step 4. Design a center weighting function used in Step 4.
def w_function(x):
	return 16*x**2*(1-x)**2

# Implement a tone mapper
def tone_map(F, stops = 0, gamma = 1.0):
	# x should be an hsv image
	# scales with exposure
	x = F.copy()
	x[:,:,2] *= 2**stops
	# Clip
	x[x[:,:,2] > 1, 2] = 1
	x[:,:,2] **= (1/gamma) 
	return x


if '__main__' == __name__:
	num_of_img = 7
	# Step 1. Load all images into a list
	Office = []
	for i in range(num_of_img):
		filename = '../Office/Office'+str(i+1)+'.pfm'
		img_ = loadPFM(filename)
		# Step 1-1. Convert images from RGB to HSV
		img_ = colors.rgb_to_hsv(img_)
		Office.append(img_)

	# Step 2. Create a blank image buffer F
	F = np.zeros(shape=Office[0].shape, dtype=np.float32)
	# Step 3. Eliminate bad values
	for i in range(num_of_img):
		img = Office[i]
		img[img[:,:,2] > 0.92, 2] = 0 #Set value to zero
		img[img[:,:,2] < 0.005, 2] = 0
		Office[i] = img

	# Step 4. Implement a center weighting function
	w = w_function
	# Step 5. Compute F(x,y)
	# Step 5-1. Compute w(Z(x,y)) and the sum of weights, update Z(x,y)
	sum_w = np.zeros(shape=(F.shape[0], F.shape[1], 1), dtype = np.float32)
	for i, img in enumerate(Office):
		weight = np.expand_dims(w(img[:,:,2]), axis = 2)
		img[:,:,2] /= 4**(i)
		F += np.log(img + 1e-10)*weight #plus 1e-10 for numerical stability
		sum_w += weight

	# Step 5-2. Divide by sum of weights(normalize) and take exponential
	F = np.exp(F/(sum_w+1e-10))
	# Save as pfm
	F_rgb = colors.hsv_to_rgb(F)
	writePFM('part1img/output_results.pfm',F_rgb)
	# print out the dynamic range
	print(F[:,:,2].max())
	print(F[:,:,2].min())
	print('max brightness - min brightness '+str(F[:,:,2].max()-F[:,:,2].min()))
	# Here we visualize the HDR image with various tone mappers
	save_params = [(5,1), (6,1), (7,1),(8,1),(9,1),(10,1),(11,1),(12,1),(5,3.0),(6,3.0),(7,2.5),(7,3.0),(8,2.2),(8,2.5)]
	stops = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
	for s in stops:
		# Here we assume gamma = 1
		g = tone_map(F, stops = s)
		g_rgb = colors.hsv_to_rgb(g)
		g_rgb = (g_rgb*255.0).astype('uint8')
		plt.imshow(g_rgb)
		plt.title('+'+str(s)+' stops, gamma = 1')
		plt.show()
		if (s, 1) in save_params:
			filename = 'part1img/stops_'+str(s)+'_gamma_1.ppm'
			writePPM(filename, g_rgb)
	stops = [5,6,7,8,9]
	gammas = [1.5, 1.8, 2.2, 2.5, 3.0]
	for s in stops:
		for gamma in gammas:
			g = tone_map(F, stops = s, gamma = gamma)
			g_rgb = colors.hsv_to_rgb(g)
			g_rgb = (g_rgb*255.0).astype('uint8')
			plt.imshow(g_rgb)
			plt.title('+'+str(s)+' stops, gamma = '+str(gamma))
			plt.show()
			if (s, gamma) in save_params:
				filename = 'part1img/stops_'+str(s)+'_gamma_'+str(gamma)+'.ppm'
				writePPM(filename, g_rgb)
