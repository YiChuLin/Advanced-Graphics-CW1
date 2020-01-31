import sys
import numpy as np
from PNM import *

import matplotlib.colors as colors

# Gamma function
def gamma(F, gamma = 1.0):
	# x should be an hsv image
	# scales with exposure
	x = colors.rgb_to_hsv(F)
	x[:,:,2] **= (1/gamma)
	x = colors.hsv_to_rgb(x)
	return x

if '__main__' == __name__:
	# Create a blank image of shape 511x511x3
	img = np.zeros(shape=(511, 511, 3), dtype=np.float32)
	# Create blank holders for n image and r image
	n_img = -np.ones(shape=(511, 511, 3), dtype = np.float32)
	r_img = -np.ones(shape=(511, 511, 3), dtype = np.float32)
	# Load the lat long map
	map_ =  loadPFM('../UrbanProbe/urbanEM_latlong.pfm')
	# Define the center of image, radius of the mirror ball, and the veiw vector
        center_pixel = (511-1)/2
	radius = 511.0/2
	v = np.array([0,0,1])


	map_hx = map_.shape[1]
	map_hy = map_.shape[0]
	for i in range(511):
		for j in range(511):
			# check if the current pixel lies within the circle
			if np.sqrt((i-center_pixel)**2+(j-center_pixel)**2) < radius:
				#img[i,j,:] = np.array([1,1,1]) #for checking purpose
				#define the normal vector for this pixel
				y = (center_pixel - i)/radius; x = -(center_pixel-j)/radius
				z_comp = np.sqrt(1-(x**2+y**2))
		        n = np.array([x, y, z_comp])
				# calculate r given n and v
				r = 2*np.dot(n,v)*n-v
				# store n and r in the blank images
				n_img[i,j,:] = n
				r_img[i,j,:] = r
				# now find the index of the latlong img
				theta = np.arccos(r[1])
				phi = np.arctan2(r[0],r[2]) + np.pi
				index_1 = min(int(phi/(2*np.pi)*map_hx), map_.shape[1]-1) #clip the value
				index_0 = min(int(theta/np.pi*map_hy), map_.shape[0]-1)
				pixel_val = map_[index_0, index_1,:]
				pixel_val[pixel_val > 1] = 1 #clip the value
				img[i,j,:] = pixel_val
	# write out the n images and the r images
	writePFM('part2img/n_image.pfm',n_img)
	writePFM('part2img/r_image.pfm',r_img)
	writePPM('part2img/n_image.ppm',((n_img+1)*127.5).astype('uint8')) ##but background is not black
	writePPM('part2img/r_image.ppm',((r_img+1)*127.5).astype('uint8'))
	# write out the mirror ball image
	writePFM('part2img/mirror_ball.pfm',img)
	writePPM('part2img/mirror_ball.ppm',(img*225).astype('uint8'))
	# apply different value of gamma
	gammas = [1.5, 1.8, 2.2, 2.5, 3.0]
	for g in gammas:
		img_gamma = gamma(img, gamma = g)
		filename = 'part2img/mirror_ball_gamma'+str(g)
		writePPM(filename+'.ppm', (img_gamma*255).astype('uint8'))
