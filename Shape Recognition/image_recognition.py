# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import imutils

def image_recognition(file_path = None, camera_port_number = None):
	
	class ColorLabeler:
		def __init__(self):
			# initialize the colors dictionary, containing the color
			# name as the key and the RGB tuple as the value
			colors = OrderedDict({
				"red": (195, 92, 109),
				"green": (45, 131, 85),
				"blue": (79, 106, 156),})
			# allocate memory for the L*a*b* image, then initialize
			# the color names list
			self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
			self.colorNames = []
			# loop over the colors dictionary
			for (i, (name, rgb)) in enumerate(colors.items()):
				# update the L*a*b* array and the color names list
				self.lab[i] = rgb
				self.colorNames.append(name)
			# convert the L*a*b* array from the RGB color space
			# to L*a*b*
			self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

		def label(self, image, c):
			# construct a mask for the contour, then compute the
			# average L*a*b* value for the masked region
			mask = np.zeros(image.shape[:2], dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1) # constructs a mask
			mask = cv2.erode(mask, None, iterations=2)
			mean = cv2.mean(image, mask=mask)[:3] # the average L*a*b* value
			# initialize the minimum distance found thus far
			minDist = (np.inf, None)
			# loop over the known L*a*b* color values
			for (i, row) in enumerate(self.lab):
				# compute the distance between the current L*a*b*
				# color value and the mean of the image
				d = dist.euclidean(row[0], mean)
				# if the distance is smaller than the current distance,
				# then update the bookkeeping variable
				if d < minDist[0]:
					minDist = (d, i)
			# return the name of the color with the smallest distance
			return self.colorNames[minDist[1]]


	class ShapeDetector:
		def __init__(self):
			pass
		def detect(self, c):
			# initialize the shape name and approximate the contour
			shape = "unidentified"
			peri = cv2.arcLength(c, True) # compute perimeter of the contour
			approx = cv2.approxPolyDP(c, 0.04 * peri, True) # constructs contour approximation
			
			# if the shape is a triangle, it will have 3 vertices
			if len(approx) == 3:
				shape = "triangle"
			# if the shape has 4 vertices, it is either a square or
			# a rectangle
			elif len(approx) == 4:
				# compute the bounding box of the contour and use the
				# bounding box to compute the aspect ratio
				(x, y, w, h) = cv2.boundingRect(approx)
				ar = w / float(h)
				# a square will have an aspect ratio that is approximately
				# equal to one, otherwise, the shape is a rectangle
				if ar >= 0.90 and ar <= 1.10:
					shape = "square"

			# otherwise, we assume the shape is a circle or not a shape
			else:
				# compute the bounding box of the contour and use the
				# bounding box to compute the aspect ratio
				(x, y, w, h) = cv2.boundingRect(approx)
				ar = w / float(h)
				# a square will have an aspect ratio that is approximately
				# equal to one, otherwise, the shape is a rectangle
				if ar >= 0.90 and ar <= 1.10:
					shape = "circle"
				
			# return the name of the shape
			return shape

	if file_path != None:
		image = cv2.imread(file_path) # read the image from a file
	elif camera_port_number != None:
		camera = cv2.VideoCapture(camera_port_number) # access camera
		_, image = camera.read() # get image from the camera
	
	# resize the image to a smaller factor so the shapes can be approximated better
	resized = imutils.resize(image, width=300)

	# blurs the image to remove high frequency noise.
	blurred = cv2.GaussianBlur(resized, (5, 5), 0)

	# The program will perform matrix multiplication between these matrices and the image (whichi is a matrix.)
	
	# For each of the RGB channels, emphasise that respective channel and remove colour contribution from any other colour.
	# This removes everything from the image except for fully red, green and blue regions.
	# I use relatively larger values so that the filtered image would always be very bright.
	# This is because the shapes are always red, green or blue.
	pure_rgb_transform = np.array([
        [20, -12, -12],
        [-12, 20, -12],
        [-12, -12, 20]
    ])

	# Works in reverse to the matrix above. Its purpose is to exaggerate white regions in the image and to make them unifromly coloured.
	# This is because the shapes always have a white background.
	white_transform = np.array([
        [-20, 12, 12],
        [12, -20, 12],
        [12, 12, -20]
    ])

    # The image will be in BGR order, we want RGB order for calculations.
	img_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

	# blurs the image again, but I'm not sure why
	img_rgb = cv2.GaussianBlur(img_rgb, (7, 7), 0)

    # This does three things:
    # - Transforms the pixels according to the transform matrix
    # - Rounds the pixel values to integers
    # - Coverts the datatype of the matrix to 'uint8' show .imshow() works
	# This isolates regions that are close to or are purely red, green or blue.
	img_pure_rgb = np.rint(img_rgb.dot(pure_rgb_transform.T).clip(min=0, max=255)).astype('uint8')

	# This exaggerates white regions in the image.
	img_white_trans = np.rint(img_rgb.dot(white_transform.T).clip(min=0, max=255)).astype('uint8')

	# Removes everything from the image that is not very close to white. The result is monocolour.
	img_white = cv2.threshold(cv2.cvtColor(img_white_trans, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY)[1]

	# Singles out the regions from just now that were identified as very white.
	white_cnts = cv2.findContours(img_white.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	white_cnts = imutils.grab_contours(white_cnts)

	# Cut out everything from the red, green and blue filtered image that is outside of the white region of the original image, by drawing a mask from what we got from above.
	mask = np.zeros(blurred.shape, np.uint8)
	white_cnts_drawn = cv2.drawContours(mask, white_cnts, -1, (255,255,255), -1)
	white_filtered_img = cv2.bitwise_and(img_pure_rgb, img_pure_rgb, mask=cv2.cvtColor(white_cnts_drawn, cv2.COLOR_RGB2GRAY))

	# blur the resized image slightly, then convert it to both
	# grayscale and the L*a*b* color spaces
	gray = cv2.cvtColor(white_filtered_img, cv2.COLOR_BGR2GRAY)
	lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
	thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]

	# find contours in the thresholded image. Contours are the outlines of individual shapes that the script picks up
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# initialize the shape detector and color labeler
	sd = ShapeDetector()
	cl = ColorLabeler()

	max_area = 0
	c = None
	# loop over the contours. We want a contour that outlines the largest shape that is valid.
	for curr in cnts:
		# detects the shape of each contour.
		shape = sd.detect(curr)

		# checks if the contour outlines a valid shape and is the largst.
		if(cv2.contourArea(curr) > max_area and shape != "unidentified"):
			max_area = cv2.contourArea(curr)

			# this contour becomes what we will be analysing.
			c = curr
	
	"""
	cv2.imshow("white_filtered_img", white_filtered_img)
	cv2.waitKey()
	cv2.imshow("blurred", blurred)
	cv2.waitKey()
	"""

	# detect the shape of the contour and label the color
	try:
		shape = sd.detect(c)
	except:
		return 0  # return 0 if it cannot detect shape

	if cv2.contourArea(c) < (resized.shape[0] * resized.shape[1])/100:
		return 0 # return 0 if the shape detected is too small, and is thus likely to be a background distraction

	# get the colour of the regions inside of the contour
	color = cl.label(lab, c)
	
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape and labeled
	# color on the image.
	# This is only for debug purposes.
	"""
	ratio = image.shape[0] / float(resized.shape[0])

	# compute the center of the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)

	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	text = "{} {}".format(color, shape)
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, text, (cX, cY),
		cv2.FONT_HERSHEY_SIMPLEX, ratio, (255, 255, 255), 5)
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey()
	"""
	

	# return the results
	if(color == "red" and shape == "circle"):
		return 1
	elif(color == "red" and shape == "triangle"):
		return 2
	elif(color == "red" and shape == "square"):
		return 3
	elif(color == "green" and shape == "circle"):
		return 4
	elif(color == "green" and shape == "triangle"):
		return 5
	elif(color == "green" and shape == "square"):
		return 6
	elif(color == "blue" and shape == "circle"):
		return 7
	elif(color == "blue" and shape == "triangle"):
		return 8
	elif(color == "blue" and shape == "square"):
		return 9
	return 0