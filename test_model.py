# USAGE
# python test_model.py --input tucson_RGB_wholeImage/tucson/test/RGB_wholeImage --model output/weights-028-0.0554.hdf5

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from helper import preprocess
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-m", "--model", required=True, help="path to input model")
args = vars(ap.parse_args())

filter = 56
depth = 3

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# randomy sample a few of the input images
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(5,),
	replace=False)

# Are we handling a whole image?
whole_image = True
# imagePaths = ['tucson_RGB_wholeImage/tucson/train/RGB_wholeImage/tucson_train_imageID_0_GT_6_6_0.png']

# loop over the image paths
for imagePath in imagePaths:

	# read image from dataset
	image = cv2.imread(imagePath)

	# Are we handling a whole image? -> let's split it into 3
	if whole_image:
		(h, w) = image.shape[:2]
		# generate 3 slices of 550hx250w
		img0 = image[70:(70+550), 130:(130+250)].copy()
		img1 = image[70:(70+550), 363:(363+250)].copy()
		img2 = image[70:(70+550), 582:(582+250)].copy()
		#cv2.imshow("left", img0)
		#cv2.imshow("middle", img1)
		#cv2.imshow("right", img2)
		#cv2.waitKey(0)
		test = [ img0, img1, img2 ]
	else:
		test = [ image ]

	output = image
	predictions = []

	humanize = { 1: 'l-empty', 2: 'l-baby', 3:'l-kid', 4:'l-adult', 5:'l-item', 6:'l-baby chair', 7:'l-chair',
				 8: 'm-empty', 9: 'm-baby', 10:'m-kid', 11:'m-adult', 12:'m-item', 13:'m-baby chair', 14:'m-chair',
				 15: 'r-empty', 16: 'r-baby', 17:'r-kid', 18:'r-adult', 19:'r-item', 20:'r-baby chair', 21:'r-chair',}

	for idx, img in enumerate(test):
		# pre-process the ROI and classify it
		roi = preprocess(img, filter, filter)
		roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
		pred = model.predict(roi).argmax(axis=1)[0] + 1
		predictions.append(str(pred))

		cv2.putText(output, str(humanize.get(pred)), (125+idx*300, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

	# show the output image
	print("[INFO] prediction: {}".format(" ".join(predictions)))
	cv2.imshow("Output", output)
	cv2.waitKey()