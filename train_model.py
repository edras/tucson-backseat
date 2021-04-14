# USAGE
# python train_model.py --dataset tucson_RGB/tucson/train/RGB --output output

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from minivggnet import MiniVGGNet
from helper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-w", "--weights", required=True, help="path to iterative weights")
args = vars(ap.parse_args())

# initialize the data and labels
data = []
labels = []
total_epoch = 40
filter = 56
depth = 3

# loop over the input images
print("[INFO] loading training images...")
for imagePath in paths.list_images(args["dataset"]):
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = preprocess(image, filter, filter)
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	seat = int(imagePath.split(os.path.sep)[-1].split('_')[-3])
	label = int(imagePath.split(os.path.sep)[-2])
	labels.append('{:0>2d}'.format(label+seat*7))

# scale the raw pixel intensities to the range [0, 1]
print("[INFO] scaling pixel intensities...")
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] spliting train and test data...")
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
print("[INFO] changing labels to vectors...")
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

# initialize the model
print("[INFO] compiling model...")
model = MiniVGGNet.build(width=filter, height=filter, depth=depth, classes=21)
opt = SGD(lr=0.01, decay=0.01 / total_epoch, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# construct the callback to save only the *best* model to disk based on the validation loss
fname = os.path.sep.join([args["output"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=32, epochs=total_epoch, callbacks=callbacks, verbose=2)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, total_epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, total_epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, total_epoch), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, total_epoch), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()