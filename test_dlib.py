# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  This code will also use CUDA if you have CUDA and cuDNN
#   installed.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2
import mxnet as mx
import cv2
import os
import time

predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

import numpy as np
res_all = []
# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
win = dlib.image_window()
def get_features(faces_folder_path):
    # Now process all the images
    res = []
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)
        win.clear_overlay()
        win.set_image(img)
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        # Now process each face we found.
        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)

            # Compute the 128D vector that describes the face in img identified by
            # shape.  In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people.  He we just print
            # the vector to the screen.
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            res.append(np.array(face_descriptor))
            # It should also be noted that you can also call this function like this:
            #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100)
            # The version of the call without the 100 gets 99.13% accuracy on LFW
            # while the version with 100 gets 99.38%.  However, the 100 makes the
            # call 100x slower to execute, so choose whatever version you like.  To
            # explain a little, the 3rd argument tells the code how many times to
            # jitter/resample the image.  When you set it to 100 it executes the
            # face descriptor extraction 100 times on slightly modified versions of
            # the face and returns the average result.  You could also pick a more
            # middle value, such as 10, which is only 10x slower but still gets an
            # LFW accuracy of 99.3%.


            dlib.hit_enter_to_continue()
    res_all.append(res)
import matplotlib.pyplot as plt
if __name__ == '__main__':
    data_dir = '/home/haowei/face/facedata0820/ver_training'
    data_dir = '/home/haowei/face/facedata0820/tmp'
    for i in os.listdir(data_dir):
        print(i)
        get_features(os.path.join(data_dir,i))
    '''
    camera = cv2.VideoCapture(0)
    while True:
        grab, frame = camera.read()
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame, (320,180))
        win.clear_overlay()
        win.set_image(img)
        #cv2.imshow("face",img)
        #cv2.waitKey(30)
        t1 = time.time()
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.

            face_descriptor = facerec.compute_face_descriptor(img, shape)
            print(face_descriptor)


            print ('time: ',time.time() - t1)
            
            draw = img.copy()
            cv2.rectangle(draw, (int(d.left()), int(d.top())), (int(d.right()), int(d.bottom())), (255, 255, 255))
            cv2.imshow("detection result", draw)
            cv2.waitKey(30)
            win.clear_overlay()
            
            win.add_overlay(d)
            win.add_overlay(shape)
            dlib.hit_enter_to_continue()
            from sklearn.linear_model import LogisticRegression
        #np.save("dlib_feature_val",np.array(res_all))
    '''
