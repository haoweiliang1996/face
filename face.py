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
# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
win = dlib.image_window()

def get_features(f):
    """
        get all features of pics in one folder
    :param faces_folder_path:
    :return: None
    """
    # Now process all the images
    print("Processing file: {}".format(f))
    img = io.imread(f)
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Now process each face we found.
    res = []
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.

        face_descriptor = facerec.compute_face_descriptor(img, shape)
        res.append(face_descriptor)
    return res
class face:
    def __init__(self):
        train = np.load('./dlib_feature.npy')
        print(len(train))
        train = np.hstack((train, np.load('./dlib_feature_train.npy')))
        #train = np.hstack((train, np.load('./record_face.npy')))
        train = list(filter(lambda s: len(s) > 0, train))
        print(len(train))
        self.database = train
        self.x = []
        self.y = []
        for i, ob in enumerate(train):
            self.x += ob
            self.y += [i] *len(ob)
        self.clf = KNeighborsClassifier(10, 'distance')
        self.face_train()
        pass

    #  take a photo when there is a face in camera
    def face_register(self,photo_nums):
        ob = []
        y = [len(self.y)] * photo_nums
        camera = cv2.VideoCapture(0)
        for _ in range(photo_nums):
            while True:
                grab, frame = camera.read()
                #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                img = cv2.resize(frame, (320,180))
                win.clear_overlay()
                win.set_image(img)
                t1 = time.time()
                dets = detector(img, 1)
                for k, d in enumerate(dets):
                    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                        k, d.left(), d.top(), d.right(), d.bottom()))
                    # Get the landmarks/parts for the face in box d.
                    shape = sp(img, d)
                    # Draw the face landmarks on the screen so we can see what face is currently being processed.

                    ob.append(facerec.compute_face_descriptor(img, shape))
                    draw = img.copy()
                    cv2.rectangle(draw, (int(d.left()), int(d.top())), (int(d.right()), int(d.bottom())), (255, 255, 255))
                    cv2.imshow("detection result", draw)
                    cv2.waitKey(30)
                    win.clear_overlay()
                if len(dets) > 0:
                    break
        self.x += ob
        self.y += y
        self.database += ob

    def face_train(self):
        self.clf.fit(self.x,self.y)
        pass

    def face_save(self):
        pass


    def face_recog(self,x):
        x = x.reshape(1, -1)
        res = self.clf.predict(x)[0]
        if self.clf.predict_proba(x).max() > 0.2127912:
            return res
        return -1

face_model = face()

if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    while True:
        grab, frame = camera.read()
        img = cv2.resize(frame, (320,180))
        win.clear_overlay()
        win.set_image(img)
        t1 = time.time()
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            print ('time: ',time.time() - t1)
            draw = img.copy()
            #cv2.rectangle(draw, (int(d.left()), int(d.top())), (int(d.right()), int(d.bottom())), (255, 255, 255))
            #cv2.imshow("detection result", draw)
            #cv2.waitKey(30)
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)
            print(face_model.face_recog(face_descriptor))
            dlib.hit_enter_to_continue()
