#......................generate a landmarks...................................

import argparse
from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np

from mtcnn import MTCNN
#from keras.model import s3fd_keras

#detector = MTCNN(margin=0,thresholds=[0.65, 0.75, 0.75], device="cpu")
detector = MTCNN()

def save_landmarks(landmarks_path):
    
    mypath='F:/deepfake_data/crops1.jpg'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    img = numpy.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        
        img[n] = cv2.imread(join(mypath,onlyfiles[n]))
        land = detector.detect_faces(img[n])
        if land != []:
            
            for person in land:
                bounding_box = person['box']
                keypoints = person['keypoints']
    
                cv2.rectangle(img[n],
                                  (bounding_box[0], bounding_box[1]),
                                  (bounding_box[0]+bounding_box[2], 
                                   bounding_box[1] + bounding_box[3]),
                                  (0,155,255),
                                  2)
                             
    
                cv2.circle(img[n],(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(img[n],(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(img[n] ,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(img[n],(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(img[n],(keypoints['mouth_right']), 2, (0,155,255), 2)
        
        
        #print(img[n])
        #frame_img = Image.fromarray(img[n])
        #print(frame_img)
        #landmarks = detector.detect_faces(frame_img)
        #img = cv2.cvtColor(np.array(img[n]), cv2.COLOR_BGR2GRAY)
        #img=np.array(img[n])
        #print(img)
        #land = detector.detect_faces(img[n])
        #print(landmarks) 
        #if landmarks is not None:
            #landmarks = np.around(landmarks[0]).astype(np.int16)
        #cv2.imwrite(landmarks_path,landmarks)
            #np.save(landmark_path, landmarks)
        
        #for land in landmarks:
        cv2.imwrite(os.path.join(landmarks_path,"{}.jpg".format(n)),img[n])
        #np.save(os.path.join(landmarks_path,"{}.jpg".format(n)),land)
            #cv2.imwrite(landmarks_path,land)
    
     
                     
                

if __name__ == '__main__':
    
    landmarks='F:/deepfake_data/save_landmarks.jpg'
    
    #save_landmarks(landmarks)
