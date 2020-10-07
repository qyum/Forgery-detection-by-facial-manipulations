#......................generate a landmarks-2...................................

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

def save_landmarks(landmarks_path,video):
    
    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frames_num):
        capture.grab()
        if i % 10 != 0:
            continue
        success, frame = capture.retrieve()
        if not success:
            continue
        id = os.path.splitext(os.path.basename(video))[0]
        
    
        land = detector.detect_faces(frame)
          
        if land != []:
            
            for person in land:
                bounding_box = person['box']
                keypoints = person['keypoints']
    
                cv2.rectangle(frame,
                                  (bounding_box[0], bounding_box[1]),
                                  (bounding_box[0]+bounding_box[2], 
                                   bounding_box[1] + bounding_box[3]),
                                  (0,155,255),
                                  2)
                             
    
                cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(frame ,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
         
        #cv2.imwrite(os.path.join(landmarks_path,"{}.jpg".format(n)),img[n])
        cv2.imwrite(os.path.join(landmarks_path,"{}_{}.jpg".format(id, i)),frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

    
     
                     
                

if __name__ == '__main__':
    
    landmarks_path='F:/deepfake_data/save_landmarks_2.jpg'
    
    #save_landmarks(landmarks)
    for i in root_dir1:
        videos=f'F:/deepfake_data/train_sample_videos_2/{i}'
        
        save_landmarks(landmarks_path,videos)
 
