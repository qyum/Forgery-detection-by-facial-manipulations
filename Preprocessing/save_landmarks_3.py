#......................generate a landmarks-3...................................

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

def save_landmarks_3(landmarks_path,video):
    
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
        
        #..................crops image.......................
        
        xmin = 0
        ymin = 0
        ymax = frame.shape[0] - 1
        xmax = frame.shape[1] - 1
        w = xmax - xmin
        h = ymax - ymin
        p_h = h // 3
        p_w = w // 3
        crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
        h, w = crop.shape[:2]
        
        #..................save landmarks..............................
        
        land = detector.detect_faces(crop)  
        if land != []:
            
            for person in land:
                bounding_box = person['box']
                keypoints = person['keypoints']
    
                cv2.rectangle(crop,
                                  (bounding_box[0], bounding_box[1]),
                                  (bounding_box[0]+bounding_box[2], 
                                   bounding_box[1] + bounding_box[3]),
                                  (0,155,255),
                                  2)
                             
    
                cv2.circle(crop,(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(crop,(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(crop,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(crop,(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(crop,(keypoints['mouth_right']), 2, (0,155,255), 2)
         
        #cv2.imwrite(os.path.join(landmarks_path,"{}.jpg".format(n)),img[n])
        cv2.imwrite(os.path.join(landmarks_path,"{}_{}.jpg".format(id, i)),crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

    
     
                     
                

if __name__ == '__main__':
    
    landmarks_path='F:/deepfake_data/save_landmarks_3.jpg'
    #save_landmarks(landmarks)
    for i in root_dir1:
        videos=f'F:/deepfake_data/train_sample_videos_2/{i}'
        
        save_landmarks_3(landmarks_path,videos)
 
