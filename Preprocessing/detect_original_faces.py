#.........................detect_original_faces..............................

import os
import cv2
import numpy as np
import argparse
import json
from tensorflow import keras 
from tensorflow.python.keras import backend as k
 
 
from PIL import Image
from mtcnn import MTCNN

detector = MTCNN()

from glob import glob
from pathlib import Path
import json
import os
import cv2

def get_original_video_paths(root_dir_json,basename=False):
    
    originals = set()
    originals_v = set()
    
    for json_path in glob(root_dir_json):
        
        dir = Path(json_path)
        with open(json_path, "r") as f:
            metadata = json.load(f)
            
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals.add(os.path.join(dir, original))
    originals = list(originals)
    originals_v = list(originals_v)
    print(len(originals))
    #print(originals)
    #print(originals_v)
    return originals_v if basename else originals




def process_videos(video):
    for i in zip(video):
        video= os.path.join(root_dir,','.join(i))
        print(video)
        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for j in range(frames_num):
            capture.grab()
            if j % 10 != 0:
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
                    #keypoints = person['keypoints']
    
                    cv2.rectangle(crop,
                                      (bounding_box[0], bounding_box[1]),
                                      (bounding_box[0]+bounding_box[2], 
                                       bounding_box[1] + bounding_box[3]),
                                      (0,155,255),
                                      2)
            original_faces_path='F:\deepfake_data\detect_original_faces.jpg'               
            cv2.imwrite(os.path.join(original_faces_path,"{}_{}.jpg".format(id, j)),crop, [cv2.IMWRITE_JPEG_QUALITY, 100])





if __name__ == "__main__":
    root_dir_json='F:/deepfake_data/metadata/metadata.json'
    root_dir='F:/deepfake_data/train_sample_videos_2/'
    originals = get_original_video_paths(root_dir_json, basename=True)
    #print(originals)
    
    process_videos(originals)
