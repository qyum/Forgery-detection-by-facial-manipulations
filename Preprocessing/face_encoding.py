#...............................face_encoding...........................
import argparse
import os
from functools import partial 
from tqdm import tqdm
import random
import face_recognition
import numpy as np


from glob import glob
from pathlib import Path
import json
import os
import cv2

def get_original_video_paths(root_dir_json,basename=True):
    
    originals = set()
    originals_v = set()
    for json_path in glob(root_dir_json):
        
        dir = Path(json_path)
        #print(dir)
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
    #print(len(originals))
    #print(originals)
    #print(originals_v)
    return originals_v if basename else originals

def write_face_encodings(video, root_dir):
    video_id, *_ = os.path.splitext(','.join(video))
    print(video)
    
    for i in zip(video):
        video= os.path.join(root_dir,','.join(i))
        #print(video)
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
            #...................encoding..........................
            #img = face_recognition.load_image_file(crop)
            #boxes = face_recognition.face_locations(crop)
		                                        
            encoding = face_recognition.face_encodings(crop,num_jitters=10)
            #print(encoding[0])
            encoding_path='F:/deepfake_data/encoding.jpg'
            #np.save(os.path.join(encoding_path, "encodings"), encodings)
            #if encoding:
             
            #cv2.imwrite(os.path.join(encoding_path,"{}_{}.jpg".format(id, j)),encodings)
            cv2.imwrite(os.path.join(encoding_path,"{}_{}.jpg".format(id, j)),encoding[0])
        
        
        

if __name__=="__main__":
    
    root_dir_json='F:/deepfake_data/metadata/metadata.json'
    root_dir='F:/deepfake_data/train_sample_videos_2/'
    originals = get_original_video_paths(root_dir_json, basename=True)
    #print(len(originals))
    #print(originals)
    write_face_encodings(originals,root_dir)
    
