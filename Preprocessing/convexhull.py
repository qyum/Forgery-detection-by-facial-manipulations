from glob import glob
import argparse
import json
import os
import random
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import convex_hull_image
from skimage.util import invert


#.............................black_out_convex_hull..............................
import dlib
import skimage.draw
from skimage import measure

detector_1 = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('G:/shape_predictor_68_face_landmarks.dat')

from facenet_pytorch.models.mtcnn import MTCNN
detector_2 = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cpu")

def blackout_convex_hull(img):
    #try:
    #img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    rect = detector_1(img,1)
    #sp = predictor(image=img, box=rect)
    #print(rect)
    for d in rect:
        
        #cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
        shape = predictor(img, d)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        #print(landmarks)
        outline =landmarks[[*range(17), *range(26, 16, -1)]]
        #for i in range(shape.num_parts):
            #p = shape.part(i)
            #cv2.circle(img, (p.x, p.y), 2, 255, 1)
            #cv2.putText(img, str(i), (p.x + 4, p.y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
       
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        cropped_img = np.zeros(img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1
        y, x = measure.centroid(cropped_img)
        y = int(y)
        x = int(x)
        first = random.random() > 0.5
        if random.random() > 0.5:
            
            if first:
                cropped_img[:y, :] = 0
            else:
                cropped_img[y:, :] = 0
        else:
            if first:
                cropped_img[:, :x] = 0
            else:
                cropped_img[:, x:] = 0

        img[cropped_img > 0] = 0
     
    
    return img
    
    
    #....................blend_original.............................

from albumentations.pytorch.functional import img_to_tensor
from scipy.ndimage import binary_erosion, binary_dilation
from albumentations import ImageCompression, OneOf, GaussianBlur, Blur


def blend_original(img):
    back_img=img.copy()
    #print(back_img)
    img = img.copy()
    h, w = img.shape[:2]
    #print(h)
    #print(w)
    rect = detector_1(img)
    print(rect)
    if len(rect) == 0:
        
        #return img
        img=img.copy()
        img = OneOf([GaussianBlur(), Blur()], p=0.5)(image=img)["image"]   
        img = ImageCompression(quality_lower=40, quality_upper=95)(image=img)["image"]
        return img
    else:
        #print(rect[0])
        rect = rect[0]
    
    #for d in rect:
        #cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
        #sp = predictor(img, d) 
        
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        #print(Y)
        #print(X)
        raw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        raw_mask[Y, X] = 1
        #print(raw_mask)
        face = img * np.expand_dims(raw_mask, -1)
        #print(face)
        # add warping
        h1 = random.randint(h - h // 2, h + h // 2)
        w1 = random.randint(w - w // 2, w + w // 2)
        while abs(h1 - h) < h // 3 and abs(w1 - w) < w // 3:
            h1 = random.randint(h - h // 2, h + h // 2)
            w1 = random.randint(w - w // 2, w + w // 2)
        face = cv2.resize(face, (w1, h1), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]))
        face = cv2.resize(face, (w, h), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]))

        raw_mask = binary_erosion(raw_mask, iterations=random.randint(4, 10))
        img[raw_mask, :] = face[raw_mask, :]
        #if random.random() < 0.2:
        img = OneOf([GaussianBlur(), Blur()], p=0.5)(image=img)["image"]
        # image compression
        #if random.random() < 0.5:
        img = ImageCompression(quality_lower=40, quality_upper=95)(image=img)["image"]
        #blended image
        b_img=img*np.expand_dims(raw_mask, -1)+(1-np.expand_dims(raw_mask, -1))*back_img
        #print(b_img)
        return b_img
        
        
        
        def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks
    
    
    
    #.......................random blackout...........................
def blackout_random(image, mask):
    binary_mask = mask > 0.4 * 255
    h, w = binary_mask.shape[:2]

    tries = 50
    current_try = 1
    while current_try < tries:
        first = random.random() < 0.5
        if random.random() < 0.5:
            pivot = random.randint(h // 2 - h // 5, h // 2 + h // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:pivot, :] = 0
            else:
                bitmap_msk[pivot:, :] = 0
        else:
            pivot = random.randint(w // 2 - w // 5, w // 2 + w // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:, :pivot] = 0
            else:
                bitmap_msk[:, pivot:] = 0

        if  np.count_nonzero(image * np.expand_dims(bitmap_msk, axis=-1)) / 3 > (h * w) / 5 \
                or np.count_nonzero(binary_mask * bitmap_msk) > 40:
            mask *= bitmap_msk
            image *= np.expand_dims(bitmap_msk, axis=-1)
            break
        current_try += 1
    return image
    
    
    
    from mtcnn import MTCNN
 
detector = MTCNN()


x='F:/test/barak_real.mp4'
#x='F:/deepfake_data/train_sample_videos_2/aagfhgtpmv.mp4'
capture_ori = cv2.VideoCapture(x)
frames_num = int(capture_ori.get(cv2.CAP_PROP_FRAME_COUNT))
print(frames_num)

data_1=[]
for j in range(0,frames_num):
    capture_ori.grab()
    if j % 100 != 0:
        continue
    success,frame = capture_ori.retrieve()
         
    #print(frame)
    if not success:
        continue
    id = os.path.splitext(os.path.basename(x))[0]
    #print(id)       
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
                 
    #crop=blackout_convex_hull(crop)
    crop=blend_original(crop)
    #mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    #if random.random() < 0.1:
        #binary_mask = mask > 0.4 * 255
        #crop= prepare_bit_masks((binary_mask * 1).astype(np.uint8))
        #print(crop)
        #crop=blackout_random(crop, mask)
        
         
        #cv2.imwrite(os.path.join(landmarks_path,"{}.jpg".format(n)),img[n])
    train_ori_img='F:/test/black_out/'
    cv2.imwrite(os.path.join(train_ori_img,"{}_{}.jpg".format(id, j)),crop, 
                     [cv2.IMWRITE_JPEG_QUALITY, 100])
    #data_1.append({'ori_vid':x, 'ori_id':"{}_{}.jpg".format(id, j)}) 

#df=pd.DataFrame(data_1)
#df=df.iloc[1:9]
#print(df)
#df.to_csv('F:/test/barak_o_real.csv',index = False, header=True)
#df.to_csv(r'F:\deepfake_data\deepfake_data.csv',index = False, header=True)

    
            


