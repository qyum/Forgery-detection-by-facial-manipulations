#........................extract crop images-2...........................
from os import listdir
from os.path import isfile, join
import numpy
import cv2

from PIL import Image as im

def extract_video_crop(crops_image_path,video):
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
         
        #print(img[n].shape[0]);print(img[n].shape[1]);print(img[n].shape)
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
        #cv2.imwrite(crops_image_path,crop)
        #cv2.imwrite('{ }.jpg'.format(crops_image_Path), crop)
        #cv2.imwrite(os.path.join(crops_image_path,"{}.jpg".format(n)),crop)
        cv2.imwrite(os.path.join(crops_image_path,"{}_{}.jpg".format(id, i)),crop, [cv2.IMWRITE_JPEG_QUALITY, 100])



if __name__ == '__main__':
    
    crops_image_path='F:/deepfake_data/crops2.jpg' 
    root_dir1=os.listdir(r'F:/deepfake_data/train_sample_videos_2') 
    #root_dir1='F:\deepfake_data\train_sample_videos_2'
    for i in root_dir1:
        videos=f'F:/deepfake_data/train_sample_videos_2/{i}'
        
        extract_video_crop(crops_image_path,videos)
     
