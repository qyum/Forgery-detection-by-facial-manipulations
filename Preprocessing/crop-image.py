#........................extract crop images...........................
from os import listdir
from os.path import isfile, join
import numpy
import cv2

from PIL import Image as im

def extract_video_crop(crops_image_path):
    
    mypath='F:/deepfake_data/image.jpg'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    img = numpy.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        
        img[n] = cv2.imread( join(mypath,onlyfiles[n]) )
        #print(img[n].shape[0]);print(img[n].shape[1]);print(img[n].shape)
        xmin = 0
        ymin = 0
        ymax = img[n].shape[0] - 1
        xmax = img[n].shape[1] - 1
        w = xmax - xmin
        h = ymax - ymin
        p_h = h // 3
        p_w = w // 3
        crop = img[n][max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
        h, w = crop.shape[:2]
        #cv2.imwrite(crops_image_path,crop)
        #cv2.imwrite('{ }.jpg'.format(crops_image_Path), crop)
        cv2.imwrite(os.path.join(crops_image_path,"{}.jpg".format(n)),crop)


if __name__ == '__main__':
    
    crops_image_path='F:/deepfake_data/crops1.jpg' 
    root_dir1=os.listdir(r'F:/deepfake_data/train_sample_videos_2') 
    #root_dir1='F:\deepfake_data\train_sample_videos_2'
    #extract_video_crop(crops_image_path)
     
     
        
