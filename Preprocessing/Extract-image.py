#.....................................extract the images................................

 
#from functools import partial
#from glob import glob
#from multiprocessing.pool import Pool
#from os import cpu_count

#import cv2
from tqdm import tqdm
#from joblib import Parallel, delayed


 
def extract_video(video,root_save_image):
     
    capture = cv2.VideoCapture(video)
    #print(capture)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(frames_num)


    for i in range(frames_num):
        capture.grab()
        if i % 10 != 0:
            continue
        success, frame = capture.retrieve()
        #print(frame)
        if not success:
            continue
        id = os.path.splitext(os.path.basename(video))[0]
         
        #print(frame)
        #print(id)
        #cv2.imwrite(os.path.join(root_save_image,"jpegs", "{}_{}.jpg".format(id, i)), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(os.path.join(root_save_image,"{}_{}.jpg".format(id, i)), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

if __name__ == '__main__':
    
    root_save_image='F:/deepfake_data/image.jpg'
    root_dir1=os.listdir(r'F:/deepfake_data/train_sample_videos_2') 
     
    
     
    for i in root_dir1:
        videos=f'F:/deepfake_data/train_sample_videos_2/{i}'
        #extract_video(videos,root_save_image)
        
