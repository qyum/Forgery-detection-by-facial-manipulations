#.............................................generate SSIM Differences..............................................
from pathlib import Path     
import argparse
import os 
from skimage.measure import compare_ssim
from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm
import glob


def get_original_with_fakes(root_dir_json):
    pairs = []
    for json_path in glob.glob(root_dir_json):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            #print(k);print(v)
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4] ))

    return pairs

#...............for frame_1......

 
def load_images_from_folder_1(folder):
    images = []
    for filename in glob.glob(folder):
        #img = cv2.imread(os.path.join(folder,filename))
        #if img is not None:
        #print(filename)
        a= cv2.imread(filename)
        #print(a)
        images.append(a)
    return images
    
#........for frame_2.......................
    
 
def load_images_from_folder_2(folder):
    images = []
    for filename in glob.glob(folder):
        #img = cv2.imread(os.path.join(folder,filename))
        #if img is not None:
        #print(filename)
        a= cv2.imread(filename)
        #print(a)
        images.append(a)
    return images




def save_diffs(pair,root_dir):
    ori_id, fake_id =list(zip(*pair))
    #ori_id=','.join(ori_id)
    #fake_id=','.join(fake_id)
    ori_id=list(ori_id)
    fake_id=list(fake_id)
    
    #print(ori_id)
    #print(fake_id)
    print(len(ori_id))
    print(len(fake_id))
    #...............for ori_id and fake_id.....................................
     
    #for i in zip(ori_id):
    for index in range(0,len(ori_id)):
        #i = i- 1
        #print(ori_id)
        print(ori_id[index])
        #........for ori_id.............................
        ori_id = os.path.join(root_dir,"{}.mp4".format(ori_id[index]))
        #ori_id = os.path.join(root_dir,"{}.mp4".format(','.join(ori_id[index])))
        #print(ori_id)
        capture_ori = cv2.VideoCapture(ori_id)
        frames_num_ori = int(capture_ori.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #........for fake id...........................
        #for k in zip(fake_id):
        #fake_id = os.path.join(root_dir,"{}.mp4".format(','.join(fake_id[index])))
        fake_id = os.path.join(root_dir,"{}.mp4".format(fake_id[index]))
        #print(fake_id)
        capture_fake = cv2.VideoCapture(fake_id)
        frames_num_fake = int(capture_fake.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(frames_num)
        
        for j in range(max(frames_num_ori,frames_num_fake)):
            
            #........for ori_id.............................
            capture_ori.grab()
            if j % 10 != 0:
                continue
            success, frame_1 = capture_ori.retrieve()    
            if not success:
                continue
            #id = os.path.splitext(os.path.basename(ori_id))[0]
            
            #........for fake id...........................
            capture_fake.grab()
            if j % 10 != 0:
                continue
            success, frame_2 = capture_fake.retrieve()    
            if not success:
                continue
            
            #........ ...................SSIm differences...................................
            d, a = compare_ssim(frame_1,frame_2, multichannel=True, full=True)
            #d,a  =  [compare_ssim(frame_1[i],frame_2[i],
            #multichannel=True, full=True) for i in range(0, max(len(frame_1), len(frame_2)))]
            a = 1 - a
            diff = (a * 255).astype(np.uint8)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_path='F:/deepfake_data/ssim_diff.jpg/'
            cv2.imwrite(diff_path, diff)   
        
 

if __name__=="__main__":
    
    root_dir_json='F:/deepfake_data/metadata/metadata.json'
    pairs = get_original_with_fakes(root_dir_json)
    #print(pairs)
    
     
    #for i in root_dir1:
        #videos=f'F:/deepfake_data/train_sample_videos_2/{i}'
    root_dir='F:/deepfake_data/train_sample_videos_2/'    
    save_diffs(pairs,root_dir)
