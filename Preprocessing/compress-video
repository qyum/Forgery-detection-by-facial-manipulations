#.......................................preparing the dataset..........................................
#..............first compress the video..............


import os
import random
import subprocess

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"



def compress_video(data_folder):

  for subdirs,dirs,files in os.walk(data_folder):
    for file in files:
      extension=os.path.splitext(file)[-1].lower()
      #print(extension)
      #print(subdirs);print(dirs)
      if extension=='.mp4':
        media_in=subdirs+'/'+file
        #print(media_in)
        media_out=subdirs+'/compressed/'+file
        #print(media_out)
        #subprocess.run('ffmpeg-i'+media_in.replace('','\\ ')+'-vcodec libx264 -crf 22'+media_out.replace('','\\ '),shell=True)
        lvl = random.choice([23, 28, 32])
        subprocess.run("ffmpeg -i {} -c:v libx264 -crf {} -threads 1 {}".format(media_in, lvl, media_out),shell=True)
   

if __name__ == '__main__':
    data_folder='F:\deepfake_data\train_sample_videos_2'
    compress_video(data_folder)
