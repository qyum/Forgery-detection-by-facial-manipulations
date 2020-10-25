
#.................................generate a folds..............................

#import glob
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



def get_original_with_fakes(root_dir_json):
    
    pairs = []
    for json_path in glob(root_dir_json):
        
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            #print(k);print(v)
            original = v.get("original",None)
            #print(original)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4] ))

    return pairs

def get_paths1(vid, label, root_dir,video_fold):
    ori_vid, fake_vid =list(zip(*vid))
    #print(ori_vid)
    #print(fake_vid)
    ori_vid=list(ori_vid)
    fake_vid=list(fake_vid)
    #print(ori_vid)
    #print(fake_vid)
    
    #ori_vid=['iufotyxgzb', 'aytzyidmgs', 'bzythlfnhq', 'vcxckqbaya', 'ybjrqnqnno', 'afoovlsmtx', 'cppdvdejkc', 'jepguaulgf', 'kuelhabsmz', 'rvoudrbyac', 'kdodrvufdh', 'ccfoszqabv', 'ujzwwfkeia', 'bxzakyopjf', 'ybetenmsye', 'dkuayagnmc', 'xlbnmndmku', 'kgbkktcjxf', 'nlerwupaqr', 'gzyzdcbuuv', 'gomwfvijiv', 'swedbyuehz', 'ywvlvpvroj', 'yoavqsqobz', 'fecysfujzk', 'lmlyvmfbbe', 'ehtdtkmmli', 'jzmdganfys', 'bwipwzzxxu', 'qzklcjjxdq', 'bulkxhhknf', 'rrcsuwgpnd', 'xngpzquyhs', 'cprhtltsjp', 'jkddywriuf', 'fysyrqfguw', 'wwqiuiwdbz', 'meawmsgiti', 'tqhbgzfwsf', 'qjlhemtkxk', 'upgerjvcjb', 'gbqrgajyca', 'gxhcuxulhi', 'mgowkzsbyx', 'gipbyjfxfp', 'vrsinxahfh', 'wgmbcqfgkp', 'gdfyzwykty', 'xxrzzncksa', 'edyncaijwx', 'dlpoieqvfb', 'sgjnvxvcpu', 'yxyhvdlrgk', 'xwcggrygwl', 'vudstovrck', 'mfzqxktxud', 'hyuipchisa', 'brwrlczjvi', 'crezycjqyk', 'gnyspcpbhd', 'hyhjfdxqxy', 'gneufaypol', 'proiippuup', 'lyvlnqduqg', 'djxdyjopjd', 'tfoixxmpoo', 'iuzdfwsefw', 'ohnonevlro', 'joeifeskbs', 'egghxjjmfg', 'puppdcffcj', 'luvasmspox', 'tivkmbqgwp', 'jomvcqqars', 'yagllixjvh', 'iieoqptzec', 'cyxlcuyznd', 'tmdformfqp', 'efwfxwwlbw', 'ellavthztb', 'qzimuostzz', 'topyiohccg', 'gjypopglvi', 'dakiztgtnw', 'cpjxareypw', 'jszyyhamrh', 'ffcwhpnpuw', 'fygviyzcjm', 'xqnykluhws', 'qokxxuayqn', 'jdubbvfswz', 'vmospzljws', 'fntskqfxxf', 'xnfwdpptym', 'sqwvfgwdxr', 'hcswybumab', 'qhkqqfznrg', 'oesxbvktem', 'rlldzrnmdn', 'jjyfvzxwwx', 'yfsnwkbafm', 'uuxqylnzls', 'atkdltyyen', 'kwyvikrgmx', 'ljaifbsfuw', 'fdcttsvjwf', 'svcnlasmeh', 'fpvduejzcw', 'ngdswpaqnt', 'yjlsxqoauz', 'grnycmbdfu', 'dzyuwjkjui', 'bejhvclboh', 'rlvgtsjyer', 'cizlkenljw', 'fgfyrfyqay', 'fkqptfouqw', 'bgwmmujlmc', 'ehccixxzoe', 'tdohqkzvbk', 'liniegczcx', 'zhfyuhonra', 'ezaajaswoe', 'tamayudqqx', 'nzquxipbye', 'chtapglbcj', 'kbvibjhfzo', 'ygdgwyqyut', 'qedsgieuqn', 'ifbdbogiqn', 'gktjowiuqe', 'kysxawkest', 'znpdbbsfvj', 'zrkinjhsuq', 'wtreibcmgm', 'ijokcwewbs', 'ixuouyigxa', 'pqvypayzrp', 'ppdpgwyjgm', 'kydlpqfrvv', 'bffwsjxghk', 'ifjktxxiln', 'lkdlzpkukw', 'fjlyaizcwc', 'slwkmefgde', 'mmhqllmlew', 'nvpluswotp', 'olakcrnuro', 'lulmevqtla', 'keecvpbncd', 'sunqwnmlkx', 'sfujxhuyje', 'ubplsigbvj', 'lietldeotq', 'caifxvsozs', 'wapebjxejr', 'kkzsnmrkqk', 'sasoxcqisz', 'inkqxytzyu', 'dbtbbhakdv', 'fkyrrigzpt', 'smggzgxymo', 'xagsvjctmp', 'ztbinwxgyu', 'abarnvbtwb', 'mfnowqfdwl', 'qtnjyomzwo', 'vgqotmftcr', 'tfoxelmnjx', 'qapnbtdypb', 'duycddgtrl', 'eckvhdusax', 'pylnolwenx', 'ytufbmkdlq', 'rjlgchzmfv', 'dbnygxtwek', 'fewcljwqkr', 'uonshkejav', 'iyefnuagav', 'iiomvouemm', 'gmihbscmwq', 'itmwoyxbas', 'znjupdqnwo', 'mfpgdgsaxg', 'cmbzllswnl', 'hqtepxaeqx', 'atvmxvwyns', 'woshnzbxmc', 'ptokilxwcx', 'qeumxirsme', 'rfzzrftgco', 'euqpvnyxrb', 'jljpdojupu', 'avmjormvsx', 'iklzfeueid', 'fhghkqdkhe', 'xnhcreiyqg', 'bdnaqemxmr', 'imzqmbfugn', 'qypgyrxcme', 'gxembgiarp', 'ekcrtigpab', 'xobhsemxmv', 'xxsxktyvzt', 'xclqbefnvc', 'drcyabprvt', 'itzmdwutdu', 'jwcsqxzdlv', 'xzvrgckqkz']
    #fake_vid=['iufotyxgzb', 'aytzyidmgs', 'bzythlfnhq', 'vcxckqbaya', 'ybjrqnqnno', 'afoovlsmtx', 'cppdvdejkc', 'jepguaulgf', 'kuelhabsmz', 'rvoudrbyac', 'kdodrvufdh', 'ccfoszqabv', 'ujzwwfkeia', 'bxzakyopjf', 'ybetenmsye', 'dkuayagnmc', 'xlbnmndmku', 'kgbkktcjxf', 'nlerwupaqr', 'gzyzdcbuuv', 'gomwfvijiv', 'swedbyuehz', 'ywvlvpvroj', 'yoavqsqobz', 'fecysfujzk', 'lmlyvmfbbe', 'ehtdtkmmli', 'jzmdganfys', 'bwipwzzxxu', 'qzklcjjxdq', 'bulkxhhknf', 'rrcsuwgpnd', 'xngpzquyhs', 'cprhtltsjp', 'jkddywriuf', 'fysyrqfguw', 'wwqiuiwdbz', 'meawmsgiti', 'tqhbgzfwsf', 'qjlhemtkxk', 'upgerjvcjb', 'gbqrgajyca', 'gxhcuxulhi', 'mgowkzsbyx', 'gipbyjfxfp', 'vrsinxahfh', 'wgmbcqfgkp', 'gdfyzwykty', 'xxrzzncksa', 'edyncaijwx', 'dlpoieqvfb', 'sgjnvxvcpu', 'yxyhvdlrgk', 'xwcggrygwl', 'vudstovrck', 'mfzqxktxud', 'hyuipchisa', 'brwrlczjvi', 'crezycjqyk', 'gnyspcpbhd', 'hyhjfdxqxy', 'gneufaypol', 'proiippuup', 'lyvlnqduqg', 'djxdyjopjd', 'tfoixxmpoo', 'iuzdfwsefw', 'ohnonevlro', 'joeifeskbs', 'egghxjjmfg', 'puppdcffcj', 'luvasmspox', 'tivkmbqgwp', 'jomvcqqars', 'yagllixjvh', 'iieoqptzec', 'cyxlcuyznd', 'tmdformfqp', 'efwfxwwlbw', 'ellavthztb', 'qzimuostzz', 'topyiohccg', 'gjypopglvi', 'dakiztgtnw', 'cpjxareypw', 'jszyyhamrh', 'ffcwhpnpuw', 'fygviyzcjm', 'xqnykluhws', 'qokxxuayqn', 'jdubbvfswz', 'vmospzljws', 'fntskqfxxf', 'xnfwdpptym', 'sqwvfgwdxr', 'hcswybumab', 'qhkqqfznrg', 'oesxbvktem', 'rlldzrnmdn', 'jjyfvzxwwx', 'yfsnwkbafm', 'uuxqylnzls', 'atkdltyyen', 'kwyvikrgmx', 'ljaifbsfuw', 'fdcttsvjwf', 'svcnlasmeh', 'fpvduejzcw', 'ngdswpaqnt', 'yjlsxqoauz', 'grnycmbdfu', 'dzyuwjkjui', 'bejhvclboh', 'rlvgtsjyer', 'cizlkenljw', 'fgfyrfyqay', 'fkqptfouqw', 'bgwmmujlmc', 'ehccixxzoe', 'tdohqkzvbk', 'liniegczcx', 'zhfyuhonra', 'ezaajaswoe', 'tamayudqqx', 'nzquxipbye', 'chtapglbcj', 'kbvibjhfzo', 'ygdgwyqyut', 'qedsgieuqn', 'ifbdbogiqn', 'gktjowiuqe', 'kysxawkest', 'znpdbbsfvj', 'zrkinjhsuq', 'wtreibcmgm', 'ijokcwewbs', 'ixuouyigxa', 'pqvypayzrp', 'ppdpgwyjgm', 'kydlpqfrvv', 'bffwsjxghk', 'ifjktxxiln', 'lkdlzpkukw', 'fjlyaizcwc', 'slwkmefgde', 'mmhqllmlew', 'nvpluswotp', 'olakcrnuro', 'lulmevqtla', 'keecvpbncd', 'sunqwnmlkx', 'sfujxhuyje', 'ubplsigbvj', 'lietldeotq', 'caifxvsozs', 'wapebjxejr', 'kkzsnmrkqk', 'sasoxcqisz', 'inkqxytzyu', 'dbtbbhakdv', 'fkyrrigzpt', 'smggzgxymo', 'xagsvjctmp', 'ztbinwxgyu', 'abarnvbtwb', 'mfnowqfdwl', 'qtnjyomzwo', 'vgqotmftcr', 'tfoxelmnjx', 'qapnbtdypb', 'duycddgtrl', 'eckvhdusax', 'pylnolwenx', 'ytufbmkdlq', 'rjlgchzmfv', 'dbnygxtwek', 'fewcljwqkr', 'uonshkejav', 'iyefnuagav', 'iiomvouemm', 'gmihbscmwq', 'itmwoyxbas', 'znjupdqnwo', 'mfpgdgsaxg', 'cmbzllswnl', 'hqtepxaeqx', 'atvmxvwyns', 'woshnzbxmc', 'ptokilxwcx', 'qeumxirsme', 'rfzzrftgco', 'euqpvnyxrb', 'jljpdojupu', 'avmjormvsx', 'iklzfeueid', 'fhghkqdkhe', 'xnhcreiyqg', 'bdnaqemxmr', 'imzqmbfugn', 'qypgyrxcme', 'gxembgiarp', 'ekcrtigpab', 'xobhsemxmv', 'xxsxktyvzt', 'xclqbefnvc', 'drcyabprvt', 'itzmdwutdu', 'jwcsqxzdlv', 'xzvrgckqkz']
    
    data_1 = []
   
    for i in zip(ori_vid):
        ori_vid = os.path.join(root_dir,"{}.mp4".format(','.join(i)))
        #print(ori_id)
        capture_ori = cv2.VideoCapture(ori_vid)
        frames_num = int(capture_ori.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(frames_num)
        for j in range(frames_num):
            capture_ori.grab()
            if j % 10 != 0:
                continue
            success,ori_img = capture_ori.retrieve()    
            if not success:
                continue
            id = os.path.splitext(os.path.basename(ori_vid))[0]
            
            #data_1.append({'ori_vid':ori_vid, 'ori_id':"{}_{}.jpg".format(id, j),'ori_img':ori_img,
                           #'label':label})
            data_1.append({'ori_vid':ori_vid, 'ori_id':"{}_{}.jpg".format(id, j),'label':label})
            
            #print(df)
            #df.to_csv(r'F:\deepfake_data\deepfake_data.csv',index = False, header=True)
            #print(x)
            #csv_data_1.append(pd.DataFrame(data))
    df=pd.DataFrame(data_1)
    print(df)
    #df.to_csv(r'F:\deepfake_data\deepfake_ori_img.csv',index = False, header=True)
    df.to_csv(r'F:\deepfake_data\deepfake_data.csv',index = False, header=True)
                                                                                                                                                                        
             

def get_paths2(vid, label, root_dir,video_fold):
    ori_vid, fake_vid =list(zip(*vid))
    #print(ori_vid)
    #print(fake_vid)
    ori_vid=list(ori_vid)
    fake_vid=list(fake_vid)
    #print(ori_vid)
    #print(fake_vid)
    
    data_2= []
    for i in zip(fake_vid):
        fake_vid = os.path.join(root_dir,"{}.mp4".format(','.join(i)))
        #print(ori_id)
        capture_fake = cv2.VideoCapture(fake_vid)
        frames_num = int(capture_fake.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(frames_num)
        for j in range(frames_num):
            capture_fake.grab()
            if j % 10 != 0:
                continue
            success,fake_img = capture_fake.retrieve()    
            if not success:
                continue
            id = os.path.splitext(os.path.basename(fake_vid))[0]
            data_2.append({'ori_vid':fake_vid, 'ori_id':"{}_{}.jpg".format(id, j),'ori_img':fake_img,
                          'label':label})
            #data_2.append({'ori_img':fake_img,'label':label})
                          
            #data_2.append({'ori_vid':fake_vid, 'ori_id':"{}_{}.jpg".format(id, j),'label':label})
             
            #csv_data=r'F:/deepfake_data/deepfake_data_1.csv'
            #x=pd.DataFrame(data)
            #print(x)
            #x.to_csv(r'F:\deepfake_data\deepfake_data_1.csv',index = False, header=True)
    
    x=pd.DataFrame(data_2)
    print(x)
    x.to_csv(r'F:\deepfake_data\deepfake_fake_img.csv',index = False, header=True) 
    #x.to_csv(r'F:\deepfake_data\deepfake_data_1.csv',index = False, header=True) 
     


if __name__=="__main__":
    
    root_dir_json='F:/deepfake_data/metadata/metadata.json'
    root_dir='F:/deepfake_data/train_sample_videos_2/'
    ori_fakes = get_original_with_fakes(root_dir_json)
    #print(ori_fakes)
    n_splits=16
    sz = 50 //n_splits
    
    folds = []
    for fold in range(n_splits):
        folds.append(list(range(sz * fold, sz * fold + sz  if fold <n_splits - 1 else 50)))
    #print(folds)
    
    video_fold = {}
    #print(os.listdir(root_dir))
    for d in os.listdir(root_dir):
        #if "dfdc" in d:
        #part = d.split("_")[-1]
        #part = int(d.split("_")[-1])
        #print(part)
        for json_path in glob(root_dir_json):
            with open(json_path, "r") as f:
                metadata = json.load(f)
            for k, v in metadata.items():
                #print(k);print(v)
                fold = None
                for i, fold_dirs in enumerate(folds):
                    #print(fold_dirs)
                    #if part in fold_dirs:
                    fold = i
                    #print(fold)
                    break
                assert fold is not None
                video_id = k[:-4]
                #print(video_id)
                #print(fold)
                video_fold[video_id] = fold
    #print(video_fold)
    #print(len(folds))
    for fold in range(len(folds)):
        holdoutset = {k for k, v in video_fold.items() if v == fold}
        #print(holdoutset)
        trainset = {k for k, v in video_fold.items() if v != fold}
        #print(trainset)
        assert holdoutset.isdisjoint(trainset), "Folds have leaks"
    
    ori_ori = set([(ori, ori) for ori, fake in ori_fakes])
    #print(ori_ori)
    
    label_0=0
    label_1=1
    #get_paths1(ori_ori,label_0,root_dir,video_fold) 
    #get_paths2(ori_fakes,label_1,root_dir,video_fold)
    
    #for_ori_data
    deepfake_data_ori=pd.read_csv('F:\deepfake_data\deepfake_data.csv')
    deepfake_data_ori.head()
    deepfake_data_1_fake=pd.read_csv('F:\deepfake_data\deepfake_data_1.csv')
    deepfake_data_1_fake.head()
    data=deepfake_data_ori.append(deepfake_data_1_fake)
    data.tail(10)
    
