import os
import os.path as osp
#from torch.backends import cudnn
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
#from config import Config
from scipy.spatial import distance
import glob
import sys
from utils import FeatureExtractor
import torchreid

#from processor import do_inference
#from utils.logger import setup_logger
def check_sport(seq,sports_dic):
    for sport in sports_dic:
        if seq in sports_dic[sport]:
            return sport

if __name__ == "__main__":
    #cfg = Config()
    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path = '../checkpoints/sports_model.pth.tar-60',
        device='cuda'
    )   
    
    data_path = '/home/hsiangwei/Desktop/sportsmot/sportsmot_publish/dataset/test'

    seqs = os.listdir('/home/hsiangwei/Desktop/sportsmot/sportsmot_publish/dataset/test')
    for s_id,seq in enumerate(seqs):
        print(seq)
        seq = seq.replace('.txt','')
        print('processing ', s_id+1)

        imgs = sorted(glob.glob(data_path+'/{}/img1/*'.format(seq)))

        width,height = Image.open(imgs[0]).size
        scale = min(1440/width, 800/height)

        detections = np.load('/home/hsiangwei/Desktop/sportsmot/detection/twcc_yolox_trainval/{}.npy'.format(seq),allow_pickle=1)

        embs = []

        for frame_id,dets in enumerate(detections,1):
            if frame_id%100==0:
                print(frame_id,'/',len(detections))

            img = Image.open(imgs[frame_id-1])
            
            frame_emb = []

            for det in dets:
                det /= scale
                a,b,c,d,_,_,_ = det
                im = img.crop((a,b,c,d))
                im = val_transforms(im.convert('RGB')).unsqueeze(0)
                features = extractor(im)
                feat = features.cpu().detach().numpy().tolist()
                frame_emb.append(feat)

            embs.append(frame_emb)
        
        embs = np.array(embs)

        assert len(embs) == len(detections)

        np.save('/home/hsiangwei/Desktop/sportsmot/embedding/twcc_yolox_trainval/{}.npy'.format(seq),embs)
