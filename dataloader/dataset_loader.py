""" Dataloader for PASCAL dataset. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
from utils.encode import build_colormap2label,voc_label_indices

#For FewShot
class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args):
        # Set the path according to train, val and test
        self.args=args
        if setname=='train':
            THE_PATH = osp.join(args.dataset_dir, 'Ctrain/')
            THE_PATHL = osp.join(args.dataset_dir, 'Ctrain/')
        elif setname=='test':
            THE_PATH = osp.join(args.dataset_dir, 'Cnovel/')
            THE_PATHL = osp.join(args.dataset_dir, 'Cnovel/')            
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label       
        
        # exit()    
        data = []
        label = []
        labeln=[]
        
        # Get the classes' names
        folders = os.listdir(THE_PATH)      
        
        for idx, this_folder in enumerate(folders):
            
            imf=osp.join(THE_PATH,this_folder)
            imf=imf + '/images/'
            lbf=osp.join(THE_PATHL,this_folder)
            lbf=lbf + '/masks/'
            
            this_folder_images = os.listdir(imf)
            for im in this_folder_images:
                data.append(osp.join(imf, im))
                label.append(osp.join(lbf, im[:-3]+'png'))    
                labeln.append(idx)
            
        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.labeln = labeln
        
        # Transformation for RGB
        image_size = 128
        self.transform = transforms.Compose([
            transforms.Resize(image_size + 10),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()])

        self.btransform = transforms.Compose([
            transforms.Resize(image_size + 10),
            transforms.CenterCrop(image_size)])

        self.colormap = build_colormap2label()  
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        inppath, labpath,idx = self.data[i], self.label[i],self.labeln[i]
        inpimage = self.transform(Image.open(inppath).convert('RGB'))
        labimage = self.btransform(Image.open(labpath).convert('RGB'))
        labn= np.asarray(labimage,dtype="int32")
        labz= voc_label_indices(labn,self.colormap)
        labmask=torch.from_numpy(labz).long()
        return inpimage,labmask,idx
