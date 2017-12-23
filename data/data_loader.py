from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
from skimage import io, transform
from sklearn import preprocessing
from  torchvision import transforms
import util.util as util
class CreateDataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, opt, csv_fileA, csv_fileB, root_dirA, root_dirB):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.attributeA_train = pd.read_csv(csv_fileA, header = 0, sep = ' ')
        self.attributeB_train = pd.read_csv(csv_fileB, header = 0, sep = ' ')

        self.root_dirA = root_dirA
        self.root_dirB = root_dirB
        self.fineSize = opt.fineSize
        self.attrA_dim = opt.attrA_dim
        self.attrB_dim = opt.attrB_dim
        self.sketch_nc = opt.sketch_nc
        self.image_nc = opt.image_nc


    def __len__(self):
        return len(self.attributeB_train)

    def __getitem__(self, idx):
        sketch_data = torch.FloatTensor(len(idx),self.sketch_nc, self.fineSize,self.fineSize)
        image_data = torch.FloatTensor(len(idx),self.image_nc, self.fineSize,self.fineSize)
        attrA_data= torch.FloatTensor(len(idx),self.attrA_dim)
        attrB_data = torch.FloatTensor(len(idx), self.attrB_dim)
        nameA = []
        nameB = []
        for i in range(len(idx)):
            img_name = os.path.join(self.root_dirA, self.attributeA_train.ix[idx[i], 0])
            # img_name = str.replace(img_name,"_A.jpg",".jpg")
            nameA.append(img_name)
            # print(img_name)
            image = io.imread(img_name)
            image = util.preprocess(image)
            image.unsqueeze(0)
            # print(image)
            image_data[i,:,:,:] = image

            sketch_name = os.path.join(self.root_dirB, self.attributeB_train.ix[idx[i], 0])
#            sketch_name = str.replace(sketch_name,"_B.jpg","_A.jpg")
            nameB.append(sketch_name)
            image = io.imread(sketch_name)
            image = util.preprocess(image)
            image.unsqueeze(0)
            sketch_data[i, :, :, :] = image
            # sketch_data[i,:,:,:] = torch.cat([image,image,image],1)
            # print(idx[i])
            attribute = self.attributeA_train.ix[idx[i], 1:].as_matrix().astype('float')
            attribute = attribute.reshape(1,-1)
            # attribute[0, 0] = -1
            # idxa = np.where(attribute == 1)
            # attribute[0,idxa]= -1 * attribute[0,idxa]
            attrA_data[i,:] = torch.from_numpy(attribute)

            attribute = self.attributeB_train.ix[idx[i], 1:].as_matrix().astype('float')
            attribute = attribute.reshape(1,-1)
            # print(attribute[0, 16])
            # attribute[0, 16] = -1*attribute[0, 16]
            # attribute[0, 9] = -1
            # print(attribute[0, 16])
            attrB_data[i,:] = torch.from_numpy(attribute)

        sample = {'sketch': sketch_data, 'attributeB': attrB_data, 'image': image_data, 'attributeA': attrA_data, 'nameA': nameA, 'nameB':nameB}


        return sample
