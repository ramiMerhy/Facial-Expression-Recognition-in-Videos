import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
import argparse
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import load

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random

def frames3average(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval, numberOfFrames,cat2Label, transformTrain,transformValidation):
    train_dataset = ImageDataset(
        #video roo is the video train folder
        video_root=root_train,
        #go to the list "afew_All.txt" (the map)
        video_list=list_train,
        #convert the emotion strings to nb
        rectify_label=cat2Label,
        #get the image and apply the data augmentation process by resizing it to 224*224; fix the image variations radomly (rotate the image, fix the zooming, translate its location and add black and white)
        transform=transformTrain,
        #number of frame taken from each video per input
        numberOfFrames=numberOfFrames
    )

    #same for the validation set
    val_dataset = ImageDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cat2Label,
        transform=transformValidation,
        numberOfFrames=numberOfFrames)
    
    #give the loader the dataset, the batch size, shuffle(to shuffle the emotions), num_workers(nb of cpu to be used to work on the dataset) and the pin_memory (to cache to images to avoid reading them everytime)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True)
    #same for the validation set
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)

    
    return train_loader, val_loader

class ImageDataset(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None,numberOfFrames=5):
        self.imgs,self.classDistribution,self.numberOfVideos,self.labeledVideo,self.weights = load_imgs_tsn(video_root, video_list,rectify_label,numberOfFrames)
        self.transform = transform
        
    def __getitem__(self, index):
        toReturn=[]
        a=[]
        #get the 3 images path and label
        for Path in self.imgs[index]:
            path_first, target_first,index = Path
            a.append(path_first)
            #open the image then convert it to RGB
            img_first = Image.open(path_first).convert("RGB")
            #apply the flip, scale to the image if exists
            if self.transform is not None:
                img_first = self.transform(img_first)

            toReturn.append(img_first)
        #return the image with the label and (a is the path of the image) and the index (map the images to the video)
        return toReturn,target_first,a,index

    def __len__(self):
        return len(self.imgs)

def load_imgs_tsn(video_root, video_list, rectify_label,numberOfFrames=5):
    img_paths=list()
    classDistribution=[]
    index=0
    labelVideo=[]
    weight=[]
    #get the video root and open the video list
    with open(video_list, 'r') as imf:
        index = -1
        count=0
        for id, line in enumerate(imf):
            index+=1
            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video
            
            labelVideo.append(label)
            #print(video_root)
            #print(video_name)
            video_path = video_root+"/" + video_name # video_path is the path of each video
            #print(video_path)  
            #give us all the images name
            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video
            #get the nb of parts
            num_per_part = int(img_count) // numberOfFrames
            toReturn=[]
            #repeat the process proportionally to the nb of images
            numberOfTimes=int((img_count/numberOfFrames))

            #make sure that I have enough images for the process
            if int(img_count) > numberOfFrames:
                for m in range(numberOfTimes):
                    for i in range(numberOfFrames):
                        #get a random number based on the parts
                        rand=random.randint(i*num_per_part,(i+1)*num_per_part-1)
                        #concatinate the path with the image name then save it
                        toReturn.append((video_path+"/"+ img_lists[rand],label,index))
                        weight.append(label)

                    img_paths.append(toReturn)
                    toReturn=[]
            else:
                #get the nb of missing images
                #print("number oOF Frames",numberOfFrames)
                #print("img_count oOF Frames",img_count)
                #print("video_path",video_path)

                missingImages=numberOfFrames-int(img_count)
                #fullfil the catched images
                for j in range(img_count):
                    toReturn.append((video_path+"/"+ img_lists[j],label,index))
                #duplicate random image to fullfil the missing nb
                for t in range(missingImages):
                    rand=random.randint(0,len(toReturn)-1)
                    toReturn.insert(rand, toReturn[rand])

                img_paths.append(toReturn)
                weight.append(label)
    #return the list of all the paths of images
    return img_paths,classDistribution,index+1,np.array(labelVideo),weight



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    

    
def frames3averagedataset(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval, numberOfFrames,cat2Label, transformTrain,transformValidation):
    train_dataset = ImageDataset2(
        #video roo is the video train folder
        video_root=root_train,
        #go to the list "afew_All.txt" (the map)
        video_list=list_train,
        #convert the emotion strings to nb
        rectify_label=cat2Label,
        #get the image and apply the data augmentation process by resizing it to 224*224; fix the image variations radomly (rotate the image, fix the zooming, translate its location and add black and white)
        transform=transformTrain,
        #number of frame taken from each video per input
        numberOfFrames=numberOfFrames
    )

    #same for the validation set
    val_dataset = ImageDataset2(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cat2Label,
        transform=transformValidation,
        numberOfFrames=numberOfFrames)
    


    
    return train_dataset, val_dataset



class ImageDataset2(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None,numberOfFrames=5):
        self.imgs,self.classDistribution,self.numberOfVideos,self.labeledVideo,self.weights = load_imgs_tsn2(video_root, video_list,rectify_label,numberOfFrames)
        self.transform = transform
        
    def __getitem__(self, index):
        toReturn=[]
        a=[]
        #get the 3 images path and label
        for Path in self.imgs[index]:
            path_first, target_first,index = Path
            a.append(path_first)
            #open the image then convert it to RGB
            img_first = Image.open(path_first).convert("RGB")
            #apply the flip, scale to the image if exists
            if self.transform is not None:
                img_first = self.transform(img_first)

            toReturn.append(img_first)
        #return the image with the label and (a is the path of the image) and the index (map the images to the video)
        return toReturn,target_first,a,index

    def __len__(self):
        return len(self.imgs)

def load_imgs_tsn2(video_root, video_list, rectify_label,numberOfFrames2=10):
    img_paths=list()
    classDistribution=[]
    index=0
    labelVideo=[]
    weight=[]
    #get the video root and open the video list
    with open(video_list, 'r') as imf:
        index = -1
        count=0
        for id, line in enumerate(imf):
            index+=1
            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video
            
            labelVideo.append(label)
            #print(video_root)
            #print(video_name)
            video_path = video_root+"/" + video_name # video_path is the path of each video
            #print(video_path)  
            #give us all the images name
            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video
            #get the nb of parts
            num_per_part = int(img_count) // numberOfFrames2
            toReturn=[]
            #repeat the process proportionally to the nb of images
            numberOfTimes=1

            #make sure that I have enough images for the process
            if int(img_count) > numberOfFrames2:
                for m in range(numberOfTimes):
                    for i in range(numberOfFrames2):
                        #get a random number based on the parts
                        rand=random.randint(i*num_per_part,(i+1)*num_per_part-1)
                        #concatinate the path with the image name then save it
                        toReturn.append((video_path+"/"+ img_lists[rand],label,index))
                        weight.append(label)

                    img_paths.append(toReturn)
                    toReturn=[]
            else:
                #get the nb of missing images
                #print("number oOF Frames",numberOfFrames)
                #print("img_count oOF Frames",img_count)
                #print("video_path",video_path)

                missingImages=numberOfFrames2-int(img_count)
                #fullfil the catched images
                for j in range(img_count):
                    toReturn.append((video_path+"/"+ img_lists[j],label,index))
                #duplicate random image to fullfil the missing nb
                for t in range(missingImages):
                    rand=random.randint(0,len(toReturn)-1)
                    toReturn.insert(rand, toReturn[rand])

                img_paths.append(toReturn)
                weight.append(label)
    #return the list of all the paths of images
    return img_paths,classDistribution,index+1,np.array(labelVideo),weight





def crossvalidationdataset(trainpaths, validatepaths, batchsize_train, batchsize_eval,numberOfFrames,cat2Label, transformTrain,transformValidation):
    train_dataset = ImageDataset3(
        path=trainpaths,
        #convert the emotion strings to nb
        rectify_label=cat2Label,
        #get the image and apply the data augmentation process by resizing it to 224*224; fix the image variations radomly (rotate the image, fix the zooming, translate its location and add black and white)
        transform=transformTrain,
        #number of frame taken from each video per input
        numberOfFrames=numberOfFrames
    )

    #same for the validation set
    val_dataset = VideoDataset3(
        path=validatepaths,
        rectify_label=cat2Label,
        transform=transformValidation,
        numberOfFrames=numberOfFrames)
    
    #give the loader the dataset, the batch size, shuffle(to shuffle the emotions), num_workers(nb of cpu to be used to work on the dataset) and the pin_memory (to cache to images to avoid reading them everytime)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True)
    #same for the validation set
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

class ImageDataset3(data.Dataset):
    def __init__(self, path, rectify_label=None, transform=None,numberOfFrames=5):
        self.imgs,self.classDistribution,self.numberOfVideos,self.labeledVideo,self.weights = load_imgs_tsn3(video_root, video_list,rectify_label,numberOfFrames)
        self.transform = transform
        
    def __getitem__(self, index):
        toReturn=[]
        a=[]
        #get the 3 images path and label
        for Path in self.imgs[index]:
            path_first, target_first,index = Path
            a.append(path_first)
            #open the image then convert it to RGB
            img_first = Image.open(path_first).convert("RGB")
            #apply the flip, scale to the image if exists
            if self.transform is not None:
                img_first = self.transform(img_first)

            toReturn.append(img_first)
        #return the image with the label and (a is the path of the image) and the index (map the images to the video)
        return toReturn,target_first,a,index

    def __len__(self):
        return len(self.imgs)

def load_imgs_tsn3(path rectify_label,numberOfFrames=5):
    img_paths=list()
    classDistribution=[]
    index=0
    labelVideo=[]
    weight=[]
    #get the video root and open the video list
    for id, (videopath,label) in enumerate(path):
            index+=1
    
            label = rectify_label[label]  # label of video
            
            labelVideo.append(label)
            #print(video_root)
            #print(video_name)
            video_path = videopath # video_path is the path of each video
            #print(video_path)  
            #give us all the images name
            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video
            #get the nb of parts
            num_per_part = int(img_count) // numberOfFrames
            toReturn=[]
            #repeat the process proportionally to the nb of images
            numberOfTimes=int((img_count/numberOfFrames))

            #make sure that I have enough images for the process
            if int(img_count) > numberOfFrames:
                for m in range(numberOfTimes):
                    for i in range(numberOfFrames):
                        #get a random number based on the parts
                        rand=random.randint(i*num_per_part,(i+1)*num_per_part-1)
                        #concatinate the path with the image name then save it
                        toReturn.append((video_path+"/"+ img_lists[rand],label,index))
                        weight.append(label)

                    img_paths.append(toReturn)
                    toReturn=[]
            else:
                #get the nb of missing images
                #print("number oOF Frames",numberOfFrames)
                #print("img_count oOF Frames",img_count)
                #print("video_path",video_path)

                missingImages=numberOfFrames-int(img_count)
                #fullfil the catched images
                for j in range(img_count):
                    toReturn.append((video_path+"/"+ img_lists[j],label,index))
                #duplicate random image to fullfil the missing nb
                for t in range(missingImages):
                    rand=random.randint(0,len(toReturn)-1)
                    toReturn.insert(rand, toReturn[rand])

                img_paths.append(toReturn)
                weight.append(label)
    #return the list of all the paths of images
    return img_paths,classDistribution,index+1,np.array(labelVideo),weight

class VideoDataset3(data.Dataset):
    def __init__(self, path, rectify_label=None, transform=None,numberOfFrames=5):
        self.imgs,self.classDistribution,self.numberOfVideos,self.labeledVideo,self.weights = load_videos(video_root, video_list,rectify_label,numberOfFrames)
        self.transform = transform
        
    def __getitem__(self, index):
        toReturn=[]
        a=[]
        #get the 3 images path and label
        for Path in self.imgs[index]:
            path_first, target_first,index = Path
            a.append(path_first)
            #open the image then convert it to RGB
            img_first = Image.open(path_first).convert("RGB")
            #apply the flip, scale to the image if exists
            if self.transform is not None:
                img_first = self.transform(img_first)

            toReturn.append(img_first)
        #return the image with the label and (a is the path of the image) and the index (map the images to the video)
        return toReturn,target_first,a,index

    def __len__(self):
        return len(self.imgs)

def load_videos(path rectify_label,numberOfFrames=5):
    img_paths=list()
    classDistribution=[]
    index=0
    labelVideo=[]
    weight=[]
    #get the video root and open the video list
    for id, (videopath,label) in enumerate(path):
            index+=1
    
            label = rectify_label[label]  # label of video
            
            labelVideo.append(label)
            #print(video_root)
            #print(video_name)
            video_path = videopath # video_path is the path of each video
            #print(video_path)  
            #give us all the images name
            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video
            #get the nb of parts
            num_per_part = int(img_count) // numberOfFrames
            toReturn=[]
            #repeat the process proportionally to the nb of images
            numberOfTimes=int((img_count/numberOfFrames))

            #make sure that I have enough images for the process
            if int(img_count) > numberOfFrames:
                for m in range(numberOfTimes):
                    for i in range(numberOfFrames):
                        #get a random number based on the parts
                        rand=random.randint(i*num_per_part,(i+1)*num_per_part-1)
                        #concatinate the path with the image name then save it
                        toReturn.append((video_path+"/"+ img_lists[rand],label,index))
                        weight.append(label)

                    img_paths.append(toReturn)
                    toReturn=[]
            else:
                #get the nb of missing images
                #print("number oOF Frames",numberOfFrames)
                #print("img_count oOF Frames",img_count)
                #print("video_path",video_path)

                missingImages=numberOfFrames-int(img_count)
                #fullfil the catched images
                for j in range(img_count):
                    toReturn.append((video_path+"/"+ img_lists[j],label,index))
                #duplicate random image to fullfil the missing nb
                for t in range(missingImages):
                    rand=random.randint(0,len(toReturn)-1)
                    toReturn.insert(rand, toReturn[rand])

                img_paths.append(toReturn)
                weight.append(label)
    #return the list of all the paths of images
    return img_paths,classDistribution,index+1,np.array(labelVideo),weight