import cv2
import numpy as np
import os
import time

import torchvision.transforms as transforms
import torch
import matplotlib
import tkinter as tk
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from PIL import Image
import torch.nn as nn
import math
init=[]
framesave=[]
peporcessing=[]
net=[]
totaltime=[]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Network definition; convolution process
#in_planes: input size; out_planes: output size; kernel_size: the mid layer (constructed by the W); stride: nb of skipped pixels; padding: how to treat the missing pixels; bias: constant values that we add or not
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#define a block
class BasicBlock(nn.Module):
    expansion = 1
#block: convolution then batch normalization then relu
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    #residual block equation
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#define the structure
class ResNet_AT(nn.Module):
    def __init__(self, block, layers,featureVectoreSize):
        super(ResNet_AT, self).__init__()
        self.inplanes = 64
        #convolution then batch normalization then relu without creating a block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        #max pooling: set the kernel and get the max value
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #defining the layers (64,128,256 are the output size)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, featureVectoreSize, layers[3], stride=2)
        #get the average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Linear(512,7)
        #initials weights based on a normal distribution
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #creating layers and each layer has some blocks then make sure that input and output size of consecutive blocks are equal
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        #change dimention order
        x=x.permute(0,4,1,2,3)
        #check each dimension size
        batch_size, seqlen, nc, h, w = x.size()
        #combine the batch size and sequence lenght together
        x = x.reshape(-1, nc, h, w)

        #apply all the above to get the result (emotion)
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)

        #reduce the 30 values to 10 (each value has 3 emotion scores)
        out = f.reshape(batch_size, seqlen, -1)
        #average the result on the 3 images from the same video
        out=out.mean(1)
        #
        out=self.classifier(out)
        return out

#allow to load the weights of the model
def model_parameters(_structure, _parameterDir):
    #checkpoint: a python dictionary where the weights are saved
    checkpoint = torch.load(_parameterDir)
    #state_dict: are the weights
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    #loading the weights
    _structure.load_state_dict(model_state_dict)
    #load them on the gpu if exists
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(_structure).cuda()
    else:
        model = torch.nn.DataParallel(_structure)

    return model

#define which basic block we need to use; the nb of blocks per layer; nb of layers; output size(stucture)
def resnet18_at(featureVectoreSize, **kwargs):
    # Constructs base a ResNet-18 model.
    model = ResNet_AT(BasicBlock, [2, 2, 2, 2],featureVectoreSize, **kwargs)
    return model


idfolder=0

idpath="path"+str(idfolder)
folder="saveVids"

pathToSave=os.path.join(folder,idpath)

if not os.path.exists(pathToSave):
    os.makedirs(pathToSave)

idImg=0


#create the network
model=resnet18_at(512)

#load the network weights
state_dict = torch.load('network_weigths.zip',map_location=torch.device('cpu'))
#print(state_dict.keys())
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    type = k[:7] # remove `module.`
    if(type=='module.'):
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    #else:
    #    new_state_dict[k] = v
        
model.load_state_dict(new_state_dict)
print("model loading")
if torch.cuda.is_available():
    model.cuda()

#evaluating not training
model.eval()

transformValidation=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
frames=[]

good=0

#add video name
video_path=".."

#import pyautogui
#myScreenshot = pyautogui.screenshot()
#myScreenshot.save(r'ss1.png')
import time
time.sleep(1)
print("start recording")
from PIL import ImageGrab 
im = ImageGrab.grab()
width, height = im.size   # Get dimensions

#left = (width)/4
#top = (height)/4
#right = (width)/4
#bottom = (height)/4


# Crop the center of the image
#im = im.crop((left, top, right, bottom))
#im.save('ss1.png')
success=True


#raise NameError("testing")
count = 0
face_cascade = cv2.CascadeClassifier('detect.xml')
faces_list=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

while success:

    #change the image to numpy array
    frame = np.array(im)
    #plt.imshow(frame)
    #plt.show()
    print('Read a new frame: ', success)
    count += 1
    #crop, rotate and find the face of the image

    
    #try:
        #revert to RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray)
    #plt.show()
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    number_of_faces=0
    print("number of faces" + str(len(faces)))
    print(faces)
    #print(type(faces))
    for i,(x, y, w, h) in enumerate(faces):
        temp2=frame.copy()
        print("face found "+str(i))
        #print(frame)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = temp2[y:y + h, x:x + w]
        #plt.imshow(face)
        #plt.show()
        cv2.imwrite("saveimages/frame%d_face#_%d.jpg" % (count,i), face)
        temp=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        temp=transformValidation(Image.fromarray(temp))
        faces_list[i].append(temp)

    #except:
    #    print("face not found")
    #    print('Whew', sys.exc_info()[0], 'occurred.')


    #    continue
    #raise nameError("found face stop")
    #raise Error("test")
  

    idImg+=1
    good+=1
    #take one frame of the video
    im = ImageGrab.grab()
    print('here')
    print(im)
    width, height = im.size   # Get dimensions

    #left = (width)/4
    #top = (height)/4
    #right = (width)/4
    #bottom = (height)/4

    # Crop the center of the image
    #im = im.crop((left, top, right, bottom))
    #im.save('ss.png')
    success=True
    print(count)
    if(count==3):
        success=False
good=0
faces_list = [x for x in faces_list if x != []]
emotion_str=[]
for frames in faces_list:
    #no_grad: no training and no loss
    with torch.no_grad():
        #convert from a python list to a tensor
        print(len(frames))
        input_var=torch.stack(frames, dim=3)
        #add dimension (add a batch size of 1: for example 1*30*3*224*224)
        input_var=input_var.unsqueeze(0)
        #change input to GPU if exist
        input_var = input_var.to(DEVICE)
        #output of the network
        pred_score = model(input_var)
        #finding emotions based on max value
        emotion=torch.argmax(pred_score).item()
        #changing scores to probabilities
        print(pred_score)
        probabilities=F.softmax(pred_score, dim=1).cpu().numpy()

    print(probabilities)

    print(emotion)
    cat2Label={ "Happy": 0,"Angry": 1, "Disgust": 2, "Fear": 3, "Sad": 4, "Neutral": 5, "Surprise": 6, "0":"happy", "1":"Angry", "2":"Disgust", "3":"Fear", "4":"Sad", "5":"Neutral", "6":"Surprise"}
    emotion_str.append(cat2Label[str(emotion)])
    print("EMOTION IS"+cat2Label[str(emotion)])

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 300)
canvas1.pack()
temp=''
for i in range(len(emotion_str)):
    temp+='image'+str(i)+': '+str(emotion_str[i])+'\n'
                                   
label1 = tk.Label(root, text= temp, fg='green', font=('helvetica', 12, 'bold'))

canvas1.create_window(150, 150, window=label1)

root.mainloop()
