import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
import argparse
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import load

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

#check if we have a gpu to use it
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else print("wtf i dont wanna run on cpu"))

#batch of images for training
batchsize_train=36
#batch of images for evaluation
batchsize_eval=1
#number of frame taken from each video per input
numberOfFrames=3
#learing rate (change the step size and it can be specified by a trial error process)
lr=0.00006
#nb of times that we repeat our training on the data set
epochs=200

#check how many time the loss doesn't change and then stop training
stoppingCriteria=10
#convert the emotion strings to nb
cat2Label={ "Happy": 0,"Angry": 1,  "Disgust": 2, "Fear": 3,"Sad": 4,"Neutral": 5,"Surprise": 6}

#get the image and apply the data augmentation process by resizing it to 224*224; fix the image variations radomly (rotate the image, fix the zooming, translate its location and add black and white)
transformTrain=transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),transforms.RandomAffine(10, translate=[0.1,0.1], scale=[0.9, 1.25], shear=0.01), transforms.ToTensor()])
#get the image and apply the data augmentation process by resizing it to 224*224; tensor: array in pytoch
transformValidation=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

root_train = './data/face/train_afew'
list_train = './data/txt/afew_All.txt'

root_eval = './data/face/val_afew'
list_eval= './data/txt/afew_eval.txt'
# load the data set
train_loader, val_loader = load.frames3average(root_train, list_train, batchsize_train, root_eval, list_eval, batchsize_eval,numberOfFrames,cat2Label, transformTrain,transformValidation)

#the train function get the images, the structure that we want to optimize, the optimizer, the loss function and the epoch
def train(train_loader, model, optimizer,loss_fn, epoch):
    #average meter: the sum values to create the elements
    losses = load.AverageMeter()
    accuaracy = load.AverageMeter()
    #tell my model that we are training so the weights will change
    model.train()
    #input_var: the entered arrage; target_var: the label(emotion); path:path of the video; index:specify the index to create a map of the videos
    for i, (input_var, target_var,path,index) in enumerate(train_loader):
        
        target_var = target_var.float().to(DEVICE)
        #convert the input_vat from a python list to a pytorch tensor
        input_var=torch.stack(input_var,4)

        input_var = input_var.to(DEVICE)

        #Feed forward command (go from the input to the label)
        pred_score = model(input_var)
        
        #loss and accuaracy computation
        loss = loss_fn(pred_score, target_var.long()).sum()
        acc_iter = discreteAccuracy(pred_score.data, target_var)
        #update the values after every training
        losses.update(loss.item(), input_var.size(0))
        accuaracy.update(acc_iter, input_var.size(0))
    
    	#Backpropagation process (changing the weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == (len(train_loader)-1):
            print('Epoch: [{:3d}][{:3d}/{:3d}]\t'
                  'Train Loss {loss.avg:.4f}\t'
                  'Train Accuaracy {accuaracy.avg:.3f}\t'
                .format(
                epoch, i, len(train_loader), loss=losses, accuaracy=accuaracy))
    
    return losses.avg,accuaracy.avg


def discreteAccuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # first position is score; second position is pred.
    pred = pred.t()  # .t() is T of matrix (256 * 1) -> (1 * 256)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target.view(1,2,2,-1): (256,) -> (1, 2, 2, 64)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0]


def val(val_loader, model,loss_fn,classes):
    
    lossesVal = load.AverageMeter()
    accuaracyVal = load.AverageMeter()

    #tell the code that we don't need to change the weight
    model.eval()
    #tell the code to not compute the gradiant
    with torch.no_grad():
        for i, (input_var, target_var,path,index) in enumerate(val_loader):          
            target_var = target_var.float().to(DEVICE)
        
            input_var=torch.stack(input_var,4)
           
            input_var = input_var.to(DEVICE)
            pred_score = model(input_var)

            loss = loss_fn(pred_score, target_var.long()).sum()
            acc_iter = discreteAccuracy(pred_score.data, target_var)
            
            lossesVal.update(loss.item(), input_var.size(0))
            accuaracyVal.update(acc_iter, input_var.size(0))

            if i == (len(val_loader)-1):
                print('Loss Validation  {loss.avg:.4f}\t'
                      'Validation accuaracy {accuaracy.avg:.3f}\t'
                    .format(loss=lossesVal, accuaracy=accuaracyVal))
    

    return accuaracyVal.avg,lossesVal.avg

#define my loss function
loss_fn = nn.CrossEntropyLoss()
predictionsForPlot=[]

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


#initialize model
_structure=resnet18_at(512)
_parameterDir = "./pretrain_model/Resnet18_FER+_pytorch.pth.tar"
#load pretrained weight from google
model = model_parameters(_structure, _parameterDir)
model.classifier=nn.Sequential(nn.Linear(512,7))
   

#inititalize optimmizer
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=0.9, weight_decay=1e-4)
#reduce to 0.2 after 1000 step
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.2)
      
best_prec1=0

predicitionsval=[]
predictionstrain=[]
lossestrain=[]
lossesval=[]

#how many times we need to repeat the training
for epoch in range(epochs):
    print("training at ", epoch)
    ######Traind and validating###########################

    print("length of train loader",len(train_loader))
    #call the training function
    loss_train,train_accuaracy=train(train_loader, model, optimizer,loss_fn, epoch)

    print("length of validaiton loader",len(val_loader))
    #call the validation function
    validationAccuaracy,lossValidation = val(val_loader, model,loss_fn,cat2Label)
    lossestrain.append(loss_train)
    lossesval.append(lossValidation)
    predictionstrain.append(train_accuaracy)
    predicitionsval.append(validationAccuaracy)

    ######Savind Best Model###############################
    is_best = validationAccuaracy > best_prec1
    if is_best:
        worseCount=0
        print('better model!\n\n')
        pathModelSave="networksave/"+"net_"+str(best_prec1)
        torch.save(model.state_dict(), pathModelSave)
    else:
        print("model didnt improve \n\n")
        worseCount+=1

    lr_scheduler.step()
    if(worseCount==stoppingCriteria):
       print("training done\n\n\n")
       break