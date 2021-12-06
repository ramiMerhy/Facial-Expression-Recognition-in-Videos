import pyautogui
import cv2
import numpy as np
import os
import time
import ruamel.yaml
import networks
import torchvision.transforms as transforms
import torch
import matplotlib
import dlib 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from PIL import Image

init=[]
framesave=[]
peporcessing=[]
net=[]
totaltime=[]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else print("wtf i dont wanna run on cpu"))

def face_alignment(imageTaken,detector,cnn_face_detector,sp):
    # Load the image using OpenCV
    bgr_img = imageTaken
    if bgr_img is None:
        print("Sorry, we could not load '{}' as an image".format(face_file_path))
        exit()

    # Convert to RGB since dlib uses RGB images
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    save_img = img.copy()
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    ''' traditional method '''
    dets = detector(img, 1)
    if len(dets) == 0:
        # first use cnn detector (to detect the face)
        dets = apply_cnn_detection(img,cnn_face_detector)
        if len(dets) == 0:
            #''' Linear '''
            #image intensity history equalization (take the highest pixels value to enhance the quality of the image)
            img = LinearEqual(img)
            dets = apply_cnn_detection(img,cnn_face_detector)
            if len(dets) == 0:
                #''' clahe '''
                #target specific regions to enhance the image to catch the face
                img = claheColor(img)
                dets = apply_cnn_detection(img,cnn_face_detector)
                if len(dets) == 0:
                    #''' Histogram_equalization '''
                    #same as above but with different interpolation method
                    img = hisEqulColor(img)
                    dets = apply_cnn_detection(img,cnn_face_detector)
                    if len(dets) == 0:
                        return None

    # Find the 5 face landmarks (eyes, noise, ...) we need to do the alignment.
    faces = dlib.full_object_detections()

    for detection in dets:
        faces.append(sp(img, detection))
    #rotate the image to be straight and specify the image size
    image = dlib.get_face_chip(save_img, faces[0], size=224, padding=0.25)
    #change the colors from RGB to BGR
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv_bgr_img

def claheColor(img):
    #YCRcb: change color space (brightness ....)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    #devide the channels using the openCV library
    channels = cv2.split(ycrcb)
    #use create clahe (work on specific region)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.split(ycrcb)
    #use history equalization
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

#different interpolation method
def LinearEqual(image):
    lut = np.zeros(256, dtype = image.dtype )
    hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])
    minBinNo, maxBinNo = 0, 255

    #get the minimum and maximum value
    #binValue: the ranges of value
    for binNo, binValue in enumerate(hist):
        if binValue != 0:
            minBinNo = binNo
            break
    for binNo, binValue in enumerate(reversed(hist)):
        if binValue != 0:
            maxBinNo = 255-binNo
            break
    for i,v in enumerate(lut):
        #set the minimum values to 0
        if i < minBinNo:
            lut[i] = 0
        #set the maximum values to 0
        elif i > maxBinNo:
            lut[i] = 255
        else:
            #apply the linear interpolation to get the new pixel value
            lut[i] = int(255.0*(i-minBinNo)/(maxBinNo-minBinNo)+0.5)   # why plus 0.5
    return cv2.LUT(image, lut)

#funtion to catch the face in the image
def apply_cnn_detection(img,cnn_face_detector):
    #cnn_face_detector is a lib that catch the face
    cnn_dets = cnn_face_detector(img, 1)
    #add the list of rectangle to rectangles
    dets = dlib.rectangles()
    dets.extend([d.rect for d in cnn_dets])
    return dets


idfolder=0

idpath="path"+str(idfolder)
folder="saveVids"

pathToSave=os.path.join(folder,idpath)

if not os.path.exists(pathToSave):
    os.makedirs(pathToSave)

idImg=0

predictor_path      = 'shape_predictor_5_face_landmarks.dat'
cnn_face_detector   = 'mmod_human_face_detector.dat'

detector = dlib.get_frontal_face_detector()
#load the network weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector)
sp = dlib.shape_predictor(predictor_path)

networkStructure=parameters['network3']
#create the network
model = networks.Network(networkStructure)
#load the network weights
model.load_state_dict(torch.load(parameters['loadnet_path']))
if torch.cuda.is_available():
    model.cuda()

#evaluating not training
model.eval()

transformValidation=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
frames=[]

good=0
#create the object that I need to read
vidcap = cv2.VideoCapture(video_path)
#take one frame of the video
success,image = vidcap.read()
count = 0

for i in range(1000):
    while success:
        #save the image
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file  
        #change the image to numpy array
        frame = np.array(image)
        print('Read a new frame: ', success)
        count += 1
        #crop, rotate and find the face of the image
        frame=face_alignment(frame,detector,cnn_face_detector,sp)

        try:
            #revert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("face not found")
            continue

        frame=transformValidation(Image.fromarray(frame))
        frames.append(frame)
        idImg+=1
        good+=1
        #take one frame of the video
        success,image = vidcap.read()

good=0

#no_grad: no training and no loss
with torch.no_grad():
    #convert from a python list to a tensor
    input_var=torch.stack(frames, dim=3)
    #add dimension (add a batch size of 1: for example 1*30*3*224*224)
    input_var=input_var.unsqueeze(0)
    #change input to GPU if exist
    input_var = input_var.to(DEVICE)
    #output of the network
    pred_score = model(input_var)
    #finding emotions based on max value
    emotion=torch.argmax(pred_score)
    #changing scores to probabilities
    probabilities=F.softmax(pred_score, dim=0).cpu().numpy()

print(probabilities)