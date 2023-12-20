import os
import torch
import cv2
import numpy as np
from config import *

#we resize to have a maximum img height of 720
def getFolderResizeFactor(folder):
    imgfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    img = cv2.imread(f"{folder}/{imgfiles[0]}")
    factor = 720/img.shape[0] #height
    return factor


def getFolderxyFactor(folder):
    imgfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    img = cv2.imread(f"{folder}/{imgfiles[0]}")
    yfactor = HEIGHT/img.shape[0] #height
    xfactor = WIDTH/img.shape[1] #width
    return xfactor, yfactor

def numpyToTorchCPU(img):
    return (torch.from_numpy(img).permute(2,0,1)/255.0).unsqueeze(0)


import time
def getAndResizeImgFromDisk(imgname):
    img = cv2.imread(imgname)
    img = cv2.resize(img, dsize = (WIDTH,HEIGHT))
    return img

def getAndResizeImgFromDiskParams(imgname, width, height):
    img = cv2.imread(imgname)
    img = cv2.resize(img, dsize = (width,height))
    return img


# i and j are floats, since we use pixel coordinates in original image(resized to hd ready)
# i and j denote the middle of the patch in pixel coordinates
def paintPixelCoordPatchonImage(img, i,j, ps):
    img = img.copy()
    if int(i +ps/2) <= img.shape[0] and int(j+ps/2) <= img.shape[1]:
        img[int(i-ps/2):int(i+ps/2), int(j-ps/2):int(j+ps/2)] = (0,0,255)
    else:
        print(f"Patch {i},{j} is out of bounds in paintPatchonImage")
    return img


#a function to draw a bounding box on an image
def drawBoundingBox(img, box):
    img = img.copy()
    box = [int(b) for b in box]
    xmin, ymin, width, height = box
    xmax = xmin + width
    ymax = ymin + height
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
    return img


#a function to draw a bounding box on an image
def drawBoundingBoxColor(img, box, color):
    img = img.copy()
    box = [int(b) for b in box]
    xmin, ymin, width, height = box
    xmax = xmin + width
    ymax = ymin + height
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax),color, 3)
    return img


@torch.no_grad()
def batch_ious(boxes1, boxes2):
    """
    Compute IoU between two batches of boxes.
    Boxes are represented with (xmin, ymin, width, height).

    Args:
    boxes1, boxes2: (N, 4) PyTorch tensors
    
    Returns:
    iou: (N,) PyTorch tensor
    """
    # Calculate the coordinates of the corners of the intersection boxes
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 0] + boxes1[:, 2], boxes2[:, 0] + boxes2[:, 2])
    y2 = torch.min(boxes1[:, 1] + boxes1[:, 3], boxes2[:, 1] + boxes2[:, 3])

    # Calculate the area of the intersection
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate the area of each input box
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    # Calculate the area of the union
    union = area1 + area2 - intersection

    # Compute the IoU
    iou = intersection / union
    return iou

#return the image shape given a folder with images
def getResizedImgShape(folder):
    imgfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    #get first image to determine if a resize is necessary
    img = getAndResizeImgFromDisk(f"{folder}/{imgfiles[0]}")
    return np.asarray(img.shape)

#centers are really centered by half patch size
def centerInBox(center, box):
    xmin, ymin, width, height = box
    if center[0] >= ymin and center[0] <= ymin + height and center[1] >= xmin and center[1] <= xmin + width:
        return True
    else:
        return False
    

#returns vector of smallest n numbers in tensor and nx2 matrix of indices of those numbers
def getSmallestNNumbersWithIndicesUnderThreshold(tensor, n, thres):
    orig_shape = tensor.shape
    tensor = tensor.flatten()
    res, indices = torch.topk(tensor, n, largest=False)
    mask = res <= thres
    res = res[mask]
    indices = indices[mask]
    #convert indices back to 2d format
    indices = torch.stack([indices // orig_shape[1], indices % orig_shape[1]], dim=1)
    return res, indices


def getSubFolders(topfolder):
    folders = []
    for folder in os.listdir(topfolder):
        if os.path.isdir(f"{topfolder}/{folder}"):
            folders.append(f"{topfolder}/{folder}")
    folders.sort()
    return folders


def int_to_eight_char_str(num):
    return f"{num:08}"


def unittopixel(box):
    box[0] = box[0] * WIDTH 
    box[2] = box[2] * WIDTH 
    box[1] = box[1] * HEIGHT
    box[3] = box[3] * HEIGHT 
    return box


import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F
from matplotlib.patches import Rectangle

def plotpred(path, label, pred):
    img = torchvision.io.read_image(path)
    img = F.interpolate(img.unsqueeze(0)/255.0, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)[0]
    img = img.permute(1,2,0)
    plt.imshow(img)
    label = unittopixel(label)
    pred = unittopixel(pred)
    plt.gca().add_patch(Rectangle((label[0],label[1]),label[2],label[3],
                        edgecolor='green',
                        facecolor='none',
                        lw=2))
    plt.gca().add_patch(Rectangle((pred[0],pred[1]),pred[2],pred[3],
                        edgecolor='red',
                        facecolor='none',
                        lw=2))
    plt.show()

def plotpredImage(path, label, pred):
    img = torchvision.io.read_image(path)
    img = F.interpolate(img.unsqueeze(0)/255.0, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)[0]
    img = img.permute(1,2,0)
    plt.imshow(img)
    label = unittopixel(label)
    pred = unittopixel(pred)
    plt.gca().add_patch(Rectangle((label[0],label[1]),label[2],label[3],
                        edgecolor='green',
                        facecolor='none',
                        lw=2))
    plt.gca().add_patch(Rectangle((pred[0],pred[1]),pred[2],pred[3],
                        edgecolor='red',
                        facecolor='none',
                        lw=2))
    # plt.show()
    #convert plot to image
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    #clear plot
    plt.clf()

    return img


@torch.no_grad()
def getPatchesInBox(objbox, emb, w_max, h_max, path = None):
    batch_size = objbox.shape[0]

    pwmin = (objbox[:, 0] * w_max).long()
    pwmax = (torch.ceil((objbox[:, 0] + objbox[:, 2]) * w_max)).long()
    phmin = (objbox[:, 1] * h_max).long()
    phmax = (torch.ceil((objbox[:, 1] + objbox[:, 3]) * h_max)).long()

    pwmax = torch.where(pwmax >= w_max, torch.tensor([w_max - 1]).to("cuda"), pwmax)
    phmax = torch.where(phmax >= h_max, torch.tensor([h_max - 1]).to("cuda"), phmax)

    for i in range(batch_size):
        if phmax[i] - phmin[i] == 0:
            if phmax[i] < h_max - 1:
                phmax[i] += 1
            else:
                phmin[i] -= 1

        if pwmax[i] - pwmin[i] == 0:
            if pwmax[i] < w_max - 1:
                pwmax[i] += 1
            else:
                pwmin[i] -= 1

    #visualize the box
    # folderpath = path[0][:path[0].rfind("/")]
    # xfactor ,yfactor = getFolderxyFactor(folderpath)
    # firstimgpath = folderpath + "/00000001.jpg" 
    # print(firstimgpath)
    # print("xfactor: ", xfactor, " yfactor: ", yfactor)
    # img = torchvision.io.read_image(firstimgpath)
    # img = F.interpolate(img.unsqueeze(0)/255.0, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)[0]
    # img = img.permute(1,2,0)

    # pixelbox =  [pwmin[0], phmin[0], pwmax[0]-pwmin[0], phmax[0]-phmin[0]]
    # pixelbox = [pixelbox[i].cpu().numpy() for i in range(len(pixelbox))]
    # pixelbox[0] = pixelbox[0] * WIDTH / w_max
    # pixelbox[2] = pixelbox[2] * WIDTH / w_max
    # pixelbox[1] = pixelbox[1] * HEIGHT /h_max
    # pixelbox[3] = pixelbox[3] * HEIGHT/ h_max
    # print(pixelbox)
    # plt.gca().add_patch(Rectangle((pixelbox[0],pixelbox[1]),pixelbox[2],pixelbox[3],
    #             edgecolor='green',
    #             facecolor='none',
    #             lw=2))
    # plt.imshow(img)
    # plt.show()

    # print("pwmin: ", pwmin)
    # print("pwmax: ", pwmax)
    # print("phmin: ", phmin)
    # print("phmax: ", phmax)
    # print(emb.shape)
    objectemb = [emb[i, :, phmin[i]:phmax[i], pwmin[i]:pwmax[i]] for i in range(batch_size)]
    return objectemb
