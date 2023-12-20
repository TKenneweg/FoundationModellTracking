import numpy as np
from cleanutil import getSubFolders
import os
from cleanutil import *
from matplotlib import pyplot as plt

def getManualGTCenter(folder):
    with open(f"{folder}/GTCenter.txt") as f:
        line = f.readline()
    center = line.split(",")
    center = [float(c) for c in center]
    return np.asarray(center)


def checkManualCenters(folders):
    for i,folder in enumerate(folders):
        dinoPS = 14
        center= getManualGTCenter(folder)
        # center = center * dinoPS + dinoPS/2
        print(f"{center}")
        imgfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        imgfiles.sort()
        img = getAndResizeImgFromDiskParams(f"{folder}/{imgfiles[0]}", WIDTH ,HEIGHT)
        img = paintPixelCoordPatchonImage(img, center[0], center[1], dinoPS)
        #change bgr to rgb
        img = img[:,:,::-1]
        plt.imshow(img)
        plt.show()


def genGTCentersFromBBs(folders):
    for folder in folders:
        with open(f"{folder}/groundtruth.txt") as f:
            box = f.readline()
        box = box.split(",")
        box = [float(c) for c in box]

        x = box[0] + box[2]/2
        y = box[1] + box[3]/2

        xfactor ,yfactor = getFolderxyFactor(folder)
        x = x * xfactor
        y = y * yfactor

        #write to file
        with open(f"{folder}/GTCenter.txt", "w") as f:
            f.write(f"{y},{x}")



if __name__ == "__main__":
    print("Hello World!")
    folders = getSubFolders("val")
    start = 172
    n =1
    folders = folders[start-1:start-1 + n]
    # genGTCentersFromBBs(folders) #WARNING THIS WILL ERASE ALL EXISTING GTCENTERS!
    checkManualCenters(folders)

    #save centers to file
    # with open("GTCenters_manual.txt", "w") as f:
    #     for center in centers:
    #         f.write(f"{center[0]},{center[1]}\n")



#folders with changed gtcenters:

#2,44,46,48,60,63,74,81,99,100,116,117,151,153,158,164-done!!!

#60 has bad bounding box