import os
import torch
# from SAMTracker import SAMTracker
# from DinoTracker import DinoTracker
from tracker import Tracker
import time
from cleanutil import getSubFolders


#everything in pytorch is batch x channels x height x width
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    foldername = "/home/tristan/dev/dinoPlayground/val"
    folders = getSubFolders(foldername)
    # tracker = Tracker("dino")
    # tracker = Tracker("sam")
    tracker = Tracker("resnet")
    tracker.to_cuda()

    # folders = folders[0:1]
    for folder in folders:
        print(folder)
    #     tracker.checkfolders(folder)
    #     tracker.generateEmbeddings(folder)
        tracker.generateTrackingIndices(folder)
        # tracker.genTrackingImagesAndVideo(folder)
        tracker.genEvaluation(folder)


    tracker.printAverageSuccessrate(folders)
    # avg_success = tracker.saveAverageSuccessrate(folders)
    # print(f"Average successrate: {avg_success}")



    # print(worstFolders)
    # tracker.writeSuccessRatesToSingleFile(folders)



#semi open stuff
#patches are out of bound on GOT-10k_Val_000015! and GOT-10k_Val_000137, GOT-10k_Val_000040, GOT-10k_Val_000074, GOT-10k_Val_000078
#why gets everything so fucking slow? - because the latter pics are large and have to resize