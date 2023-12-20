import torch
import os
from cleanutil import * 
from metrics import *
from segment_anything import SamPredictor, sam_model_registry
from transformers import ResNetModel

class Tracker():
    def __init__(self, backbonetype):
        self.backbonetype = backbonetype #options are dino, sam and resnet
        self.width = 1274
        self.height = 714
        self.patch_size = 0
        self.w_emb= 0
        self.h_emb =0
        self.c_emb = 0
        if backbonetype == "dino":
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            self.patch_size = 14
            self.w_emb= 91
            self.h_emb =51
            self.c_emb = 384
        if backbonetype == "sam":
            sam = sam_model_registry["vit_h"](checkpoint="./models/sam_vit_h_4b8939.pth")
            self.backbone = SamPredictor(sam)
            self.patch_size = 16
            self.w_emb = 64
            self.h_emb = 64 #gets squared internally by sam
            self.c_emb = 256

        if backbonetype == "resnet":
            self.backbone = ResNetModel.from_pretrained("microsoft/resnet-50")
            self.backbone.eval()
            self.patch_size_i = 31.04
            self.patch_size_j = 31.85
            self.w_emb = 40
            self.h_emb = 23 
            self.c_emb = 2048        


    def to_cuda(self):
        if self.backbonetype == "dino":
            self.backbone.to("cuda")
            self.backbone = torch.compile(self.backbone)
        if self.backbonetype == "sam":
            self.backbone.model.to("cuda")
        if self.backbonetype == "resnet":
            self.backbone.to("cuda")

    def checkfolders(self, folder):
        if not os.path.isdir(f"{folder}/tracking"):
            os.mkdir(f"{folder}/tracking")
        if not os.path.isdir(f"{folder}/tracking/{self.backbonetype}"):
            os.mkdir(f"{folder}/tracking/{self.backbonetype}") 
        if not os.path.isdir(f"{folder}/tracking/{self.backbonetype}/embeddings"):
            os.mkdir(f"{folder}/tracking/{self.backbonetype}/embeddings") 
        if not os.path.isdir(f"{folder}/tracking/{self.backbonetype}/frames"):
            os.mkdir(f"{folder}/tracking/{self.backbonetype}/frames") 


    def generateEmbeddings(self, folder):
        imgfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        imgfiles.sort()
        for i,file in enumerate(imgfiles):
            img = numpyToTorchCPU(getAndResizeImgFromDiskParams(f"{folder}/{file}", self.width, self.height))
            img = img.to("cuda")
            if self.backbonetype == "dino":
                patchembedding = self.backbone.model(img,is_training=True)
                patchembedding = patchembedding["x_norm_patchtokens"].squeeze().to("cpu") #height and width come in the first dimesion
                patchembedding = patchembedding.reshape(self.h_emb, self.w_emb, self.c_emb)
                patchembedding = patchembedding.permute(2,0,1)#channel,  height, width
            if self.backbonetype == "sam":
                img = img[0].permute(1,2,0)
                img = (img*255).to("cpu").numpy().astype(np.uint8)
                patchembedding = self.backbone.set_image(img)
                patchembedding = self.backbone.features.to("cpu").squeeze(0)
            if self.backbonetype == "resnet":
                patchembedding = self.backbone(img)["last_hidden_state"].squeeze(0)


            torch.save(patchembedding.clone(), f"{folder}/tracking/{self.backbonetype}/embeddings/{i}.pt")

        print(f"Saved embeddings to {folder}/tracking/{self.backbonetype}/embeddings/{i}.pt")



    
    def generateTrackingIndices(self, folder):
        GTCenter = self.getManualGTCenter(folder) #this is in resized image coordinates
        if self.backbonetype == "sam":
            GTCenter = GTCenter * self.getSAMScale(folder)
        if self.backbonetype != "resnet":
            GTCenter /= self.patch_size #convert to patch coordinates
        else: 
            GTCenter[0] /= self.patch_size_i
            GTCenter[1] /= self.patch_size_j
        GTCenter = GTCenter.astype(int)
        embpath = f"{folder}/tracking/{self.backbonetype}/embeddings"
        embfiles = [f for f in os.listdir(embpath) if f.endswith(".pt")]
        embfiles.sort(key=lambda x: int(x.split('.')[0]))
        # print(embfiles)

        emb0 = torch.load(f"{embpath}/0.pt")
        tp = emb0[:,GTCenter[0],GTCenter[1]] #tracking patch
        #repeat tp along the second and third dimension p_shape[0] and p_shape[1] times
        tpmatrix = tp.repeat(self.h_emb, self.w_emb,1) # for efficient distance calculation
        tpmatrix = tpmatrix.permute(2,0,1) #permute to embedding,h,w (does not work directly with repeat)
        # print(torch.allclose(tp, tpmatrix[:,10,20]))

        imgfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        imgfiles.sort()

        #add first tracking center
        trackingCenters = [] #in resized image coordinates
        # trackingCenters.append(GTCenter* self.patch_size + self.patch_size/2 )
        for i,embfile in enumerate(embfiles):
            emb = torch.load(f"{embpath}/{embfile}")
            ind = self.getNearestPatch(tpmatrix, emb)
            if self.backbonetype != "resnet":
                trackingCenters.append(ind * self.patch_size + self.patch_size/2)
            else:
                ind[0]  = ind[0] * self.patch_size_i + self.patch_size_i/2
                ind[1] =  ind[1] * self.patch_size_j + self.patch_size_j/2
                trackingCenters.append(ind )



        torch.save(trackingCenters, f"{folder}/tracking/{self.backbonetype}/trackingCenters.pt")
        print(f"Saved tracking centers to {folder}/tracking/{self.backbonetype}/trackingCenters.pt")
        

    #for new text file in folder. Might be manually edited in case the initial center
    #is not on the object
    def getManualGTCenter(self,folder):
        with open(f"{folder}/GTCenter.txt") as f:
            line = f.readline()
        center = line.split(",")
        center = [float(c) for c in center]
        return np.asarray(center)

    #ugly but just dont touch it
    ############################################
    def getNearestPatch(self, tpmatrix, imgEmbeddings):
        #calculate distance between tracking patch and all patches in the image
        # print(tpmatrix.shape)
        # print(imgEmbeddings.shape)
        # embeddingDists = getEuclideanDistance(imgEmbeddings, tpmatrix)
        # embeddingDists = getManhattenDistance(imgEmbeddings, tpmatrix)
        embeddingDists = getCosineSimilarityDistance(imgEmbeddings, tpmatrix)
        embdists, indices = getSmallestNNumbersWithIndicesUnderThreshold(embeddingDists,n=1,thres=embeddingDists[0,0])
        return indices[0]
    
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
    ############################################
    

    def genTrackingImagesAndVideo(self, folder):
        try: 
            trackingCenters = torch.load(f"{folder}/tracking/{self.backbonetype}/trackingCenters.pt")
        except:
            print("Failed to load tracking centers")
            return
        if self.backbonetype == "sam":
            trackingCenters = [center /self.getSAMScale(folder) for center in trackingCenters]
        #generate images with tracking patch and bounding box
        imgfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        # imgfiles.sort()
        imgfiles = sorted(imgfiles, key=lambda x: int(x.split(".")[0]))
        boxes = self.getGTBoxes(folder)
        for center, box, imgfile in zip(trackingCenters,boxes, imgfiles):
            #load resize factor from file
            img = getAndResizeImgFromDiskParams(f"{folder}/{imgfile}", self.width, self.height)
            img = drawBoundingBox(img, box)
            img = paintPixelCoordPatchonImage(img, center[0], center[1], self.patch_size_i)
            #convert to BGR
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #save img to disc
            cv2.imwrite(f"{folder}/tracking/{self.backbonetype}/frames/{imgfile}", img)
        #generate video
        self.genVideo(f"{folder}")
        print(f"Generated video for {folder}")

    def genVideo(self,folder):
        video_name = f"{folder}/tracking/{self.backbonetype}/trackingVideo.avi"
        images = [img for img in os.listdir(f"{folder}/tracking/{self.backbonetype}/frames")
                if img.endswith(".jpg") or
                    img.endswith(".jpeg") or
                    img.endswith("png")]
        images = sorted(images, key=lambda x: int(x.split(".")[0]))

        
        # setting the frame width, height width
        # the width, height of first image
        frame = cv2.imread(f"{folder}/tracking/{self.backbonetype}/frames/{images[0]}")
        height, width, layers = frame.shape  
        fps = 15
        video = cv2.VideoWriter(video_name, 0, fps, (width, height)) 
    
        # Appending the images to the video one by one
        for image in images: 
            video.write(cv2.imread(f"{folder}/tracking/{self.backbonetype}/frames/{image}")) 
        
        # Deallocating memories taken for window creation
        cv2.destroyAllWindows() 
        video.release()  # releasing the video generated    

    #returns gtboxes in resized image coordinates
    def getGTBoxes(self,folder):
        xfactor ,yfactor = self.getFolderxyFactor(folder)
        with open(f"{folder}/groundtruth.txt") as f:
            lines = f.readlines()
        gtboxes = []
        for line in lines:
            box = line.split(",")
            box = [float(b) for b in box]
            gtboxes.append(box)

        gtboxes = torch.tensor(gtboxes)
        #convert according to dingsda
        gtboxes[:,0] = gtboxes[:,0] * xfactor# * self.getSAMScale(folder)
        gtboxes[:,2] = gtboxes[:,2] * xfactor# * self.getSAMScale(folder)
        gtboxes[:,1] = gtboxes[:,1] * yfactor #* self.getSAMScale(folder)
        gtboxes[:,3] = gtboxes[:,3] * yfactor# * self.getSAMScale(folder)

        tmplist = []
        for i in range(gtboxes.shape[0]):
            tmp = [int(gtboxes[i,0]), int(gtboxes[i,1]), int(gtboxes[i,2]), int(gtboxes[i,3])]
            tmplist.append(tmp)
        return tmplist 
    



    def genEvaluation(self,folder):
          #read indices.pt
        centers = torch.load(f"{folder}/tracking/{self.backbonetype}/trackingCenters.pt")
        if self.backbonetype == "sam":
            centers = [center /self.getSAMScale(folder) for center in centers]
        GTboxes = self.getGTBoxes(folder)
            
        nt = 0
        for center, box in zip(centers,GTboxes):
            if centerInBox(center, box):
                nt += 1

        imgfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        successrate = nt/len(imgfiles) #divide by len(imgfiles instead of len(centers) because its less error prone
        print(f"Successrate for {folder}: {successrate}")
        #save successrate as txt
        with open(f"{folder}/tracking/{self.backbonetype}/successrate.txt", "w") as f:
            f.write(f"{successrate}")


    def printAverageSuccessrate(self,folders):
        successrates = []
        for folder in folders:
            with open(f"{folder}/tracking/{self.backbonetype}/successrate.txt") as f:
                successrate = float(f.readline())
                print(successrate)
                successrates.append(successrate)
        avg = sum(successrates)/len(successrates)

        print(f"Average successrate for backbone {self.backbonetype}: ", avg)


    def getFolderxyFactor(self,folder):
        imgfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        img = cv2.imread(f"{folder}/{imgfiles[0]}")
        yfactor = self.height/img.shape[0] #height
        xfactor = self.width/img.shape[1] #width
        return xfactor, yfactor
    

        #internally sam scales images so that the longest side is 1024, for this it applies this formula
    def getSAMScale(self,folder):
        scale = 1024 * 1.0 / max(self.height, self.width)
        return scale
    