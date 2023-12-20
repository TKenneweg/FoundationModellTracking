from torch.utils.data import Dataset
import os
from cleanutil import *
import torch
from config import *
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import umap

def cluster_and_find_mean(objemb, img = None, objbox =None):
    """
    Function to perform clustering along the channel dimension and return
    the mean of the largest cluster.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape (channel, height, width)

    Returns:
        torch.Tensor: The mean tensor of the largest cluster
    """
    # Ensure the tensor is on CPU for numpy operations
    objemb = objemb.cpu()
    
    # Get shape
    C, H, W = objemb.shape
    
    # Reshape the tensor for clustering
    # Shape: (channel, height * width)
    tensor_reshaped = objemb.flatten(start_dim = 1).permute(1,0).numpy()
    
    # Perform K-Means Clustering
    # Note: You might want to choose an appropriate number of clusters (n_clusters)
    kmeans = KMeans(n_clusters=2, random_state=0,n_init="auto").fit(tensor_reshaped)

    #visualize using umap and scatter plot
    # embedding = umap.UMAP().fit_transform(tensor_reshaped)
    # plt.scatter(
    #     embedding[:, 0],
    #     embedding[:, 1],
    #     c=kmeans.labels_,)
    # plt.show()

    # Find the largest cluster
    cluster_counts = Counter(kmeans.labels_)
    largest_cluster_label = max(cluster_counts, key=cluster_counts.get)
    
    # Mask for isolating the largest cluster and calculate the mean
    mask = kmeans.labels_ == largest_cluster_label
    largest_cluster_data = tensor_reshaped[mask]
    
    plt.imshow(img)
    resmask = mask.reshape((H,W))
    for i in range(resmask.shape[0]):
        for j in range(resmask.shape[1]):
            if resmask[i,j]:
                plt.gca().add_patch(Rectangle((objbox[0] * WIDTH+ j * PATCH_SIZE,objbox[1] * HEIGHT + i * PATCH_SIZE),PATCH_SIZE,PATCH_SIZE,edgecolor='None',facecolor='red',lw=2))
# plt.gca().add_patch(Rectangle((pixelbox[0],pixelbox[1]),pixelbox[2],pixelbox[3],

    plt.show()

    # Calculate the mean of the largest cluster along the channel dimension
    largest_cluster_mean = np.mean(largest_cluster_data, axis=0)
    
    # Convert the mean back to a PyTorch tensor
    largest_cluster_mean_tensor = torch.tensor(largest_cluster_mean, dtype=objemb.dtype)
    
    return largest_cluster_mean_tensor

class GOT10kDatasetMLP(Dataset):
    def __init__(self, root_dir, train=True, emb_out = "linear", n_train = 953, freq = 1):
        self.root_dir = root_dir
        self.emb_out = emb_out  
        self.train_freq = freq
        if train:
            self.root_dir= self.root_dir + "/train"
            self.folders = getSubFolders(self.root_dir)[0:n_train] #first 500 for now
        else:
            self.root_dir= self.root_dir + "/val"
            self.folders = getSubFolders(self.root_dir)
        self.length= None
        self.indextoPathIndexandBox = []

        for folder in self.folders:
            xfactor ,yfactor = getFolderxyFactor(folder)
            with open(f"{folder}/groundtruth.txt") as f:
                lines = f.readlines()
            gtboxes = []
            for line in lines:
                box = line.split(",")
                box = [float(b) for b in box]
                gtboxes.append(box)

            gtboxes = torch.tensor(gtboxes)
            #convert according to dingsda
            gtboxes[:,0] = gtboxes[:,0] * xfactor / WIDTH
            gtboxes[:,2] = gtboxes[:,2] * xfactor / WIDTH
            gtboxes[:,1] = gtboxes[:,1] * yfactor / HEIGHT
            gtboxes[:,3] = gtboxes[:,3] * yfactor / HEIGHT

            count =0
            filenames = os.listdir(folder)
            filenames.sort()
            for filename in filenames:
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    self.indextoPathIndexandBox.append((folder, count, gtboxes[count], gtboxes[0]))
                    count += 1
        
        print("Dataset initialized")
    
    def __len__(self):
        return len(self.indextoPathIndexandBox)//self.train_freq
    
    def __getitem__(self, index):
        index = index * self.train_freq
        #obj is box of first frame that contains the to be tracked object
        folder, indexInFolder, gtbox, objbox = self.indextoPathIndexandBox[index]
        # load the correct embedding
        objectemb= torch.load(folder + "/tracking/DINOv2/embeddingsDINO/" + str(0) + ".pt")
        pwmin = int(objbox[0] * W_EMB_DINO)
        pwmax = int(torch.ceil((objbox[0]+objbox[2]) * W_EMB_DINO).item())
        phmin = int(objbox[1] * H_EMB_DINO)
        phmax = int(torch.ceil((objbox[1]+objbox[3]) * H_EMB_DINO).item())
        pwmax = W_EMB_DINO -1 if pwmax  >= W_EMB_DINO else pwmax
        phmax = H_EMB_DINO -1 if phmax >= H_EMB_DINO else phmax
        if phmax - phmin == 0:
            if phmax < H_EMB_DINO -1:
                phmax += 1
            else:
                phmin -= 1
        if pwmax - pwmin == 0:
            if pwmax < W_EMB_DINO -1:
                pwmax += 1
            else:
                pwmin -= 1

        objectemb = objectemb[:,phmin:phmax, pwmin:pwmax]
        path = folder + "/" + f"{indexInFolder+1:08}" + ".jpg"

        if objectemb.shape[1] * objectemb.shape[2] == 1:
            objembvar = torch.zeros(C_EMB_DINO)
        else:
            objembvar = torch.var(objectemb, dim = (1,2))

        if self.emb_out == "mean":
            #get biggest cluster of objectemb along first dimension
            # img = torchvision.io.read_image(folder + "/" + f"{1:08}" + ".jpg")
            # img = F.interpolate(img.unsqueeze(0)/255.0, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)[0]
            # img = img.permute(1,2,0)
            # objectemb = cluster_and_find_mean(objectemb, img, objbox)
            objectemb = torch.mean(objectemb, dim = (1,2))
        if self.emb_out == "linear":
            # objectemb = F.pad(objectemb, (0,W_EMB_DINO - objectemb.shape[2],0,H_EMB_DINO - objectemb.shape[1],0,0))
            objectemb = F.interpolate(objectemb.unsqueeze(0), size=(W_EMB_DINO// 4, H_EMB_DINO//4), mode="bilinear", align_corners=False)[0]


        emb = torch.load(folder + "/tracking/DINOv2/embeddingsDINO/" + str(indexInFolder) + ".pt")

        return emb, gtbox, objectemb, objembvar, path



class GOT10kDatasetImg(Dataset):
    def __init__(self, root_dir, train=True,n_train = 953 , n_test = 180, freq =1):
        self.train_freq = freq
        self.root_dir = root_dir
        if train:
            self.root_dir= self.root_dir + "/train"
            self.folders = getSubFolders(self.root_dir)[0:n_train]
        else:
            self.root_dir= self.root_dir + "/val"
            self.folders = getSubFolders(self.root_dir)[0:n_test]
        self.length= None
        self.indextoPathIndexandBox = []

        for folder in self.folders:
            xfactor ,yfactor = getFolderxyFactor(folder)
            with open(f"{folder}/groundtruth.txt") as f:
                lines = f.readlines()
            gtboxes = []
            for line in lines:
                box = line.split(",")
                box = [float(b) for b in box]
                gtboxes.append(box)

            gtboxes = torch.tensor(gtboxes)
            #convert according to dingsda
            gtboxes[:,0] = gtboxes[:,0] * xfactor / WIDTH 
            gtboxes[:,2] = gtboxes[:,2] * xfactor / WIDTH
            gtboxes[:,1] = gtboxes[:,1] * yfactor / HEIGHT
            gtboxes[:,3] = gtboxes[:,3] * yfactor / HEIGHT

            count =0
            filenames = os.listdir(folder)
            filenames.sort()
            for filename in filenames:
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    self.indextoPathIndexandBox.append((folder, count, gtboxes[count], gtboxes[0]))
                    count += 1
        
            # self.transform=  torchvision.transforms.Compose([
            #     torchvision.transforms.Resize((HEIGHT, WIDTH), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            # ])
        print("Dataset initialized")
    
    def __len__(self):
        return len(self.indextoPathIndexandBox)//self.train_freq
    
    def __getitem__(self, index):
        index = index * self.train_freq
        folder, indexInFolder, gtbox, objbox = self.indextoPathIndexandBox[index]
        path = folder + "/" + f"{indexInFolder+1:08}" + ".jpg"
        img = torchvision.io.read_image(path)
        img = F.interpolate(img.unsqueeze(0)/255.0, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False).squeeze(0)
        objimg = torchvision.io.read_image(folder + "/" + f"{0+1:08}" + ".jpg")
        objimg = F.interpolate(objimg.unsqueeze(0)/255.0, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False).squeeze(0)
        return img, gtbox, objimg, objbox, path







# img = torchvision.io.read_image(folder + "/" + f"{0+1:08}" + ".jpg")
# img = F.interpolate(img.unsqueeze(0)/255.0, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)[0]
# img = img.permute(1,2,0)
# #draw a box on the image
# embunitbox = [pwmin, phmin, pwmax-pwmin, phmax-phmin]
# embunitbox[0] = embunitbox[0] * WIDTH / W_EMB
# embunitbox[2] = embunitbox[2] * WIDTH / W_EMB
# embunitbox[1] = embunitbox[1] * HEIGHT / H_EMB
# embunitbox[3] = embunitbox[3] * HEIGHT / H_EMB
# pixelbox = embunitbox
# print(pixelbox)
# plt.gca().add_patch(Rectangle((pixelbox[0],pixelbox[1]),pixelbox[2],pixelbox[3],
#             edgecolor='green',
#             facecolor='none',
#             lw=2))
# plt.imshow(img)
# plt.show()