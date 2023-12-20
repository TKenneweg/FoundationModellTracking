import torch

#coordinates must be channel,h,w 
def getEuclideanDistance(tensor1, tensor2):
    return (tensor1 - tensor2).pow(2).sum(dim=0).sqrt()
     
def getManhattenDistance(tensor1,tensor2):
    return (tensor1- tensor2).abs().sum(dim=0)

def getCosineSimilarityDistance(tensor1,tensor2):
    #perform elementwise dot product
    sim = -torch.nn.functional.cosine_similarity(tensor1,tensor2, dim= 0)
    # print(sim)
    return sim
    
def getMahalabonisDistance(tensor1,tensor2):
    pass

