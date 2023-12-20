import torch.nn as nn
import torch.nn.functional as F
import torch
from config import *
import torchvision
from transformers import ResNetModel
from math import floor
from cleanutil import *

class BoxMlpModelConv(nn.Module):
    def __init__(self, c,h,w, num_outputs=4):
        super().__init__()
        self.dimsin = [(h, w)]
        ks = 5 # kernel size
        stride =1 # stride
        pd = 0 # padding
        self.conv1 = torch.nn.Conv2d(in_channels=2*c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        self.conv2 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        self.convbox1 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        self.convbox2 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd)        
        for i in range(2):
            #dimensions as they resulting w,h from the conv layers
            self.dimsin.append((floor((self.dimsin[-1][0] - ks + 2*pd)/stride + 1), 
            floor((self.dimsin[-1][1] - ks + 2*pd)/stride + 1)))

        # h_lin = self.dimsin[-1][0]
        # w_lin = self.dimsin[-1][1]
        self.dense1 = nn.Linear(c, c)
        # self.dense2 = nn.Linear(c, c)
        self.dense3 = nn.Linear(c, num_outputs)

    #x is image
    def forward(self, x, objemb):
        objemb= F.gelu(self.convbox1(objemb))
        objemb= self.convbox2(objemb)
        objemb = torch.mean(objemb, dim = (2,3))
        objemb = objemb.unsqueeze(-1).unsqueeze(-1).expand(-1,-1, x.shape[2], x.shape[3])
        x= torch.cat((x, objemb), dim = 1)
        x= F.gelu(self.conv1(x))
        x= self.conv2(x)
        x = torch.mean(x, dim = (2,3))
        x = F.gelu(self.dense1(x))
        # x = F.gelu(self.dense2(x))
        x = self.dense3(x)
        return torch.sigmoid(x)
        # return torch.tanh(x) + lastbox 


class BoxMlpModelConvMeanResnet(nn.Module):
    def __init__(self, c,h,w, num_outputs=4):
        super().__init__()
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.model.eval()
        self.dimsin = [(h, w)]
        ks = 3 # kernel size
        stride =2 # stride
        pd = 1 # padding
        self.convdown1 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        self.convdown2 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        for i in range(2):
            #dimensions as they resulting w,h from the conv layers
            self.dimsin.append((floor((self.dimsin[-1][0] - ks + 2*pd)/stride + 1), 
            floor((self.dimsin[-1][1] - ks + 2*pd)/stride + 1)))

        h_lin = self.dimsin[-1][0]
        w_lin = self.dimsin[-1][1]
        self.dense1 = nn.Linear(c*w_lin*h_lin + c, c)
        self.dense2 = nn.Linear(c, c)
        self.dense3 = nn.Linear(c, num_outputs)
        self.w =w
        self.h = h
        self.c = c

    #x is image, x_obj object image
    def forward(self, x, x_obj, objbox):
        with torch.no_grad():
            x = self.model(x)["last_hidden_state"]
            x_obj = self.model(x_obj)["last_hidden_state"]


        x_obj = getPatchesInBox(objbox, x_obj, self.w, self.h)
        objembmean = [objem.mean(dim=(1,2)) for objem in x_obj]
        objembvar = [objem.var(dim=(1,2)) for objem in x_obj]
        objembmean = torch.stack(objembmean)
        objembvar = torch.stack(objembvar)

        x = x - objembmean.unsqueeze(2).unsqueeze(3)

        x= self.convdown1(x)
        x= self.convdown2(x)
        x = x.flatten(start_dim=1)
        x = torch.cat((x,objembvar), dim = 1) #cat variance
        x = F.gelu(self.dense1(x))
        x = F.gelu(self.dense2(x))
        x = self.dense3(x)
        return torch.sigmoid(x) 



class BoxMlpModelConvMean(nn.Module):
    def __init__(self, c,h,w, num_outputs=4):
        super().__init__()
        self.dimsin = [(h, w)]
        ks = 3 # kernel size
        stride =2 # stride
        pd = 1 # padding
        self.convdown1 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        self.convdown2 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        for i in range(2):
            #dimensions as they resulting w,h from the conv layers
            self.dimsin.append((floor((self.dimsin[-1][0] - ks + 2*pd)/stride + 1), 
            floor((self.dimsin[-1][1] - ks + 2*pd)/stride + 1)))

        h_lin = self.dimsin[-1][0]
        w_lin = self.dimsin[-1][1]
        self.dense1 = nn.Linear(c*w_lin*h_lin, c)
        self.dense2 = nn.Linear(c, c)
        self.dense3 = nn.Linear(c, num_outputs)

    #x is embedding of current img, meanemb is mean of object embeddings, objembvar is variance of object embeddings
    def forward(self, x, meanemb):
        x = x - meanemb.unsqueeze(2).unsqueeze(3)
        x= self.convdown1(x)
        x= self.convdown2(x)
        x = x.flatten(start_dim=1)
        # x = torch.cat((x,objembvar), dim = 1) #cat variance
        x = F.gelu(self.dense1(x))
        x = F.gelu(self.dense2(x))
        x = self.dense3(x)
        return torch.sigmoid(x) #todo shit has trouble getting to 1 and 0


class BoxMlpModelConvLinear(nn.Module):
    def __init__(self, c,h,w, num_outputs=4):
        super().__init__()
        self.dimsin = [(h, w)]
        self.dimsinbox = [(h//4, w//4)]
        ks = 3 # kernel size
        stride =2 # stride
        pd = 1 # padding
        self.convdown1 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        self.convdown2 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        self.convdownbox1 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd) 
        self.convdownbox2 = torch.nn.Conv2d(in_channels=c, out_channels=c, kernel_size=ks, stride=stride, padding=pd)        
        for i in range(2):
            #dimensions as they resulting w,h from the conv layers
            self.dimsin.append((floor((self.dimsin[-1][0] - ks + 2*pd)/stride + 1), 
            floor((self.dimsin[-1][1] - ks + 2*pd)/stride + 1)))
            self.dimsinbox.append((floor((self.dimsinbox[-1][0] - ks + 2*pd)/stride + 1), 
            floor((self.dimsinbox[-1][1] - ks + 2*pd)/stride + 1)))

        h_lin = self.dimsin[-1][0]
        w_lin = self.dimsin[-1][1]
        self.dense1 = nn.Linear(c*w_lin*h_lin, c)
        self.dense2 = nn.Linear(c, c)
        self.dense3 = nn.Linear(c, num_outputs)
        self.densebox1= nn.Linear(c*self.dimsinbox[-1][0]*self.dimsinbox[-1][1], c)
        self.densebox2= nn.Linear(c, c)

    def forward(self, x, objemb):
        objemb= self.convdownbox1(objemb)
        objemb= self.convdownbox2(objemb)
        objemb = objemb.flatten(start_dim = 1)
        objemb = F.gelu(self.densebox1(objemb))
        meanemb = F.gelu(self.densebox2(objemb))
        x = x - meanemb.unsqueeze(2).unsqueeze(3)
        x= self.convdown1(x)
        x= self.convdown2(x)
        x = x.flatten(start_dim=1)
        x = F.gelu(self.dense1(x))
        x = F.gelu(self.dense2(x))
        x = self.dense3(x)
        return torch.sigmoid(x) #todo shit has trouble getting to 1 and 0


class BoxMlpModelMean(nn.Module):
    def __init__(self, c,h,w, num_outputs=4):
        super().__init__()
        hidden_dim = 256
        self.dense1 = nn.Linear(c*w*h + c, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x, objemb, objembvar):
        x = x - objemb.unsqueeze(2).unsqueeze(3)
        x = x.flatten(start_dim=1)
        x = torch.cat((x,objembvar), dim = 1)
        x = F.gelu(self.dense1(x))
        x = F.gelu(self.dense2(x))
        x = self.dense3(x)
        return torch.sigmoid(x) 



class BoxMlpModelLinear(nn.Module):
    def __init__(self, c,h,w, num_outputs=4):
        super().__init__()
        hidden_dim = 256
        self.dense1 = nn.Linear(c*w*h + c, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_outputs)
        self.densebox1= nn.Linear(c*w*h, c)
        self.densebox2= nn.Linear(c, c)

    def forward(self, x, objemb, objembvar):
        objemb = objemb.flatten(start_dim = 1)
        objemb = F.gelu(self.densebox1(objemb))
        objemb = F.gelu(self.densebox2(objemb))
        x = x - objemb.unsqueeze(2).unsqueeze(3)
        x = x.flatten(start_dim=1)
        x = torch.cat((x,objembvar), dim = 1)
        x = F.gelu(self.dense1(x))
        x = F.gelu(self.dense2(x))
        x = self.dense3(x)
        return torch.sigmoid(x)

import math
class WeightSumLayer(nn.Module):
    #n_vec is number of vectors, l_vec is length of vectors
    def __init__(self,n_vec, l_vec):
        super().__init__()
        self.w = nn.Parameter(torch.empty(l_vec,  n_vec))
        nn.init.uniform_(self.w, a=-1/math.sqrt(l_vec), b=1/math.sqrt(l_vec))

    #dims of x are batch, channel, count
    def forward(self, x):
        x = x * self.w 
        x = torch.sum(x, dim = 2)
        return x



class SparseLinearModel(nn.Module):
    def __init__(self, c,h,w, num_outputs=4):
        super().__init__()
        self.weightsum1 = WeightSumLayer(h*w, c)
        self.weightsum2 = WeightSumLayer(h*w, c)
        self.weightsum3 = WeightSumLayer(h*w, c)
        self.weightsum4 = WeightSumLayer(h*w, c)
        self.weightsumbox1 = WeightSumLayer(h*w, c)
        self.weightsumbox2 = WeightSumLayer(h*w, c)
        hidden_dim = 256
        self.dense1 = nn.Linear(6*c, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, num_outputs)
    def forward(self, x, objemb):
        objemb = objemb.flatten(start_dim = 2)
        b1 = self.weightsumbox1(objemb)
        b2 = self.weightsumbox2(objemb)
        # x = x - objemb.unsqueeze(2).unsqueeze(3)
        x = x.flatten(start_dim=2)
        p1 = self.weightsumbox1(x)
        p2 = self.weightsumbox1(x)
        p3 = self.weightsumbox1(x)
        p4 = self.weightsumbox1(x)

        #cat b1,b2,p1,p2,p3,p4
        x = torch.cat((b1.unsqueeze(2),b2.unsqueeze(2),p1.unsqueeze(2),p2.unsqueeze(2),p3.unsqueeze(2),p4.unsqueeze(2)), dim = 2)
        x = x.flatten(start_dim=1)
        x = F.gelu(self.dense1(x))
        x = F.gelu(self.dense2(x))
        x = self.dense3(x)
        return torch.sigmoid(x)



class BoxEncoderModel(nn.Module):
    def __init__(self,c = 384, num_outputs=4) -> None:
        super().__init__()
        self.emblayer = nn.Linear(1, c)
        self.classemblayer = nn.Linear(1,c)
        self.enclayer= nn.TransformerEncoderLayer(c, 
                                        nhead=6, 
                                        dim_feedforward=4*c, 
                                        dropout=0.1,
                                        batch_first=True)
        self.projection = nn.Linear(c,num_outputs)
        self.c = c

    #todo padding mask
    #todo less seq length 
    def forward(self,x,objemb):
        # x = batch,c,h,w
        # objemb = batch,c,h_box,w_box
        a = self.emblayer(torch.ones(1).to("cuda"))
        b = self.emblayer(torch.zeros(1).to("cuda"))
        a= a.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        b= b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # print("a shape ", a.shape)
        x = x + a
        # print("objemb sahpe ", objemb.shape)
        objemb = objemb + b

        x = x.flatten(start_dim=2).permute(0,2,1)
        objemb = objemb.flatten(start_dim=2).permute(0,2,1) #batch seq feature

        classtoken = torch.ones((x.shape[0], 1)).to("cuda") #batch feature
        classtoken = self.classemblayer(classtoken).unsqueeze(1) #batch seq feature

        x = torch.cat((classtoken, x,objemb), dim = 1)
        # print("x shape: ", x.shape)

        x = self.enclayer(x)
        x = self.projection(x[:,0])
        return torch.sigmoid(x) #todo shit


