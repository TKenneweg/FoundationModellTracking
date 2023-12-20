
import torch
from model import *
from GOT10kDataset import *
import numpy as np
import wandb
from cosine_scheduler import CosineWarmupScheduler
from torch.utils.data import DataLoader
from cleanutil import *
from config import *
from types import SimpleNamespace

 
def getnumparam(model):
    count = 0
    for p in model.parameters():
        count += p.numel()
    return count


config = SimpleNamespace(
    epochs=50,
    train_batch_size=1,
    test_batch_size=1,
    lr=0.01,
    model_name = "BoxMlpModelConvLinear",
    batch_size =8,
    backbone = "dinov2",
    emb_out = "linear",
    wandb_project = "tracking",
    optimizer=  "SGD",
    foldername = "/home/tristan/ExtraDisk/GOT-10k",
    seed = 0,
    log_freq = 100,
    n_train = 953, #max is 953
    n_test =180, #max is 180
    train_freq = 1,
    test_freq = 1,
    n_params = 0,
    seq = False,
)
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    if config.model_name == "BoxMlpModelMean":
        model = BoxMlpModelMean(c=C_EMB_DINO, w=W_EMB_DINO, h=H_EMB_DINO) 
    if config.model_name == "BoxMlpModelConvMean":
        model = BoxMlpModelConvMean(h = H_EMB_DINO, w= W_EMB_DINO, c=C_EMB_DINO) 
    if config.model_name == "BoxMlpModelConvLinear":
        model = BoxMlpModelConvLinear(c=C_EMB_DINO, w=W_EMB_DINO, h=H_EMB_DINO) 
    if config.model_name == "BoxMlpModelConvMeanResnet":
        model = BoxMlpModelConvMeanResnet(c=2048, h=23, w=40) #fitting for resnet 50 and standard input sizes
    if config.model_name == "BoxEncoderModel":
        model = BoxEncoderModel(c=C_EMB_DINO)
    if config.model_name == "SparseLinearModel":
        model = SparseLinearModel(c=C_EMB_DINO, w=W_EMB_DINO, h=H_EMB_DINO)
    if config.model_name == "BoxMlpModelConv":
        model = BoxMlpModelConv(c=C_EMB_DINO, w=W_EMB_DINO, h=H_EMB_DINO)
    config.nparams = getnumparam(model)
    print("The model has ", config.nparams, " parameters")
    model.to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    # compiled_model = torch.compile(model)
    compiled_model = model
    wandb.init(project="tracking", config=config)
    torch.manual_seed(config.seed)
    if config.backbone == "resnet50":
        dataset  = GOT10kDatasetImg(config.foldername, train=True, n_train=config.n_train)
        testset = GOT10kDatasetImg(config.foldername, train=False, )
    if config.backbone == "dinov2":    
        dataset  = GOT10kDatasetMLP(config.foldername, train=True, emb_out= config.emb_out, n_train=config.n_train, freq=config.train_freq)
        testset = GOT10kDatasetMLP(config.foldername, train=False, emb_out= config.emb_out,n_train=config.n_train, freq=config.test_freq)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(testset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    mseloss = torch.nn.MSELoss()
    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=config.epochs * len(dataloader))


    running_loss = 0.0
    best_acc = 0.0
    bestloss = 1e6
    n_iou_05_train = 0
    for epoch in range(config.epochs):
        compiled_model.train()
        for i, data in enumerate(dataloader):
            #in image case is img, gtbox, objimg, objbox, path 
            emb, gtbox, objectemb, objembvar = (x.to('cuda:0') for x in data[:-1])
            path = data[-1]
            optimizer.zero_grad()

            outputs = compiled_model(emb, objectemb)
            # outputs = compiled_model(emb, objectemb, objembvar)
            loss = mseloss(outputs, gtbox)
            loss.backward()

            optimizer.step()
            scheduler.step()


            running_loss += loss.item()
            ious = batch_ious(outputs, gtbox)
            for iou in ious:
                if iou > 0.5:
                    n_iou_05_train += 1
            wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "iou": iou})

            if i % config.log_freq == 0 and i > 0:
                avg_loss = running_loss / config.log_freq # loss per batch
                f_50 = n_iou_05_train / config.log_freq
                n_iou_05_train = 0
                img = wandb.Image(plotpredImage(path[0], gtbox[0].detach().cpu(), outputs[0].detach().cpu()))
                wandb.log({"trainpred": img, "Succes Rate > 0.5_train": f_50})
                running_loss = 0.0
                print(f"batch {i+1} avg_loss: {avg_loss}")


        #exexute once each epoch
        running_lossTest = 0.0
        running_loss = 0.0
        n_iou_05 = 0
        compiled_model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                emb, gtbox, objectemb, objembvar = (x.to('cuda:0') for x in data[:-1])
                path = data[-1]

                outputs = compiled_model(emb, objectemb)
                # outputs = compiled_model(emb, objectemb, objembvar)
                loss = mseloss(outputs, gtbox)

                running_lossTest += loss.item()
                ious = batch_ious(outputs, gtbox)
                for iou in ious:
                    if iou > 0.5:
                        n_iou_05 += 1

            avg_loss = running_lossTest / len(test_dataloader) # loss per batch
            f_50 = n_iou_05 / (len(test_dataloader) * config.batch_size)
            img = wandb.Image(plotpredImage(path[0], gtbox[0].detach().cpu(), outputs[0].detach().cpu()))
            print("avg_testloss: ", avg_loss, " Succes Rate > 0.5: ", f_50)
            wandb.log({"avg_testloss": avg_loss, "Succes Rate > 0.5_test": f_50,"testpred": img})

            if avg_loss < bestloss:
                bestloss = avg_loss
                torch.save(model, f"models/{wandb.run.name}.pt")
                print(f"Best model saved as {wandb.run.name}.pt")




