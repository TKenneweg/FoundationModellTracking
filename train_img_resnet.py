
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




if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    config = SimpleNamespace(
        epochs=50,
        train_batch_size=1,
        test_batch_size=1,
        lr=0.01,
        model_name = "BoxMlpModelConvLinear",
        batch_size =4,
        backbone = "resnet",
        emb_out = "linear",
        wandb_project = "tracking",
        optimizer=  "SGD",
        foldername = "/home/tristan/ExtraDisk/GOT-10k",
        seed = 0,
        log_freq = 100,
        n_train = 9334, #max is 953, 9335 for pure img
        n_test =180, #max is 180
        train_freq = 10,
        test_freq = 1,
        n_params = 0,
        seq = False,
    )
    if config.model_name == "BoxMlpModelMean":
        model = BoxMlpModelMean(c=C_EMB_DINO) 
    if config.model_name == "BoxMlpModelConvMean":
        model = BoxMlpModelConvMean(h = H_EMB_DINO, w= W_EMB_DINO, c=C_EMB_DINO) 
    if config.model_name == "BoxMlpModelConvLinear":
        model = BoxMlpModelConvLinear(c=C_EMB_RESNET, w=W_EMB_RESNET, h=H_EMB_RESNET) 
    model.to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    compiled_model = torch.compile(model)
    compiled_model = model
    wandb.init(project="tracking", config=config)
    torch.manual_seed(config.seed)
    dataset  = GOT10kDatasetImg(config.foldername, train=True, n_train=config.n_train)
    testset = GOT10kDatasetImg(config.foldername, train=False, n_test=config.n_test )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(testset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    mseloss = torch.nn.MSELoss()
    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=config.epochs * len(dataloader))

    # dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    # for p in dino.parameters():
    #     p.requires_grad = False
    # # dino = torch.compile(dino)
    # dino.eval()
    # dino.to("cuda")
    backbone = ResNetModel.from_pretrained("microsoft/resnet-50")
    backbone = backbone.to("cuda")

    running_loss = 0.0
    best_acc = 0.0
    bestloss = 1e6
    n_iou_05_train = 0

    for epoch in range(config.epochs):
        compiled_model.train()
        for i, data in enumerate(dataloader):
            img, gtbox, objimg, objbox = (x.to('cuda:0') for x in data[:-1])
            path = data[-1]
            optimizer.zero_grad()

            with torch.no_grad():
                if img.shape[0] != config.batch_size or objimg.shape[0] != config.batch_size:
                    print("wrong batch size")
                    continue
                # norm_trans = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                # img = norm_trans(img)
                # objimg = norm_trans(objimg)
                emb = backbone(img)["last_hidden_state"]
                objemb = backbone(objimg)["last_hidden_state"]
                objemb = getPatchesInBox(objbox, objemb, W_EMB_RESNET, H_EMB_RESNET, path) #returns list of batch elements, whcih are tensors, batching is not possbile since sizes varies

                tmp = []
                abort = False
                for obj in objemb:
                    if obj.shape[0] == 0 or obj.shape[1] == 0 or obj.shape[2] == 0 or obj.shape is None:
                        abort = True
                        break
                    tmp.append(F.interpolate(obj.unsqueeze(0), size=(W_EMB_RESNET// 4, H_EMB_RESNET//4), mode="bilinear", align_corners=False)[0])
                if abort:
                    print("######### empty objemb ##########")
                    continue
                objemb = tmp
                objemb = torch.stack(objemb) #stack to get batch dimension

            outputs = compiled_model(emb, objemb)
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
        running_iou = 0
        compiled_model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                img, gtbox, objimg, objbox = (x.to('cuda:0') for x in data[:-1])
                path = data[-1]
                if img.shape[0] != config.batch_size or objimg.shape[0] != config.batch_size:
                    print("wrong batch size")
                    print(img.shape)
                    print(objimg.shape)
                    continue

                emb = backbone(img)["last_hidden_state"]
                objemb = backbone(objimg)["last_hidden_state"]
                objemb = getPatchesInBox(objbox, objemb, W_EMB_RESNET, H_EMB_RESNET, path) #returns list of batch elements, whcih are tensors, batching is not possbile since sizes varies

                tmp = []
                abort = False
                for obj in objemb:
                    if obj.shape[0] == 0 or obj.shape[1] == 0 or obj.shape[2] == 0 or obj.shape is None:
                        abort = True
                        break
                    tmp.append(F.interpolate(obj.unsqueeze(0), size=(W_EMB_RESNET// 4, H_EMB_RESNET//4), mode="bilinear", align_corners=False)[0])
                if abort:
                    print("######### empty objemb ##########")
                    continue
                objemb = tmp
                objemb = torch.stack(objemb) #stack to get batch dimension

                outputs = compiled_model(emb, objemb)
                loss = mseloss(outputs, gtbox)
                running_lossTest += loss.item()
                ious = batch_ious(outputs, gtbox)
                for iou in ious:
                    running_iou += iou
                    if iou > 0.5:
                        n_iou_05 += 1

            avg_loss = running_lossTest / len(test_dataloader) # loss per batch
            f_50 = n_iou_05 / (len(test_dataloader) * config.batch_size)
            avg_iou = running_iou / (len(test_dataloader) * config.batch_size)
            img = wandb.Image(plotpredImage(path[0], gtbox[0].detach().cpu(), outputs[0].detach().cpu()))
            print("avg_testloss: ", avg_loss, " Succes Rate > 0.5: ", f_50)
            wandb.log({"avg_testloss": avg_loss, "Succes Rate > 0.5_test": f_50,"testpred": img, "avg_iou_test": avg_iou})

            if avg_loss < bestloss:
                bestloss = avg_loss
                torch.save(model, f"models/{wandb.run.name}.pt")
                print(f"Best model saved as {wandb.run.name}.pt")




