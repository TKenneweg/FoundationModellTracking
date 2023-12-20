import torch
from GOT10kDataset import *
from torch.utils.data import DataLoader
import wandb
from cleanutil import *
from model import *
from trainBBtracker import config
from types import SimpleNamespace




if __name__ == "__main__":
    config = SimpleNamespace(
        model_name = "woven-armadillo-335.pt",
        batch_size =4,
        wandb_project = "tracking",
        foldername = "/home/tristan/ExtraDisk/GOT-10k",
        seed = 0,
        n_test =5, #max is 180
        test_freq = 1,
    )
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision('high')
    model = torch.load(f"models/{config.model_name}")
    model.eval()
    compiled_model = torch.compile(model)
    compiled_model.eval()
    testset = GOT10kDatasetImg(config.foldername, train=False, n_test=config.n_test )
    test_dataloader = DataLoader(testset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    mseloss = torch.nn.MSELoss()
    wandb.init(project="tracking", config=config)

    backbone = ResNetModel.from_pretrained("microsoft/resnet-50")
    backbone = backbone.to("cuda")
    backbone.eval()


    f_50 = 0
    running_loss = 0.0
    running_iou = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            img, gtbox, objimg, objbox = (x.to('cuda:0') for x in data[:-1])
            path = data[-1]
            if img.shape[0] != config.batch_size or objimg.shape[0] != config.batch_size:
                print("wrong batch size")
                continue
            emb = backbone(img)["last_hidden_state"]
            objemb = backbone(objimg)["last_hidden_state"]
            objemb = getPatchesInBox(objbox, objemb, W_EMB_RESNET, H_EMB_RESNET, path) #returns list of batch elements, whcih are tensors, batching is not possbile since sizes varies

            tmp = []
            abort = False
            for obj in objemb:
                if obj.shape[0] == 0:
                    abort = True
                    break
                tmp.append(F.interpolate(obj.unsqueeze(0), size=(W_EMB_RESNET// 4, H_EMB_RESNET//4), mode="bilinear", align_corners=False)[0])
            if abort:
                print("######### empty objemb ##########")
                continue
            objemb = tmp  
            objemb = torch.stack(objemb) #stack to get batch dimension

            print(emb.shape)
            print(objemb.shape)
            print(compiled_model)
            outputs = compiled_model(emb, objemb)
            loss = mseloss(outputs, gtbox)
            running_loss += loss.item()
            batch_iou = batch_ious(outputs, gtbox)
            for iou in batch_iou:
                running_iou += iou
                if iou > 0.5:
                    f_50 += 1

        avg_loss = running_loss / len(test_dataloader) # loss per batch
        avg_iou = running_iou / (len(test_dataloader) * config.batch_size)
        f_50 = f_50 / (len(test_dataloader) * config.batch_size)
        logdict= {"avg_testloss": avg_loss, "avg_iou": avg_iou, "f_50": f_50}
        print(logdict)
        wandb.log(logdict)



    # avg_loss = running_loss2 / len(test_dataloader) # loss per batch
    # print("avg_testloss", avg_loss)
    # log_dict = {"avg_testloss": avg_loss}
    # wandb.log(log_dict)


#q: why different test loss for same model?