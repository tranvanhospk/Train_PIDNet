import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.utils import AverageMeter, Metric_mIoU

from models import get_train_model
from dataset import UTE_Dataset
from dataset import train_albumentations
from utils import OhemCrossEntropy, BondaryLoss, CrossEntropy


import warnings
warnings.filterwarnings("ignore")



def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print(msg)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model



def train(model, train_dataloader, sem_loss, bd_loss, optimizer, miou_metric, device):
    model.train()

    loss_meter = AverageMeter()
    miou_metric.reset()

    for images, labels, edges in tqdm(train_dataloader):
        images, labels, edges = images.to(device), labels.to(device), edges.to(device)

        outputs = model(images)

        # compute loss
        h, w = labels.size(1), labels.size(2)
        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if ph != h or pw != w:
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(outputs[i], size=(h, w), mode='bilinear', align_corners=True)


        loss_s = sem_loss(outputs[:-1], labels)
        loss_b = bd_loss(outputs[-1], edges)

        filler = torch.ones_like(labels) * 255
        bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>0.8, labels, filler)

        loss_sb = sem_loss([outputs[-2]], bd_label)
        loss = loss_s + loss_b + loss_sb

        loss_meter.update(loss.item())

        # compute miou
        miou_metric.update(outputs[-2], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, optimizer, loss_meter.get(), miou_metric.get()


def val(model, val_dataloader, sem_loss, bd_loss, miou_metric, device):
    model.eval()
    loss_meter = AverageMeter()
    miou_metric.reset()

    for images, labels, edges in tqdm(val_dataloader):
        images, labels, edges = images.to(device), labels.to(device), edges.to(device)

        outputs = model(images)

        # compute loss
        h, w = labels.size(1), labels.size(2)
        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if ph != h or pw != w:
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(outputs[i], size=(h, w), mode='bilinear', align_corners=True)

        loss_s = sem_loss(outputs[:-1], labels)
        loss_b = bd_loss(outputs[-1], edges)

        filler = torch.ones_like(labels) * 255
        bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>0.8, labels, filler)
        loss_sb = sem_loss([outputs[-2]], bd_label)
        loss = loss_s + loss_b + loss_sb


        loss_meter.update(loss.item())

        # compute miou
        miou_metric.update(outputs[-2], labels)

    return loss_meter.get(), miou_metric.get()



def train_loop(model, train_dataloader, val_dataloader, sem_loss, bd_loss, optimizer, metric, epochs, device):

    lf = lambda x: (1 - x / epochs) * (1.0 - 0.01) + 0.01  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    for epoch in range(1, epochs+1):
        model, optimizer, train_loss, train_miou =  train(model, train_dataloader, sem_loss, bd_loss, optimizer, metric, device)

        # with torch.no_grad():
        #     val_loss, val_miou = val(model, val_dataloader, sem_loss, bd_loss, metric, device)

        scheduler.step()

        torch.save(model.state_dict(), 'weights/model_PID_s.pth')

        print( f" Epoch: { epoch }, Train loss: {train_loss:.3f}, Train miou: { train_miou:.3f}" )
        
        # print( f" Epoch: { epoch }, Train loss: {train_loss:.3f}, Valid loss: { val_loss:.3f}" )
        # print( f" Epoch: { epoch }, Train miou : {train_miou:.3f}, Valid miou : { val_miou:.3f}" )



def main(config):
    model = get_train_model(name='s', num_classes=config.num_classes).to(config.device)
    model = load_pretrained(model, pretrained=config.pretrained)


    train_dataset = UTE_Dataset(config.data_root, config.train_list_file, transform=train_albumentations)
    val_dataset = UTE_Dataset(config.data_root, config.val_list_file)

    train_dataloaer = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    sem_loss = CrossEntropy()
    bd_loss = BondaryLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=0.05)

    miou_metric = Metric_mIoU(config.num_classes)

    train_loop(model=model, 
               train_dataloader=train_dataloaer, 
               val_dataloader=val_dataloader, 
               sem_loss=sem_loss, 
               bd_loss=bd_loss, 
               optimizer=optimizer, 
               metric=miou_metric, 
               epochs=config.EPOCHS, 
               device=config.device)


class Trainer_Config:

    data_root = './data'

    train_list_file = [
        './data_path/DatasetUTE/train.txt',
        './data_path/DatasetUTE/val.txt'

        # './data_path/Cityscapes/train.txt',
        # './data_path/Cityscapes/val.txt',

        # './data_path/CamVid/train.txt',
        # './data_path/CamVid/val.txt',
        # './data_path/CamVid/test.txt',

        # './data_path/Bdd100k/train.txt',
        # './data_path/Bdd100k/val.txt',
    ]
    val_list_file = [
        './data_path/DatasetUTE/val.txt'
    ]

    pretrained = './weights/PIDNet_S_Cityscapes_val.pt'

    num_classes = 4

    batch_size = 2

    LR = 0.0001

    EPOCHS = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    config = Trainer_Config()
    main(config)




