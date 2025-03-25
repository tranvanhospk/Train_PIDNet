import numpy as np
import torch


class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self):
        return self.avg
    

def converter(data):
    if isinstance(data,torch.Tensor):
        data = data.cpu().data.numpy().flatten()
    return data.flatten()

def fast_hist(label_pred, label_true,num_classes):
    hist = np.bincount(num_classes * label_true.astype(int) + label_pred, minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist

class Metric_mIoU():
    def __init__(self,class_num):
        self.class_num = class_num
        self.hist = np.zeros((self.class_num,self.class_num))

    def update(self, predict, target):
        predict = torch.argmax(predict, dim=1)
        predict, target = converter(predict), converter(target)
        self.hist += fast_hist(predict, target, self.class_num)

    def reset(self):
        self.hist = np.zeros((self.class_num, self.class_num))

    def get_miou(self):
        miou = np.diag(self.hist) / (np.sum(self.hist, axis=1) + np.sum(self.hist, axis=0) - np.diag(self.hist))
        miou = np.nanmean(miou)
        return miou

    def get_acc(self):
        acc = np.diag(self.hist) / self.hist.sum(axis=1)
        acc = np.nanmean(acc)
        return acc
    
    def get(self):
        return self.get_miou()