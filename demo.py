import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image

from models.pidnet import get_pred_model


class SegmentPredictor:
    def __init__(self, weight, device):
        self.weight = weight
        self.device = device

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225] 

        self.color_map = [
            (0, 0, 0),      # backgroud
            (31, 120, 180), # road
            (227, 26, 28),  # person 
            (106, 61, 154), # car
            ]

        get_model = get_pred_model('s', num_classes=4)
        self.model = self.load_model(get_model, weight=weight).eval()

    def prepare_input(self, image):
        image = cv2.resize(image, (640, 480))
        # normolize
        image = image.astype(np.float32) / 255.0
        image -= self.mean
        image /= self.std
        image = torch.Tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
        return image
    
    def load_model(self, model, weight):
        weight_dict = torch.load(weight,  map_location=torch.device(self.device))
        model_dict = model.state_dict()
        weight_dict = {k: v for k, v in weight_dict.items() if k in model_dict}
        model_dict.update(weight_dict)
        model.load_state_dict(model_dict)
        return model
    
    def visualize(self, mask, image):

        sv_img = np.zeros_like(image).astype(np.uint8)
        for i, color in enumerate(self.color_map):
            for j in range(3):
                sv_img[:,:,j][mask==i] = self.color_map[i][j]

        image_seg = cv2.addWeighted(sv_img, 0.5, image, 0.5, 0)

        return image_seg

        

    def __call__(self, image):
        image = self.prepare_input(image)
        pred = self.model(image)
        
        # postprocess
        pred = F.interpolate(pred, size=(480, 640), mode='bilinear', align_corners=True)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.squeeze(0).numpy()

        return pred
    


if __name__ == '__main__':

    device = 'cpu'
    predictor = SegmentPredictor('./weights/model_PID_s_3_class.pt', device)

    image_path = ('./data/DatasetUTE/val/images/data_1203.jpg')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))

    output = predictor(image)

    visualize_image = predictor.visualize(output, image)

    cv2.imshow('im', visualize_image)
    cv2.waitKey(0) 




