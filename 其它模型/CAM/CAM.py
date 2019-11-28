import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from ImageNetDict import class_dict


# 加载在ImageNet上面预训练的ResNet-18模型
def load_net():
    net = resnet18()
    model_path = "resnet18.pth"
    net.load_state_dict(torch.load(model_path))
    print("load pretrained resnet model successfully !")
    return net


def get_cam(img_path):
    img = Image.open(img_path)
    ori_w = img.size[0]
    ori_h = img.size[1]
    img2tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = img2tensor(img)
    img_tensor.unsqueeze_(dim=0)  # [3,256,256] -> [1,3,256,256]
    print("load img successfully !")

    net = load_net()
    net.eval()

    heat_map_net = nn.Sequential(*list(net.children())[:-2])  # [n,3,256,256] -> [n,512,8,8]
    origin_fm = heat_map_net(img_tensor)[0]
    last_fc_param = list(net.parameters())[-2]  # [1000,512]
    predict = net(img_tensor)
    logits = torch.argmax(predict, dim=1)  # 图像分类
    print(class_dict[logits.item()])
    weights = torch.softmax(last_fc_param[logits], dim=0)  # 获取置信度分最高的神经元的权重

    res = torch.zeros((7, 7))
    for i in range(512):
        res += weights[0][i] * origin_fm[i]

    res = res / torch.max(res)
    res.clamp_(0, 1)

    tensor2img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
    ])
    heat = tensor2img(res)
    res = transforms.ToTensor()(heat)
    img_tensor.squeeze_(dim=0)
    res = res * 0.6 + img_tensor * 0.4
    res = transforms.ToPILImage()(res)
    res = transforms.Resize((ori_h, ori_w))(res)
    res.save('data/cam.jpg')
    cam = cv2.imread('data/cam.jpg')
    hm = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cv2.imwrite('data/cam.jpg', hm)


def main():
    get_cam('data/car1.jpg')  # 输入你的图片路径


if __name__ == '__main__':
    main()
