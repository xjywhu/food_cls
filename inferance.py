'''
    请大家自行实现测试代码，注意提交格式
'''
import os

import torch

from dataset import Food_LT
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
# from model import resnet34
from resnet50 import ResNet50
model = ResNet50()
checkpoint = torch.load('/home/zhangzhengxin/PythonProject/MachineLearning/food_cls/ckpt/model_best.pth.tar')
# start_epoch=checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict_model'])
# model.load_state_dict()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

def predict_img(image):
    image_tensor = transform_test(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor).to('cpu')
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

root = '/home/zhangzhengxin/PythonProject/MachineLearning/food_cls/data/food/test/test/'
f = open('submission.txt','w')
f.write('Id, Expected\n')
for filename in os.listdir(root):
    image = Image.open(root+filename)
    print(filename)
    res = filename + ', ' + str(predict_img(image))+'\n'

    f.write(res)

