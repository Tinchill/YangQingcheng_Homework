'''
测试接口：test_port.py
   可以运行此.py文件,快速使用预训练好的模型查看效果
'''

import glob
import torch
import torchvision
from visualize import predict, display
from model import TinySSD
_device = 'cpu'

files = glob.glob('detection/test/*.jpg')

testnet = TinySSD(num_classes=1)
testnet = testnet.to(_device)
model_location = './pretrained/'
model_pkl = 'net_SGD_lr=0.2_50'

if __name__ == '__main__':
    for file_idx in range(len(files)):
        X = torchvision.io.read_image(files[file_idx]).unsqueeze(0).float()
        img = X.squeeze(0).permute(1, 2, 0).long()

        testnet.load_state_dict(torch.load(model_location + model_pkl + '.pkl'))
        testnet.eval()
        output = predict(X, testnet, _device)
        display(img, output.cpu(), 0.6, model_pkl, file_idx)

