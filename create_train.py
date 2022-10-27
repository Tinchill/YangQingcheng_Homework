'''
测试接口：create_train.py
   运行此.py文件,建立训练集: ./detection/sysu_train,
                         ./detection/sysu_train_merged
'''

import os
import glob
import cv2
import numpy as np
import csv
import random
random.seed(0)

def add_alpha_channel(img):
    """ 为 jpg 图像添加 alpha 通道 """
 
    b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
 
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    return img_new
 
def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    """ 将 png 透明图像与 jpg 图像叠加
        y1,y2,x1,x2为叠加位置坐标值
    """
    
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    
    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
 
    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]
 
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_jpg = 1 - alpha_png
    
    # 开始叠加
    for c in range(0,3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg*jpg_img[y1:y2,x1:x2,c]) + (alpha_png*png_img[yy1:yy2,xx1:xx2,c]))
 
    return jpg_img
 
    
 
if __name__ == '__main__':
    
    location = './detection/sysu'
    if not os.path.exists(location + '_train/images/'):
        os.makedirs(location + '_train/images/')
    
    with open(location + '_train/label.csv',"w", newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["img_name","label","xmin","ymin","xmax",'ymax'])
        #写入多行用writerows

    files = glob.glob('./detection/background/*.jpg')
    for ct, name in enumerate(files):
        while 1:
            name = name.replace('\\', '/')
            img_jpg = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            img_jpg = cv2.resize(img_jpg, (256, 256))
        
            alpha = random.uniform(0.05, 0.5)   # 0.1 ~ 1
            x1 = random.randint(0, 256)  # 0 ~ 256
            y1 = random.randint(0, 256)  # 0 ~ 256
            
            # label 0
            label = 0
            img_png = cv2.imread('./detection/target/' + str(label) + '.png', cv2.IMREAD_UNCHANGED)
            w, h, _ = img_png.shape
            if w > h:
                img_png = cv2.resize(img_png, (int(alpha*h*256./w), int(alpha*256)))
            else:
                img_png = cv2.resize(img_png, (int(alpha*256), int(alpha*h*256./w)))
    
    
            # 设置叠加位置坐标
            x2 = x1 + img_png.shape[1]
            y2 = y1 + img_png.shape[0]
            
            if x2 > img_jpg.shape[0] or y2 > img_jpg.shape[1]:
                continue
            else:
                break
     
        # 开始叠加
        res_img = merge_img(img_jpg, img_png, y1, y2, x1, x2)
     
        cv2.imwrite(location + '_train/images/' + name.split('/')[-1], res_img)
        with open(location + '_train/label.csv',"a",newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([name.split('/')[-1], label, x1, y1, x2, y2])
      
        if ct == 999:
            break
        
        
        
    
