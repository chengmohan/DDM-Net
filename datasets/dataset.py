from os import listdir
from os.path import join

import torch.utils.data as data
from libtiff import TIFFfile
from PIL import Image
import  numpy as np
import random
from My_function import reorder_imec, mask_input
from random import randint
#判断图像的扩展名是不是.png,.jpg,.jpeg,.tif
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])
#将tiff文件读取，最终读取成ndarray类型
def load_img(filepath):
    # img = Image.open(filepath+'/1.tif')
    # y = np.array(img).reshape(1,img.size[0],img.size[1])
    # m = np.tile(y, (2, 1, 1))
    # tif = TIFFfile(filepath+'/IMECMine_D65.tif')
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].transpose(2, 1, 0)
    # img_test = Image.fromarray(img[:,:,1])
    return img
#随机裁剪将图片a从随机位置裁剪为crop_size*crop_size大小的图片
def randcrop(a, crop_size):
    [wid, hei, nband]=a.shape
    crop_size1 = crop_size
    Width = random.randint(0, wid - crop_size1 - 1)
    Height = random.randint(0, hei - crop_size1 - 1)

    return a[Width:(Width + crop_size1),  Height:(Height + crop_size1), :]
#计算有效的剪裁尺寸
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)
#定一个一个来自文件夹的类 继承 data.Dataset 自定义一个数据集 
class DatasetFromFolder(data.Dataset):
    #本身 图像目录 旗帜 输入变形，目标变形 声明
    def __init__(self, image_dir, norm_flag, input_transform=None, target_transform=None, augment=False):
        super(DatasetFromFolder, self).__init__()
        # print(listdir(image_dir))
        # for y in listdir(image_dir):
        #     print(y)
        #     if is_image_file(y):
        #         print(y)
        #print(join(image_dir, x))
        # self.image_filenames = [join(image_dir, x, x) for x in listdir(image_dir)]
        #从一个图像文件夹下面读取 所有图片 sorted的目的是将图像排序
        #生成列表里面包含了所有波段文件的名字
        self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir))]
        #self.image_filenames1 = [join(image_dir, x) for x in listdir(image_dir1)]
        print(self.image_filenames)
        #将不同图片的文件打乱
        random.shuffle(self.image_filenames)
        #输入图像名字
        print(self.image_filenames)
        #ToDo 确认这里是否需要随机打乱文件，由于不同光照的存在
        #裁剪成和光谱波段数整数倍的大小
        self.crop_size = calculate_valid_crop_size(128, 4)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.augment = augment
        self.norm_flag = norm_flag
    #允许对象以 a[index]的方式调用该函数
    def __getitem__(self, index):
        # input_image = load_img(self.image_filenames[index]) # 512 512 16
        #按照传入index加载数据，加载特定波段的图像，这只能加载npy后缀的图片
        input_image  = np.load(self.image_filenames[index])
        #把图像转化为32位的浮点数据类型 这个可能用不了了
        input_image = input_image.astype(np.float32)
        %判断是否将图像归一化
        if self.norm_flag:
            norm_name = 'maxnorm'
            #得到这个图像像素的最大值
            max_raw = np.max(input_image)
            max_subband = np.max(np.max(input_image, axis=0), 0)
            norm_factor = max_raw / max_subband
            for bn in range(16):
                input_image[:, :, bn] = input_image[:, :, bn] * norm_factor[bn]
        #将图像随机裁剪为相应的大小
        input_image = randcrop(input_image, self.crop_size)
        #图像的一些变化
        if self.augment:
            if np.random.uniform() < 0.5:
                input_image = np.fliplr(input_image)
            if np.random.uniform() < 0.5:
                input_image = np.flipud(input_image)
            # ToDo 增强方式是否足够
            input_image = np.rot90(input_image, k=np.random.randint(0, 4))
        
        target = input_image.copy()
        #ToDo 确认这里的mask
        ###原本的im_gt_y按照实际相机滤波阵列排列 对多光谱图像进行采样
        input_image = mask_input(target,4)
        ###按照实际相机滤波阵列排列逆还原为从大到小的顺序
        input_image = reorder_imec(input_image) # sparase_image
        target = reorder_imec(target)  # multi-spectral
        random_index = randint(0, 15)

        if self.input_transform:
            #将矩阵沿着波长方向累加
            raw = input_image.sum(axis=2)   #  how sum(axis=0 or 2)  2
            raw = self.input_transform(raw)  #this
            sparse_image = input_image[:, :, random_index]  # ? channel axis 0?
            sparse_image = np.expand_dims(sparse_image, 0) #稀疏图像是一个二维的转换为3维 矩阵变为 1 高 长
            sparse_image = sparse_image.transpose(1,2,0) #变为和原始数据一样的 高 长 波段
            sparse_image = self.input_transform(sparse_image)
            input_image = self.input_transform(input_image)   # sparase_image
            # print(input_image.size())
            # print(raw.size())
        if self.target_transform:
            target_PPI = target.sum(axis=2)/16.0
            target_PPI = self.target_transform(target_PPI)
            target_demosaic = target[ :, :,random_index]
            target_demosaic = np.expand_dims(target_demosaic, 0)
            target_demosaic = target_demosaic.transpose(1, 2, 0)
            target_demosaic = self.target_transform(target_demosaic)
            target = self.target_transform(target)

        # random_index = randint(0, 15)


        return raw,target_PPI,sparse_image,target_demosaic
        # return raw, input_image, target

    def __len__(self):
        return len(self.image_filenames)

if __name__ == '__main__':
    root = '/media/ssd1/zyg/'
    dataset = DatasetFromFolder(root)
    train_loader,_ = dataset.loaders(batch_size=32)
    for data in train_loader:
        mosaicked,ref = data
        print(mosaicked.shape)
