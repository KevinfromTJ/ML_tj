"""
使用深度神经网络模型的数据预处理
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from torchvision import transforms as T

from Utilties.my_tools import get_classes, get_images_recur




class MyDataset(Dataset):
    def __init__(self, data_dir, transform_train_list=None, transform_test_list=None):
        self.data_dir = data_dir
        self.transform_train_list = transform_train_list
        self.transform_test_list = transform_test_list


        self.len="not calculated"
        self.train_data = None
        self.test_data = None

        mean, std = self._cal_mean_std()

        transform_train_list.append(T.Normalize(mean, std))
        transform_train_list.append(T.RandomErasing(p=0.2, value="random"))
        transform_test_list.append(T.Normalize(mean, std))

        if self.transform_train_list:
            self.train_data = datasets.ImageFolder(
                os.path.join(data_dir, "train/"),
                transform=T.Compose(transform_train_list),
            )
        if self.transform_test_list:
            self.test_data = datasets.ImageFolder(
                os.path.join(data_dir, "test/"),
                transform=T.Compose(transform_test_list),
            )

        self.val_data = None
        
        self.mean = mean
        self.std = std

    # def __getitem__(self, index):
    #     image_path = os.path.join(self.data_dir, self.images[index])
    #     image = Image.open(image_path)
    #     if self.transform:
    #         image = self.transform(image)
    #     return image

    def __len__(self):
        return self.len
    
    def __str__(self):
        print("MyDataset metadata dir: ", self.data_dir)
        print("MyDataset data length: ", self.len)
        print("MyDataset data mean: ", self.mean)
        print("MyDataset data std: ", self.std)
        return "MyDataset"
    


    def _cal_mean_std(self):
        """
        计算数据集的均值和方差
        :return:
        """

        image_paths = get_images_recur(self.data_dir)

        # 计算均值与方差
        self.len=0
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for image_path in image_paths:
            self.len+=1
            image = Image.open(image_path)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image)
            mean += torch.mean(image, dim=(1, 2))
            std += torch.std(image, dim=(1, 2))
        mean /= len(self)
        std /= len(self)

        # 缩放到[0,1]之间(好像已经缩放了？)
        # mean /= 255.0
        # std /= 255.0

        return mean, std
