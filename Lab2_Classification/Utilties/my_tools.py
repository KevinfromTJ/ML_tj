from Utilties.setup import import_torch
import_torch() # 导入torch位置到sys环境变量中
import os

from torchvision import datasets


def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir) #作用：返回一个ImageFolder对象，该对象包含了所有图片的路径和标签
    return all_data.classes

def get_images_recur(folder_path):
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('jpeg', 'png', 'jpg')):
                images.append(os.path.join(root, file))
    return images