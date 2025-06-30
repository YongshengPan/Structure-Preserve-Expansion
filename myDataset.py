# coding=utf-8
import torch
from torch.utils.data import Dataset, DataLoader
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


class NiiDataset(Dataset):
    def __init__(self, folder_path, json_path):
        self.folder_path = folder_path
        # 读取 dataset_split.json
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.file_names = []
        for entry in data:
            if entry['name'] == 'Healthy-Total-Body-CTs-UpperBody':
                self.file_names.extend(entry['train'] * 80)  # 改 80 --> 30 ?
            else:
                self.file_names.extend(entry['train'])
            # self.file_names.extend(entry['test'])  # extend 是展开后加入到列表中
        # 去掉最前面的 './'，保留后面的子路径
        self.file_names = [os.path.join(self.folder_path, f.lstrip('./')) for f in self.file_names]
        # 筛一下大小，不够扩展一次的数据不要
        small = []
        for i in self.file_names:
            temp = sitk.ReadImage(i, outputPixelType=sitk.sitkInt16)
            size = temp.GetSize()
            if size[2] < 64  or size[0] < 64 or size[1] < 64:  # 筛选条件
                small.append(i)
        for j in small:
            self.file_names.remove(j)
        print(len(self.file_names))
        # self.file_names = self.file_names[:1]
        # print(self.file_names[0])
        # self.data_pool = {}
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        CTIMG = sitk.ReadImage(file_name, outputPixelType=sitk.sitkInt16)
        # print(file_name, CTIMG.GetSize())
        numpy_image = sitk.GetArrayFromImage(CTIMG)  # 转为numpy
        # print(np.min(numpy_image), np.max(numpy_image))
        # input_image为[D, H, W]，现对其进行处理
        # todo: 需要修改 patch_size 和 gap
        # patch_size = (128, 224, 224)
        patch_size = (32, 64, 64)
        gap = 32  # 输入数据在深度上滑动的距离为gap，得到输出数据
        input_tensor, target_tensor = self.extract_patch(numpy_image, patch_size=patch_size, gap=gap, max_iters=50)
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_tensor).float().unsqueeze(0)
        # extend2_tensor = torch.from_numpy(extend2_tensor).float().unsqueeze(0)
        # dic = {'CT1': torch.randn(1, 64, 256, 256), 'CT2': torch.randn(1, 64, 256, 256)}
        # dic = {'CT1': input_tensor, 'CT2': target_tensor, 'CT3': extend2_tensor}
        dic = {'CT1': input_tensor, 'CT2': target_tensor}
        return dic


    def extract_patch(self, data, patch_size, gap, max_iters=50):
        # print("after resize", data.shape, type(data), "-----")
        D, H, W = data.shape
        # print(D, H, W)
        patch_d, patch_h, patch_w = patch_size
        # 随机选择第一个子块的起始位置 (a, b, c)，确保子块不会超出数据边界
        a = np.random.randint(0, D - patch_d + 1 - gap)  # 随机选择 D 维的起始位置，确保滑动后子块不超出边界
        b = np.random.randint(0, H - patch_h + 1)  # 随机选择 H 维的起始位置
        c = np.random.randint(0, W - patch_w + 1)  # 随机选择 W 维的起始位置
        # 第一个子块位置
        first_patch = data[a:(a+patch_d), b:(b+patch_h), c:(c+patch_w)]
        # 计算第二个子块的位置 (a+gap, b, c)
        second_patch = data[a+patch_d:(a+gap+patch_d), b:(b+patch_h), c:(c+patch_w)]
        # second_patch = data[a+gap:(a+gap+patch_d), b:(b+patch_h), c:(c+patch_w)]
        first_patch, second_patch = self.divide1000_normalize(first_patch), self.divide1000_normalize(second_patch)
        return first_patch, second_patch

    def divide1000_normalize(self, numpy_data):
        numpy_data = np.clip(numpy_data, a_min=-1000, a_max=3000)
        normalized_data = numpy_data / 1000 + 1
        return normalized_data

def load_nifti_sitk(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # Shape: [Z, Y, X]
    return arr

def show_3_slices(ct_array, title="CT Slices"):
    ct_array = ct_array.squeeze().detach().cpu().numpy() # tensor --> array

    ct_array = ct_array[::-1, :, :]  # 变为头朝上
    z = ct_array.shape[0] // 2
    y = ct_array.shape[1] // 2
    x = ct_array.shape[2] // 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(ct_array[z, :, :], cmap="gray")
    axs[0].set_title('Axial (Z)')

    axs[1].imshow(ct_array[:, y, :], cmap="gray")
    axs[1].set_title('Coronal (Y)')

    axs[2].imshow(ct_array[:, :, x], cmap="gray")
    axs[2].set_title('Sagittal (X)')

    for ax in axs:
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    folder_path = "/media/userdisk0/zyliu/Data/normalizeDataSpacing4/"
    json_path = 'dataset_split_grouped.json'  # 替换为你的文件路径
    # folder_path = r'D:\MachineLearning\nnunet-learn\train_images_resampled'  # 替换为你的文件夹路径
    dataLoader = DataLoader(NiiDataset(folder_path, json_path), batch_size=1,
                                        shuffle=True, prefetch_factor=2, num_workers=4)
    for batch_idx, dic in tqdm(enumerate(dataLoader)):
        show_3_slices(dic['CT1'], title='show1')
        show_3_slices(dic['CT2'], title='show2')
        # show_3_slices(dic['CT3'], title='show3')
        # shape1, shape2, shape3 = dic['CT1'].shape, dic['CT2'].shape, dic['CT3'].shape
        # print(batch_idx, shape1, shape2, shape3)
        # shape1, shape2 = dic['CT1'].shape, dic['CT2'].shape
        # print(batch_idx, shape1, shape2)
        break
