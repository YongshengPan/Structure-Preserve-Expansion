# coding=utf-8
import os

import SimpleITK as sitk
import torch
from tqdm import tqdm
from model import Generic_UNetwork, Trusteeship, AdverserialNetwork
import matplotlib.pyplot as plt
import numpy as np


# 读取 NIfTI 文件并转换为 Tensor
def read_nifti_to_tensor(file_path):
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)  # 转换为 NumPy 数组
    return img_array, img  # 返回图像数组和原始 SimpleITK 图像


# 保存 Tensor 为 NIfTI 文件
def save_tensor_as_nifti(tensor, original_img, output_path):
    new_array = tensor.cpu().detach().numpy()  # 转换为 NumPy 数组
    new_image = sitk.GetImageFromArray(new_array)  # 转换为 SimpleITK 图像
    # 保持原始的 spacing, origin 和 direction
    print(original_img.GetSpacing(), original_img.GetOrigin(), original_img.GetDirection())
    new_image.SetSpacing(original_img.GetSpacing())
    new_image.SetOrigin(original_img.GetOrigin())
    new_image.SetDirection(original_img.GetDirection())
    sitk.WriteImage(new_image, output_path)

def save_array_as_nifti(array, original_img, output_path):
    new_image = sitk.GetImageFromArray(array)  # 转换为 SimpleITK 图像
    # 保持原始的 spacing, origin 和 direction
    print(original_img.GetSpacing(), original_img.GetOrigin(), original_img.GetDirection())
    new_image.SetSpacing(original_img.GetSpacing())
    new_image.SetOrigin(original_img.GetOrigin())
    new_image.SetDirection(original_img.GetDirection())
    sitk.WriteImage(new_image, output_path)

def divide1000_normalize(tensor_data):
    normalized_data = tensor_data / 1000 + 1
    return normalized_data

def multi1000(numpy_data):
    normalized_data = (np.clip(numpy_data, 0, 4) - 1) * 1000
    return normalized_data

def show_3_slices(ct_array, title="CT Slices"):
    # ct_array = ct_array.squeeze().detach().cpu().numpy() # tensor --> array
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


def generate_slices(dim_size, patch_len, step, start=0):
    """
    返回 [(start, end), ...] 切片区间列表
    参数:
        dim_size: 总体长度（如图像的某个维度）
        patch_len: 每个 patch 的长度
        step: 滑动步长
        start: 起始位置
    """
    slices = []
    pos = start
    while pos + patch_len < dim_size:
        slices.append((pos, pos + patch_len))
        pos += step
    # 最后一块保证覆盖末尾
    slices.append((dim_size - patch_len, dim_size))
    return slices

def pad_z_to_200_tail(img_array, target_z=200):
    z, y, x = img_array.shape
    if z >= target_z:
        return img_array[:target_z, :, :]  # 只保留前200层
    else:
        pad_after = target_z - z
        padded = np.pad(img_array,
                        ((0, pad_after), (0, 0), (0, 0)),
                        mode='constant', constant_values=0)
        return padded


if __name__ == '__main__':
    input_nifti = ""
    output_nifti = ""  # 输出的 NIfTI 文件路径

    img_array, original_img = read_nifti_to_tensor(input_nifti)
    print(original_img.GetSize()) # 举例一个 120 层的 3D 图像
    # img_array = pad_z_to_200_tail(img_array, target_z=300)

    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    # print(torch.min(img_tensor), torch.max(img_tensor))
    img_tensor = divide1000_normalize(img_tensor)
    # print(torch.min(img_tensor), torch.max(img_tensor))
    start_slice = 420 # 410, 84
    img_tensor[start_slice:, :, :] = 0  # start_slice以上部分清零
    # print(torch.min(img_tensor), torch.max(img_tensor))
    image_array = img_tensor.cpu().detach().numpy()  # 转回numpy
    D,H,W = img_array.shape

    patch_size = (Pd, Ph, Pw) = (32, 64, 64)
    # patch_size = (Pd, Ph, Pw) = (32, 256, 256)
    sd, sh, sw = Pd, Ph//2, Pw//2
    # 假设模型输出也是单通道，若多通道则扩展第一维
    output = np.zeros((D, H, W), dtype=np.float32)
    weight = np.zeros((D, H, W), dtype=np.float32)

    # 示例
    d_slices = generate_slices(D-patch_size[0], Pd, sd, start_slice-patch_size[0])[:1]
    h_slices = generate_slices(H, Ph, sh)
    w_slices = generate_slices(W, Pw, sw)
    print(d_slices, h_slices, w_slices)

    output[:start_slice, :, :] = image_array[:start_slice, :, :]
    show_3_slices(output)
    device = torch.device("cuda:0")

    freeze_model = Generic_UNetwork(1, 1, basedim=64, downdepth=3, model_type='3D',
                                    isresunet=True, use_triD=False, activation_function=None)
    state_dict = torch.load(os.path.join("weightsTotal2", '_'.join(('mae', 'msl', 'thd')),
                                         "resUnet", "resUnet_chkpt_982.h5"),
                            weights_only=False, map_location='cpu')
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")  # 删除 "module." 前缀
        new_state_dict[new_k] = v
    freeze_model.load_state_dict(new_state_dict, strict=True)
    freeze_model.to(device).eval()

    # 创建并加载模型
    gan_model_res = Generic_UNetwork(1, 1, basedim=64, downdepth=3, model_type='3D',
                                    isresunet=True, use_triD=False, activation_function=None)
    state_dict = torch.load(os.path.join("weightsTotal3", '_'.join(('mae', 'msl', 'thd')),
                                         "resUnet", "resUnet_chkpt_844.h5"),
                            weights_only=False, map_location='cpu')
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")  # 删除 "module." 前缀
        new_state_dict[new_k] = v
    gan_model_res.load_state_dict(new_state_dict, strict=True)
    gan_model_res.to(device).eval()

    with torch.no_grad():
        # for d_start, d_end in tqdm(d_slices):
        for d_start, d_end in d_slices:
            for h_start, h_end in h_slices:
                for w_start, w_end in w_slices:
                    # 1) 从原始 volume 载入 patch
                    patch = output[
                            d_start:d_end, h_start:h_end, w_start:w_end
                            ]  # shape (Pd, Ph, Pw)
                    # 2) 推理
                    inp = torch.from_numpy(patch[None, None]).to(device)  # [B=1,C=1,Pd,Ph,Pw]
                    inp = inp.to(torch.float32)

                    pred = freeze_model(inp)[1].to(device)

                    pred = pred.cpu().numpy()[0, 0]  # shape (Pd,Ph,Pw)
                    # show_3_slices(pred)
            #         break
            #     break
            # break
                    # 3) 写入 output 和 weight
                    d_overlap = patch_size[0] - sd
                    # if d_start == d_slices[0][0]:  # 第一次扩，重叠部分取上一次的，无需取平均
                    ds = slice(d_start + d_overlap + sd, d_end + sd)
                    pd_slice = slice(d_overlap, patch_size[0])
                    # 平面方向：累加 + 权重
                    h_slice = slice(h_start, h_end)
                    w_slice = slice(w_start, w_end)
                    # print("---", ds, h_slice, w_slice, pd_slice, "---")
                    output[ds, h_slice, w_slice] = 0
                    # output[ds, h_slice, w_slice] += pred[pd_slice, :, :]

                    output[ds, h_slice, w_slice] += pred[:, :, :]

    print("Final result shape:", output.shape)
    output = multi1000(output)
    show_3_slices(output)
    show_3_slices(img_array)
    # save_array_as_nifti(output, original_img, output_nifti)
