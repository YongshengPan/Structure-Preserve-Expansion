# coding=utf-8
import torch
from torch.utils.data import Dataset, DataLoader
from myDataset import NiiDataset
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from model import Trusteeship, Generic_UNetwork, AdverserialNetwork
from datetime import datetime
import json
from torch.nn.parallel import DataParallel
import config

def test_model():
    from torchviz import make_dot
    model = Generic_UNetwork(1, 1, basedim=64, downdepth=3, model_type='3D',
                                         isresunet=True, use_triD=False, activation_function='relu')
    # In = torch.rand(1, 1, 16, 32, 32)
    In = torch.rand(1, 1, 32, 64, 64)
    out = model(In)[1]
    print(out.shape)
    # # 生成计算图
    # dot = make_dot(out, params=dict(model.named_parameters()))
    # dot.format = "png"
    # dot.render("simple_net_graph")  # 会输出 simple_net_graph.png


def train_g1(dataloader, module):
    size = len(dataloader.dataset)
    print(size)
    module.train()
    # metric names and storage
    # metric_keys = [
    #     'D1', 'G1', 'gen1', 'fm1', 'grad1', 'D2', 'G2', 'gen2', 'fm2', 'grad2'
    # ]
    metric_keys = [
        'G1', 'D1', 'gen1', 'fm1', 'grad1'
    ]
    # initialize lists for each metric
    metrics = {k: [] for k in metric_keys}

    for batch_idx, batch in enumerate(dataloader):
        datadict = {k: v.to(device) for k, v in batch.items()}
        # collect per-module outputs
        batch_vals = {k: [] for k in metric_keys}
        # print(batch_vals)
        out = module.train_step(datadict)
        # print(out)
        for key, val in zip(metric_keys, out):
            # print(key, val, type(val))
            batch_vals[key].append(val.item())
        # compute mean per batch and store
        for key in metric_keys:
            metrics[key].append(np.mean(batch_vals[key]))
        # periodic logging
        if batch_idx % 1000 == 0:
            # current = batch_idx * dataloader.batch_size
            # print(f"Batch {batch_idx} [{current}/{size}]")
            for key in metric_keys:
                vals = batch_vals[key]
                # print(f"  {key}: {', '.join(f'{v:7f}' for v in vals)}")
    # compute and print overall means
    means = {k: '%7f' % np.mean(metrics[k]) for k in metric_keys}
    print(f"==== {module.ckpt_prefix} Summary ====")
    for key in metric_keys:
        print(f"Mean {key}: {means[key]}")
    # return as tuple in order
    return [means[k] for k in metric_keys]
    # return tuple([ [means[k]] for k in metric_keys ])


def train(dataloader, module, freeze_model):
    size = len(dataloader.dataset)
    print(size)
    module.train()
    # metric names and storage
    # metric_keys = [
    #     'D1', 'G1', 'gen1', 'fm1', 'grad1', 'D2', 'G2', 'gen2', 'fm2', 'grad2'
    # ]
    metric_keys = [
        'G1', 'D1', 'gen1', 'fm1', 'grad1'
    ]
    # initialize lists for each metric
    metrics = {k: [] for k in metric_keys}

    for batch_idx, batch in enumerate(dataloader):
        datadict = {k: v.to(device) for k, v in batch.items()}
        # collect per-module outputs
        batch_vals = {k: [] for k in metric_keys}
        # print(batch_vals)
        out = module.train_step(datadict, freeze_model)
        # print(out)
        for key, val in zip(metric_keys, out):
            # print(key, val, type(val))
            batch_vals[key].append(val.item())
        # compute mean per batch and store
        for key in metric_keys:
            metrics[key].append(np.mean(batch_vals[key]))
        # periodic logging
        if batch_idx % 1000 == 0:
            # current = batch_idx * dataloader.batch_size
            # print(f"Batch {batch_idx} [{current}/{size}]")
            for key in metric_keys:
                vals = batch_vals[key]
                # print(f"  {key}: {', '.join(f'{v:7f}' for v in vals)}")
    # compute and print overall means
    means = {k: '%7f' % np.mean(metrics[k]) for k in metric_keys}
    print(f"==== {module.ckpt_prefix} Summary ====")
    for key in metric_keys:
        print(f"Mean {key}: {means[key]}")
    # return as tuple in order
    return [means[k] for k in metric_keys]
    # return tuple([ [means[k]] for k in metric_keys ])

# Initialize a weight list to store (loss, file name)
ckpt_prefixs = ['resUnet']
# best_weights = {ckpt_prefixs[0]:[],ckpt_prefixs[1]:[]}
best_weights = {ckpt_prefixs[0]:[]}

def save_weight_with_loss(loss, t, trustship):
    global best_weights
    # print(best_weights)
    best_weight = best_weights[str(trustship.ckpt_prefix)]
    # Define the current weight file name
    # If the list is full and the current loss is smaller than the maximum loss, replace it
    if len(best_weight) == 5:
        # Find the maximum loss and its corresponding file name in the list
        max_loss, max_loss_file = max(best_weight, key=lambda x: float(x[0][0]))
        # If the current loss is smaller, replace the maximum loss entry
        if loss < max_loss:
            # Delete the old weight file
            if os.path.exists(max_loss_file):
                os.remove(max_loss_file)
            # Remove the maximum loss record from the list
            best_weight.remove((max_loss, max_loss_file))
            # Add the current weight record
            max_loss_file = trustship.save_dict(dict_name='chkpt_%d.h5' % (t + 1))  # save .h5
            best_weight.append((loss, max_loss_file))
    else:
        # If the list is not full, add directly
        best_weight.append((loss, trustship.save_dict(dict_name='chkpt_%d.h5' % (t + 1))))
    # Sort by loss to ensure the list is ordered
    # best_weight.sort(key=lambda x: x[0])
    best_weight.sort(key=lambda x: float(x[0][0]))

# File to store the best_weights list
BEST_WEIGHTS_FILE = config.BEST_WEIGHTS_FILE
def save_best_weights():
    """Save best_weights to a JSON file."""
    with open(BEST_WEIGHTS_FILE, "w") as f:
        json.dump(best_weights, f)


# test_model()
if __name__ == '__main__':
    gen_basedim = config.gen_basedim
    adv_basedim = config.adv_basedim
    device_ids = config.device_ids
    device = torch.device(f"cuda:{device_ids[0]}")  # 主卡设为 1
    gen = Generic_UNetwork(1, 1, basedim=gen_basedim, downdepth=3, model_type='3D',
                           isresunet=True, use_triD=False, activation_function=None)
    gen = DataParallel(gen, device_ids=device_ids)  # 额外加这一行
    adv = AdverserialNetwork(1, basedim=adv_basedim, downdepth=3, model_type='3D',
                             activation_function=None)
    adv = DataParallel(adv, device_ids=device_ids)  # 额外加这一行
    gan = Trusteeship(gen, loss_fn=('mae', 'msl', 'thd'),
                                volin=('CT1',), volout=('CT2',), metrics=('thd',),
                                advmodule=adv,
                                device=device, ckpt_prefix=ckpt_prefixs[0])
    # gan_model_res.load_dict("CTtoCT_res_Ncard_chkpt_568.h5")
    all_trustships = [gan]
    folder_path = config.folder_path  # data
    json_path = config.json_path
    dataloader = DataLoader(NiiDataset(folder_path, json_path), drop_last=False, shuffle=True,
                                        batch_size=config.batch_size,
                                        prefetch_factor=config.prefetch_factor,
                                        num_workers=config.num_workers
                            )
    epochs = config.epochs
    start_epochs = config.start_epochs # 上一次服务器断电，到
    # 中断后继续训练
    if start_epochs == 0:
        # 创建 TensorBoard 的 SummaryWriter 实例
        # logs这里可以加时间戳，这样就不用自己调了  tensorboard --logdir=./logs/2025-05-19-12-30
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")  # 获取当前时间并格式化
        writer = SummaryWriter(log_dir=f'./logs/{current_time}')  # 设置日志目录，TensorBoard 会在该目录下生成数据
    else:
        continue_time = "2025-06-11-17-49"  # need to change
        for trustship in all_trustships:
            trustship.to_device(device)
            trustship.load_dict(f"{trustship.ckpt_prefix}_chkpt_{start_epochs}.h5")
        # start_epochs：上次训练已跑到的 epoch 数
        writer = SummaryWriter(
            log_dir=f"./logs/{continue_time}",
            purge_step=start_epochs
        )

    # keys = ['G1', 'D1', 'gen1', 'fm1', 'grad1', 'G2', 'D2', 'gen2', 'fm2', 'grad2']
    keys = ['G1', 'D1', 'gen1', 'fm1', 'grad1']

    # todo: 这里加载模型，传入train中使用
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

    for epoch in tqdm(range(start_epochs, epochs), desc="Epochs"):
        print(f"Epoch {epoch}\n{'-' * 25}")
        for module in all_trustships:
            metrics = train(dataloader, module, freeze_model)
            # save and update weights for each module
            save_weight_with_loss(metrics, epoch, module)
            save_best_weights()
            # log all metrics
            for idx, value in enumerate(metrics, start=1):
                name = keys[idx - 1]  # reuse keys from train
                writer.add_scalar(f"Loss/{module.ckpt_prefix}/{name}", value, epoch)
    writer.close()
    print("Done!")
