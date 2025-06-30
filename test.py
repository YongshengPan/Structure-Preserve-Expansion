# coding=utf-8
import SimpleITK as sitk
import torch
from model import Generic_UNetwork, Trusteeship, AdverserialNetwork
from torch.utils.data import DataLoader
from myDataset import NiiDataset

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error as mse

# 读取 NIfTI 文件并转换为 Tensor
def read_nifti_to_tensor(file_path):
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)  # 转换为 NumPy 数组
    return img_array, img  # 返回图像数组和原始 SimpleITK 图像


def calculate_metrics(y_pred, y_true):
    """
    计算MAE、PSNR、RMSE、SSIM和Dice系数。
    参数：
    y_pred (torch.Tensor): 预测的3D影像，形状为(N, D, H, W)。
    y_true (torch.Tensor): 真实的3D影像，形状为(N, D, H, W)。
    返回：
    dict: 包含各项指标的字典。
    """
    # 将张量转换为NumPy数组
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    # 初始化指标
    mae_list = []
    psnr_list = []
    rmse_list = []
    ssim_list = []
    # dice_list = []

    # 遍历每个样本
    for i in range(y_pred.shape[0]):
        pred = y_pred[i]
        true = y_true[i]

        # 计算MAE
        mae = np.mean(np.abs(pred - true))
        mae_list.append(mae)

        # 计算PSNR
        psnr_value = psnr(true, pred, data_range=255)
        psnr_list.append(psnr_value)

        # 计算RMSE
        rmse_value = np.sqrt(mse(true.flatten(), pred.flatten()))
        rmse_list.append(rmse_value)

        # 计算SSIM
        ssim_value = ssim(true[0], pred[0], data_range=255)
        ssim_list.append(ssim_value)

    # 计算平均值
    metrics = {
        'MAE': np.mean(mae_list),
        'PSNR': np.mean(psnr_list),
        'RMSE': np.mean(rmse_list),
        'SSIM': np.mean(ssim_list),
        # 'Dice': np.mean(dice_list)
    }

    return metrics

def result(dataloader, modules):
    device = torch.device("cpu")
    essamble_metrics = {trustship.ckpt_prefix: [] for trustship in modules}
    # print(essamble_metrics)  # {'XraytoCT_mv_res': [], 'XraytoCT_mv_trans': [], 'XraytoCT_mv_res_trans': []}
    with (torch.no_grad()):
        for batch, dic in enumerate(dataloader):
            datadict = {it: dic[it].to(device) for it in dic}
            # print(datadict.keys())  # dict_keys(['CT1', 'CT2'])，其中CT1是输入，CT2是ground_truth，都是归一化到0-255的
            before_model, ground_truth = datadict["CT1"], datadict["CT2"]
            for trustship in modules:
                trustship.to_device(device)
                trustship.load_dict(f"{trustship.ckpt_prefix}_chkpt_1210.h5")
                after_model = trustship.module(before_model)[1]
                # todo: 各种指标计算，ssim, 等
                # matrics = calculate_metrics(after_model, ground_truth)
                # matrics = calculate_metrics(after_model[:,:,50: 96,:,:], ground_truth[:,:,50: 96,:,:])
                matrics = calculate_metrics(after_model, ground_truth)
                essamble_metrics[trustship.ckpt_prefix].append(matrics)
                # print(trustship.ckpt_prefix)
                # print(matrics)
    return essamble_metrics


if __name__ == '__main__':
    basedim = 16
    device = torch.device("cpu")
    gan_model_res = Trusteeship(Generic_UNetwork(1, 1, basedim=basedim, downdepth=5, model_type='3D',
                                                 isresunet=True, use_triD=False, activation_function='relu'),
                                loss_fn=('mae', 'msl'), volin=('CT1',), volout=('CT2',), metrics=('thd',),
                                advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D',
                                                             activation_function=None),
                                device=device, ckpt_prefix='XraytoCT_mv_res', )
    gan_model_trans = Trusteeship(Generic_UNetwork(1, 1, basedim=basedim, downdepth=5, model_type='3D',
                                                   istransunet=True, use_triD=False, activation_function='relu'),
                                  loss_fn=('mae', 'msl'), volin=('CT1',), volout=('CT2',), metrics=('thd',),
                                  advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D',
                                                               activation_function=None),
                                  device=device, ckpt_prefix='XraytoCT_mv_trans', )
    gan_model_res_trans = Trusteeship(Generic_UNetwork(1, 1, basedim=basedim, downdepth=5, model_type='3D',
                                                       isresunet=True, istransunet=True, use_triD=False,
                                                       activation_function='relu'),
                                      loss_fn=('mae', 'msl'), volin=('CT1',), volout=('CT2',), metrics=('thd',),
                                      advmodule=AdverserialNetwork(1, basedim=16, downdepth=4, model_type='3D',
                                                                   activation_function=None),
                                      device=device, ckpt_prefix='XraytoCT_mv_res_trans', )
    # all_trustships = [gan_model_res, gan_model_trans, gan_model_res_trans]
    all_trustships = [gan_model_res]
    batch_size = 1
    folder_path = r''  # data
    dataloader = DataLoader(NiiDataset(folder_path), batch_size=batch_size)
    performance = result(dataloader, all_trustships)
    print(performance)
    # print("Done!")

    """performance number"""
    import numpy as np

    # model_performance = {'XraytoCT_mv_res': [{'MAE': np.float32(6.736823), 'PSNR': np.float64(23.627190391757146), 'RMSE': np.float32(16.795025), 'SSIM': np.float64(0.7859781190296614), 'Dice': np.float64(0.9978365325480846)}, {'MAE': np.float32(6.353344), 'PSNR': np.float64(25.105124474417686), 'RMSE': np.float32(14.1672), 'SSIM': np.float64(0.7990695756855393), 'Dice': np.float64(0.9975874902665732)}, {'MAE': np.float32(5.150289), 'PSNR': np.float64(26.482969043380876), 'RMSE': np.float32(12.089038), 'SSIM': np.float64(0.8338211216653173), 'Dice': np.float64(0.9962567878610861)}, {'MAE': np.float32(8.0420885), 'PSNR': np.float64(24.729165115832945), 'RMSE': np.float32(14.793876), 'SSIM': np.float64(0.8355311102744097), 'Dice': np.float64(0.9952334262019681)}, {'MAE': np.float32(2.6861854), 'PSNR': np.float64(29.847862571348905), 'RMSE': np.float32(8.206294), 'SSIM': np.float64(0.8740062241448701), 'Dice': np.float64(0.9952263717509255)}, {'MAE': np.float32(1.8697972), 'PSNR': np.float64(36.84620076965463), 'RMSE': np.float32(3.6663184), 'SSIM': np.float64(0.9236887682441293), 'Dice': np.float64(0.9372677668249653)}, {'MAE': np.float32(4.8123393), 'PSNR': np.float64(26.582818810227735), 'RMSE': np.float32(11.950864), 'SSIM': np.float64(0.8268961944374335), 'Dice': np.float64(0.9986404378235313)}, {'MAE': np.float32(27.330498), 'PSNR': np.float64(15.48445098910058), 'RMSE': np.float32(42.886196), 'SSIM': np.float64(0.45448048464054175), 'Dice': np.float64(0.9194682603729891)}, {'MAE': np.float32(7.8802614), 'PSNR': np.float64(24.046248874205837), 'RMSE': np.float32(16.00397), 'SSIM': np.float64(0.7829506478440896), 'Dice': np.float64(0.995331633827697)}, {'MAE': np.float32(29.823174), 'PSNR': np.float64(15.322085334963322), 'RMSE': np.float32(43.695423), 'SSIM': np.float64(0.5166875837755048), 'Dice': np.float64(0.9125660007448897)}, {'MAE': np.float32(5.585149), 'PSNR': np.float64(25.581884966070284), 'RMSE': np.float32(13.410529), 'SSIM': np.float64(0.8086217527327557), 'Dice': np.float64(0.9985055549603467)}, {'MAE': np.float32(20.527843), 'PSNR': np.float64(17.607444955772003), 'RMSE': np.float32(33.586742), 'SSIM': np.float64(0.5605289542913766), 'Dice': np.float64(0.9262009809466618)}, {'MAE': np.float32(45.87804), 'PSNR': np.float64(13.543221802362563), 'RMSE': np.float32(53.626457), 'SSIM': np.float64(0.6362656771379321), 'Dice': np.float64(0.9990955320294957)}, {'MAE': np.float32(5.38869), 'PSNR': np.float64(26.779146011605288), 'RMSE': np.float32(11.683764), 'SSIM': np.float64(0.8496355179268046), 'Dice': np.float64(0.9885725624388176)}, {'MAE': np.float32(1.6189176), 'PSNR': np.float64(37.21745155696153), 'RMSE': np.float32(3.5129142), 'SSIM': np.float64(0.9201937063967202), 'Dice': np.float64(0.9513699218953665)}, {'MAE': np.float32(5.7354383), 'PSNR': np.float64(26.223834131808893), 'RMSE': np.float32(12.455135), 'SSIM': np.float64(0.8373894469118736), 'Dice': np.float64(0.9994807593028135)}, {'MAE': np.float32(1.5976143), 'PSNR': np.float64(37.07336441556673), 'RMSE': np.float32(3.571675), 'SSIM': np.float64(0.9195327636045242), 'Dice': np.float64(0.8597882084053526)}], 'XraytoCT_mv_trans': [{'MAE': np.float32(7.2442102), 'PSNR': np.float64(23.57335902060816), 'RMSE': np.float32(16.899439), 'SSIM': np.float64(0.7888077252725793), 'Dice': np.float64(0.9980335383662764)}, {'MAE': np.float32(4.8771105), 'PSNR': np.float64(25.751476677737788), 'RMSE': np.float32(13.151229), 'SSIM': np.float64(0.8095622444742608), 'Dice': np.float64(0.9978136762834657)}, {'MAE': np.float32(5.4842677), 'PSNR': np.float64(26.230657952386522), 'RMSE': np.float32(12.445353), 'SSIM': np.float64(0.8330375202430104), 'Dice': np.float64(0.9961385049212248)}, {'MAE': np.float32(5.3728685), 'PSNR': np.float64(26.722410646591513), 'RMSE': np.float32(11.760333), 'SSIM': np.float64(0.849933655786877), 'Dice': np.float64(0.9936310811322763)}, {'MAE': np.float32(2.7419543), 'PSNR': np.float64(29.59060534941102), 'RMSE': np.float32(8.452981), 'SSIM': np.float64(0.8711811433752484), 'Dice': np.float64(0.9952395675493355)}, {'MAE': np.float32(3.9576256), 'PSNR': np.float64(32.119527669924146), 'RMSE': np.float32(6.317769), 'SSIM': np.float64(0.8793808709933635), 'Dice': np.float64(0.8347436678083967)}, {'MAE': np.float32(4.965602), 'PSNR': np.float64(26.704882814656333), 'RMSE': np.float32(11.784088), 'SSIM': np.float64(0.8323277928636843), 'Dice': np.float64(0.998638956858005)}, {'MAE': np.float32(27.078953), 'PSNR': np.float64(15.556276034222233), 'RMSE': np.float32(42.533035), 'SSIM': np.float64(0.4538443184146622), 'Dice': np.float64(0.9194682603729891)}, {'MAE': np.float32(8.156632), 'PSNR': np.float64(23.904735160020554), 'RMSE': np.float32(16.266846), 'SSIM': np.float64(0.7900728309791468), 'Dice': np.float64(0.9953091675865691)}, {'MAE': np.float32(31.87626), 'PSNR': np.float64(14.718535273455231), 'RMSE': np.float32(46.839626), 'SSIM': np.float64(0.5030079080940066), 'Dice': np.float64(0.9125451941033306)}, {'MAE': np.float32(6.08266), 'PSNR': np.float64(25.162551890783046), 'RMSE': np.float32(14.07384), 'SSIM': np.float64(0.8082306406978068), 'Dice': np.float64(0.9985060589822664)}, {'MAE': np.float32(25.843105), 'PSNR': np.float64(15.608176462967238), 'RMSE': np.float32(42.27965), 'SSIM': np.float64(0.49351271788850987), 'Dice': np.float64(0.926619980719125)}, {'MAE': np.float32(50.0285), 'PSNR': np.float64(12.773533626714212), 'RMSE': np.float32(58.59539), 'SSIM': np.float64(0.6030662249797175), 'Dice': np.float64(0.9995246706888138)}, {'MAE': np.float32(3.7742846), 'PSNR': np.float64(27.71542502207045), 'RMSE': np.float32(10.489843), 'SSIM': np.float64(0.8506260399220504), 'Dice': np.float64(0.9890875377241779)}, {'MAE': np.float32(4.478162), 'PSNR': np.float64(31.707123258879935), 'RMSE': np.float32(6.6249714), 'SSIM': np.float64(0.8905060681236268), 'Dice': np.float64(0.9535997076337265)}, {'MAE': np.float32(4.62101), 'PSNR': np.float64(26.373982409424272), 'RMSE': np.float32(12.241681), 'SSIM': np.float64(0.8367529701496951), 'Dice': np.float64(0.9994766140747336)}, {'MAE': np.float32(1.370529), 'PSNR': np.float64(37.86882056515995), 'RMSE': np.float32(3.2591116), 'SSIM': np.float64(0.9351047536605932), 'Dice': np.float64(0.9277443273177275)}], 'XraytoCT_mv_res_trans': [{'MAE': np.float32(6.424522), 'PSNR': np.float64(23.79878201204302), 'RMSE': np.float32(16.466494), 'SSIM': np.float64(0.787051484868478), 'Dice': np.float64(0.9979779359524051)}, {'MAE': np.float32(5.454996), 'PSNR': np.float64(24.928989335646953), 'RMSE': np.float32(14.457415), 'SSIM': np.float64(0.7971007767417232), 'Dice': np.float64(0.9974865581743061)}, {'MAE': np.float32(6.1843567), 'PSNR': np.float64(25.580342277320803), 'RMSE': np.float32(13.4129095), 'SSIM': np.float64(0.8214161303769506), 'Dice': np.float64(0.9964486257033883)}, {'MAE': np.float32(7.048269), 'PSNR': np.float64(25.571175461769656), 'RMSE': np.float32(13.427075), 'SSIM': np.float64(0.8402944718477752), 'Dice': np.float64(0.9942167252450225)}, {'MAE': np.float32(2.6716263), 'PSNR': np.float64(29.66610354713485), 'RMSE': np.float32(8.3798275), 'SSIM': np.float64(0.8702117482804036), 'Dice': np.float64(0.9958736456699179)}, {'MAE': np.float32(3.0201676), 'PSNR': np.float64(33.95539056982801), 'RMSE': np.float32(5.114116), 'SSIM': np.float64(0.8966661637271298), 'Dice': np.float64(0.8442946221640626)}, {'MAE': np.float32(6.404663), 'PSNR': np.float64(25.753177194279694), 'RMSE': np.float32(13.148655), 'SSIM': np.float64(0.825539415397981), 'Dice': np.float64(0.9986344660632763)}, {'MAE': np.float32(27.1823), 'PSNR': np.float64(15.48725184358588), 'RMSE': np.float32(42.87238), 'SSIM': np.float64(0.4555149346068359), 'Dice': np.float64(0.9194640206053281)}, {'MAE': np.float32(8.146609), 'PSNR': np.float64(23.691515118303297), 'RMSE': np.float32(16.67111), 'SSIM': np.float64(0.7774544310821656), 'Dice': np.float64(0.9954571189302452)}, {'MAE': np.float32(31.759655), 'PSNR': np.float64(14.745830102465803), 'RMSE': np.float32(46.692673), 'SSIM': np.float64(0.49888809838114445), 'Dice': np.float64(0.9125259366784314)}, {'MAE': np.float32(6.1058683), 'PSNR': np.float64(25.117212652188655), 'RMSE': np.float32(14.147496), 'SSIM': np.float64(0.8042567343219833), 'Dice': np.float64(0.9985050598740131)}, {'MAE': np.float32(23.51654), 'PSNR': np.float64(16.39707455508901), 'RMSE': np.float32(38.608814), 'SSIM': np.float64(0.5131239178722248), 'Dice': np.float64(0.926480377683076)}, {'MAE': np.float32(44.177975), 'PSNR': np.float64(13.759178504456598), 'RMSE': np.float32(52.30959), 'SSIM': np.float64(0.631311143882043), 'Dice': np.float64(0.9998777473847125)}, {'MAE': np.float32(4.655241), 'PSNR': np.float64(27.228949881811708), 'RMSE': np.float32(11.094116), 'SSIM': np.float64(0.849350316880934), 'Dice': np.float64(0.9889196630108137)}, {'MAE': np.float32(3.9622898), 'PSNR': np.float64(32.60749439844709), 'RMSE': np.float32(5.9726295), 'SSIM': np.float64(0.8965770305309531), 'Dice': np.float64(0.9537488141277864)}, {'MAE': np.float32(5.5422196), 'PSNR': np.float64(25.99206035427192), 'RMSE': np.float32(12.791964), 'SSIM': np.float64(0.8306096393404674), 'Dice': np.float64(0.9993571945333818)}, {'MAE': np.float32(1.7767237), 'PSNR': np.float64(36.829689764433155), 'RMSE': np.float32(3.6732936), 'SSIM': np.float64(0.9275019968927473), 'Dice': np.float64(0.9242729369975723)}]}
    model_performance = performance
    # 计算每个模型的指标的平均值，最大值和最小值
    import numpy as np

    def calculate_stats(model_data):
        stats = {}
        for metric in ['MAE', 'PSNR', 'RMSE', 'SSIM', 'Dice']:
            values = [entry[metric] for entry in model_data]
            stats[metric] = {
                'Average': np.mean(values),
                'Max': np.max(values),
                'Min': np.min(values),
                'Std': np.std(values)  # 计算标准差
            }
        return stats


    # 输出每个模型的结果
    for model, data in model_performance.items():
        print(f"\nModel: {model}")
        stats = calculate_stats(data)
        for metric, stat in stats.items():
            print(f"    Average +- Std: {stat['Average']:.2f}+-{stat['Std']:.2f}")
