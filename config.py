# Extend
folder_path = r'/media/userdisk0/zyliu/Data/normalizeDataSpacing4'  # data
json_path = 'dataset_split_grouped.json'
BEST_WEIGHTS_FILE = 'best_weights.json'
weightsStoreFile = 'weightsTotal'
gen_basedim = 64
adv_basedim = 32
epochs = 1000
start_epochs = 0  # 若中途断开，可从 start_epochs 开始训练
batch_size=32
# device_ids = [0, 1, 2, 3]
device_ids = [4, 5, 6, 7]
prefetch_factor=32
num_workers=32
