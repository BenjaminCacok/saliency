import os
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import matplotlib.pyplot as plt

sauc_list = []
valued = 0

pred_dir = "./test/output/"
gt_dir = "C:/Users/18336/Documents/SJTUTIS/salicon/maps/train"

pred_files = sorted(os.listdir(pred_dir))
gt_files = sorted(os.listdir(gt_dir))

for pred_file, gt_file in zip(pred_files, gt_files):
    pred_map = np.array(Image.open(os.path.join(pred_dir, pred_file)))
    gt_map = np.array(Image.open(os.path.join(gt_dir, gt_file)))

    pred_map = pred_map.flatten()
    gt_map = gt_map.flatten()
    gt_map = gt_map / 255.0
    gt_map = (gt_map > 0.5).astype(int)
    assert set(np.unique(gt_map)) <= {0, 1}

    salient_indices = np.where(gt_map == 1)[0]
    nonsalient_indices = np.where(gt_map == 0)[0]
    random_nonsalient_indices = random.sample(list(nonsalient_indices), len(salient_indices))

    new_gt_map = np.concatenate((gt_map[salient_indices], gt_map[random_nonsalient_indices]))
    new_pred_map = np.concatenate((pred_map[salient_indices], pred_map[random_nonsalient_indices]))

    sauc_value = roc_auc_score(new_gt_map, new_pred_map)
    sauc_list.append(sauc_value)
    valued = valued + 1
    if valued % 100 == 0:
        print("{} pictures have been valued".format(valued))

sAUC_average = np.mean(sauc_list)
sAUC_average = round(sAUC_average, 4)

plt.figure(figsize=(20, 10), dpi=100)
x_range = list(range(1, 10001))
plt.plot(x_range, sauc_list, c='red')
plt.scatter(x_range, sauc_list, c='red')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("picture number", fontdict={'size': 16})
plt.ylabel("sAUC", fontdict={'size': 16})
plt.title("The average sAUC is {}".format(sAUC_average), fontdict={'size': 20})
plt.show()
