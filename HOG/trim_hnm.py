# Trimming hard negative patches

import csv
import pandas as pd
import random
import shutil
from config import hnm_patchnames, hnm_images, hnm_images_trimmed, hnm_list, train_difficult_patches_list

random.seed(6)

max_ = 5000


def read_csv(file, header=True):
    data = pd.read_csv(file, header=header).values.tolist()
    return data


hnm = read_csv(hnm_patchnames, header=None)
hnm_filenames = [f[0] for f in hnm]
random.shuffle(hnm_filenames)


diff_patches = []
for i in range(max_):
    file = hnm_filenames[i]
    print(file)
    shutil.copy(hnm_images + file, hnm_images_trimmed + file)
    diff_patches.append([file, 0])

with open(train_difficult_patches_list, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(diff_patches)
