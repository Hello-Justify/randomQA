import torch
import numpy as np


def get_img(name):
    # return the image with shape(height, width) like cv2.imread(name, 0)
    return np.zeros((1080, 1920), dtype='uint8')

# read image list from txt file
with open('data_list.txt') as f:
    img_ids = [line.rstrip('\n') for line in f.readlines()]

train_data_length = len(img_ids)

batch_size = 4
patch_size = 128

for batch_id in range(int(train_data_length/batch_size)):

    input_batch_list = []

    for _ in range(batch_size):

        index = np.random.randint(train_data_length)

        # asumme the image shape: height = 1080, width = 1920
        H = 1080
        W = 1920

        # to get an even number, we use "//2*2"
        xx = np.random.randint(0, H - patch_size * 2 + 1) // 2 * 2
        yy = np.random.randint(0, W - patch_size * 2 + 1) // 2 * 2

        input_full  = get_img(img_ids[index])

        input_patch = input_full[xx:xx+patch_size*2,yy:yy+patch_size*2]

        input_patch = input_patch[np.newaxis, np.newaxis, :, :]

        input_batch_list.append(input_patch)

    input_batch = np.concatenate(input_batch_list, axis=0)

    in_data = torch.from_numpy(input_batch)
