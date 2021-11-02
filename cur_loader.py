import sys
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils import set_seed, get_img, random_modules

# default random module is random
if len(sys.argv) > 1:
    module_name = sys.argv[1].lower()
else:
    module_name = "random"

try:
    func_randint = random_modules[module_name]
except KeyError:
    print(f"KeyError: {module_name} not in [random, numpy, torch]")
    exit()

class DemoDataset(Dataset):
    def __init__(self, patch_size=128):
        super(DemoDataset, self).__init__()
        # read image list from txt file
        with open('data_list.txt') as f:
            self.img_ids = [line.rstrip('\n') for line in f.readlines()]

        self.ps = patch_size

    def __getitem__(self, index):
        
        # index = np.random.randint(len(self))

        # asumme the image shape: height = 1080, width = 1920
        H = 1080
        W = 1920

        # to get an even number, we use "//2*2"
        # choose module (random | numpy | torch)
        xx = func_randint(0, H - self.ps * 2 + 1) // 2 * 2
        yy = func_randint(0, W - self.ps * 2 + 1) // 2 * 2

        # log
        print(f"index: {index:1d}, xx: {xx:4d}, yy: {yy:4d}")

        input_full  = get_img(self.img_ids[index])
        input_patch = input_full[xx:xx+self.ps*2,yy:yy+self.ps*2]

        input_patch = input_patch[np.newaxis, :, :]
        return input_patch

    def __len__(self):
        return len(self.img_ids)


if __name__ == '__main__':
    set_seed(0)
    
    dataset = DemoDataset()
    loader = DataLoader(dataset, shuffle=True, batch_size=4, num_workers=2)

    for _ in loader:
        pass
