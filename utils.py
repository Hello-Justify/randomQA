import torch
import random
import numpy as np


random_modules = {
    "random": lambda x,y: random.randint(a=x, b=y-1),
    "numpy" : lambda x,y: np.random.randint(low=x, high=y),
    "torch" : lambda x,y: torch.randint(low=x, high=y, size=(1,)).item()
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_img(name):
    # return the image with shape(height, width) like cv2.imread(name, 0)
    return np.zeros((1080, 1920), dtype='uint8')


def numpy_random_init(worker_id):
    process_seed = torch.initial_seed()
    base_seed    = process_seed - worker_id
    ss  = np.random.SeedSequence([worker_id, base_seed])
    np.random.seed(ss.generate_state(4))