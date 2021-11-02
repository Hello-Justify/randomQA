<!-- ABOUT THE PROJECT -->
# Random in multiprocessing
> When you use numpy.random with torch.utils.data.Dataset, something bad happens.


## Prerequisites

* python
* numpy
* torch


<!-- USAGE EXAMPLES -->
## Usage

```bash
python ori_loader.py

python cur_loader.py 
# equal to 
# python cur_loader.py random

python cur_loader.py numpy

# for train
python cur_loader.py numpy random

# for validation and test, shuffle should be False
python cur_loader.py numpy fix

python cur_loader.py torch
```
