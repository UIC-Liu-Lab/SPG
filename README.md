# Soft-masking of Parameter-level Gradient flow (SPG)

This repository provides the code for our proposed SPG in our [ICML 2023 paper](https://arxiv.org/abs/2306.14775).

## Datasets Preparation

We use 9 datasets in the paper. To reproduce the results, some of these datasets need to be prepared manually.

### CIFAR100-based (**C-10** and **C-20**)

You do not need do anything for these datasets as they will be automatically downloaded.

### TinyImageNet-based (**T-10** and **T-20**)

You can download the datasets from [the official site](https://image-net.org/).

1. Download the Tiny ImageNet file.
2. Extract the file, and place them as follows.

<pre>
data/tiny-imagenet-200/
|- train/
|  |- n01443537/
|  |- n01629819/
|  +- ...
+- val/
   |- val_annotations.txt
   +- images/
      |- val_0.JPEG
      |- val_1.JPEG
      +- ...
</pre>

3. Run `prep_tinyimagenet.py` to reorganise files so that `torchvision.datasets.ImageFolder` can read them.
4. Make sure you see the structure as follows.

<pre>
data/tiny-imagenet-200/
|- test/
|  |- n01443537/
|  |- n01629819/
|  +- ...
|- train/
|  |- n01443537/
|  |- n01629819/
|  +- ...
+- val/
   +- # These files are not used any more. 
</pre>

### ImageNet-based (**I-100**)

You can download the datasets from [the official site](https://image-net.org/).

1. Download the downsampled image data (32x32).
2. Extract the files, and place the extracted files under `./data/imagenet/`.
3. Make sure you see the structure as follows.

<pre>
data/imagenet/
|- test/
|  +- val_data
+- train/
   |- train_data_batch_1
   |- ...
   +- train_data_batch_10
</pre>

### Federated CelebA-based (**FC-10** and **FC-20**)

1. Follow [the instruction](https://github.com/TalwalkarLab/leaf/tree/master/data/celeba) to create data.
2. Place the raw images under `data/fceleba/raw/img_align_celeba/`.
3. Make sure you see the structure as follows.

<pre>
data/fceleba/
|- iid/
|  |- test/
|  |  +- all_data_iid_01_0_keep_5_test_9.json
|  +- train/ 
|     +- all_data_iid_01_0_keep_5_train_9.json
+- raw/
   +- img_align_celeba/
      |- 000001.jpg
      |- 000002.jpg
      +- ...
</pre>

### Federated EMNIST-based (**FE-10** and **FE-20**)

1. Follow [the instruction](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist) to create data.
2. Place the raw images under `data/femnist/raw/train/` and `data/femnist/raw/test/`.
3. Make sure you see the structure as follows.

<pre>
data/femnist/
+- raw/
   |- test
   |  |- all_data_0_iid_01_0_keep_0_test_9.json
   |  |- ...
   |  +- all_data_34_iid_01_0_keep_0_test_9.json
   +- train
      |- all_data_0_iid_01_0_keep_0_train_9.json
      |- ...
      +- all_data_34_iid_01_0_keep_0_train_9.json
</pre>

## Experiments

Experiments can be reproduced by running

```shell
python3 main.py appr=spg seq=<seq> 
```

with specifying `<seq>` for a dataset you want to run.

For `<seq>`, you can choose one from the following datasets.

- `cifar100_10` for **C-10** (CIFAR100 with 10 tasks)
- `cifar100_20` for **C-20**
- `tinyimagenet_10` for **T-10** (TinyImageNet with 10 tasks)
- `tinyimagenet_20` for **T-20**
- `imagenet_100` for **I-100** (ImageNet with 100 tasks)
- `fceleba_10` for **FC-10** (Federated CelebA with 10 tasks)
- `fceleba_20` for **FC-20**
- `femnist_10` for **FE-10** (Federated EMNIST with 10 tasks)
- `femnist_20` for **FE-20**

As CIFAR100-based datasets will be automatically downloaded by PyTorch, you can test **C-10** or **C-20** right now by running

```shell
python3 main.py appr=spg seq=cifar100_10 # or seq=cifar100_20
```

## Reference

We would appreciate if if you could refer to our paper as one of your baselines!

```
@inproceedings{konishi2023spg,
  title={{Parameter-Level Soft-Masking for Continual Learning}},
  author={Konishi, Tatsuya and Kurokawa, Mori and Ono, Chihiro and Ke, Zixuan and Kim, Gyuhak and Liu, Bing},
  booktitle={Proc. of ICML},
  year={2023},
}
```