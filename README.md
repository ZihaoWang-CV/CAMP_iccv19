# Introduction

This repository is for [CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval](https://arxiv.org/abs/1909.05506) from CUHK-SenseTime Joint Lab (appear in ICCV 2019).

It is built on top of the [VSE++](https://github.com/fartashf/vsepp) and [SCAN](https://kuanghuei.github.io/SCANProject/) in PyTorch.


## Requirements and Installation
We recommended the following dependencies.

* Python 3
* [PyTorch](http://pytorch.org/) (>1.0)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [torchvision]()
* [matplotlib]()

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data

Download the dataset files and pre-trained models. We use the same pre-extracted features and splits as [SCAN](https://kuanghuei.github.io/SCANProject/).

The splits are produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

The precomputed image features of MS-COCO are from [here](https://github.com/peteanderson80/bottom-up-attention). The precomputed image features of Flickr30K are extracted from the raw Flickr30K images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). 

The pre-extracted image features are from [SCAN](https://kuanghuei.github.io/SCANProject/), produced by [Kuang-Huei Lee](https://kuanghuei.github.io/). The data can be downloaded from:

```bash
wget https://scanproject.blob.core.windows.net/scan-data/data.zip
```

We refer to the path of extracted files for `data.zip` as `./data` directory.


## Training new models
Run `train.py` in the directory of the corresponding config path:

Training the cross-attention model on Flickr30K dataset:
```bash
cd ./experiments/f30k_cross_attention
python python -u ../../train.py --config ./config_256.yaml
```

Training the full CAMP model on Flickr30K dataset:
```bash
cd ./experiments/f30k_gate_fusion
python python -u ../../train.py --config ./config_finetune.yaml
```
**We initialize the network weights from the pretrained cross-attention model to train the full CAMP model. The weights for attention map are fixed for the first several epochs and then we finetune the whole network.**


## Evaluate trained models
Changing the `resume` arguments in the coreesponding config file and running evaluation in the project root directory:

```python
from test_modules import test_CAMP_model

#config_path = "./experiments/f30k_cross_attention/config_test.yaml"
test_CAMP_model(config_path)
```
Pretrained model for Flickr30K could be downloaded [here](https://drive.google.com/drive/folders/1o8rUv78uS_aX4P1hMPELl53cxnZ8UqiF?usp=sharing).

## Reference

If you found this code useful, please cite the following paper:

```
@InProceedings{Wang_2019_ICCV,
author = {Wang, Zihao and Liu, Xihui and Li, Hongsheng and Sheng, Lu and Yan, Junjie and Wang, Xiaogang and Shao, Jing},
title = {CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

*I have left CUHK and the email address is deprecated. Please directly open a new issue or contact zihaowang.cv@gmail.com if you have further quetions. Thanks!*
