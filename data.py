import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
from random import shuffle, seed, choice, randint
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import model
from model import max_length

def get_paths(path, name="coco", use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open("dataset_coco.json","r"))
    >>> A=[]
    >>> for i in range(len(D["images"])):
    ...   if D["images"][i]["split"] == "val":
    ...     A+=D["images"][i]["sentids"][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if "coco" == name:
        imgdir = os.path.join(path, "images")
        capdir = os.path.join(path, "annotations")
        roots["train"] = {
            "img": os.path.join(imgdir, "train2014"),
            "cap": os.path.join(capdir, "captions_train2014.json")
        }
        roots["val"] = {
            "img": os.path.join(imgdir, "val2014"),
            "cap": os.path.join(capdir, "captions_val2014.json")
        }
        roots["test"] = {
            "img": os.path.join(imgdir, "val2014"),
            "cap": os.path.join(capdir, "captions_val2014.json")
        }
        roots["trainrestval"] = {
            "img": (roots["train"]["img"], roots["val"]["img"]),
            "cap": (roots["train"]["cap"], roots["val"]["cap"])
        }
        ids["train"] = np.load(os.path.join(capdir, "coco_train_ids.npy"))
        ids["val"] = np.load(os.path.join(capdir, "coco_dev_ids.npy"))[:5000]
        ids["test"] = np.load(os.path.join(capdir, "coco_test_ids.npy"))
        ids["trainrestval"] = (
            ids["train"],
            np.load(os.path.join(capdir, "coco_restval_ids.npy")))
        if use_restval:
            roots["train"] = roots["trainrestval"]
            ids["train"] = ids["trainrestval"]
    elif "f8k" == name:
        imgdir = os.path.join(path, "images")
        cap = os.path.join(path, "dataset_flickr8k.json")
        roots["train"] = {"img": imgdir, "cap": cap}
        roots["val"] = {"img": imgdir, "cap": cap}
        roots["test"] = {"img": imgdir, "cap": cap}
        ids = {"train": None, "val": None, "test": None}
    elif "f30k" == name:
        imgdir = os.path.join(path, "images")
        cap = os.path.join(path, "dataset_flickr30k.json")
        roots["train"] = {"img": imgdir, "cap": cap}
        roots["val"] = {"img": imgdir, "cap": cap}
        roots["test"] = {"img": imgdir, "cap": cap}
        ids = {"train": None, "val": None, "test": None}
    elif "CUHK-PEDES" == name:
        imgdir = os.path.join(path, "imgs")
        cap = os.path.join(path, "reid_raw.json")
        roots["train"] = {"img": imgdir, "cap": cap}
        roots["val"] = {"img": imgdir, "cap": cap}
        roots["test"] = {"img": imgdir, "cap": cap}
        ids = {"train": None, "val": None, "test": None}

    return roots, ids

# CUHK DATASET

class CUHKDataset(data.Dataset):
    """CUHKDataset test on the person retrieval task."""
    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        imgs = jsonmod.load(open(json, "r"))
        self.imgs = [x for x in imgs if x["split"] == split ]
        self.ids = []
        for i, d in enumerate(self.imgs):
            self.ids += [(i, x) for x in range(len(d["captions"]))]

    def __getitem__(self, index):
        vocab = self.vocab
        ann_id = self.ids[index]
        img_id = ann_id[0]
        img = self.imgs[img_id]
        image = Image.open(os.path.join(self.root, img["file_path"]))
        if self.transform is not None:
            image= self.transform(image)
        tokens = nltk.tokenize.word_tokenize(
                 str(img["captions"][ann_id[1]]).lower()) # deop decode
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        # print len(tokens), len(caption)
        # print caption
        target = torch.Tensor(caption)

        return image, target, index, img_id 

    def __len__(self):
        return len(self.ids)

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root, caption, img_id, path, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower()) # drop decode
        #tokens = tokens[:max_length]
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]["caption"]
        img_id = coco.anns[ann_id]["image_id"]
        path = coco.loadImgs(img_id)[0]["file_name"]
        image = Image.open(os.path.join(root, path)).convert("RGB")

        return root, caption, img_id, path, image

    def __len__(self):
        return len(self.ids)


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    Formats:
        "images":[{
            "sentids": [0, 1, 2, 3, 4],
            "imgid": 0,
            "sentences": [{
                "tokens": ["a", "sample", "example"],
                "raw": "A sample example.",
                "imgid": 0,
                "sentid": 0
            }, ... ]
            "split": "train/val/test",
            "filename:" "xxx.jpg",
        }, ... ]

    """
    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, "r"))["images"]
        self.ids = []
        self.img_num = 0
        for i, d in enumerate(self.dataset):
            if d["split"] == split:
                self.ids += [(i, x, self.img_num) for x in range(len(d["sentences"]))]
                self.img_num += 1

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        img_cls = ann_id[2]
        caption = self.dataset[img_id]["sentences"][ann_id[1]]["raw"]
        path = self.dataset[img_id]["filename"]

        image = Image.open(os.path.join(root, path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower()) # drop decode
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        target = torch.Tensor(caption)
        return image, target, index, img_id, img_cls

    def __len__(self):
        return len(self.ids)


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + "/"

        # Captions
        self.captions = []
        with open(loc+"%s_caps.txt" % data_split, "r") as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+"%s_ims.npy" % data_split)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn"t
        if self.images.shape[0] != len(self.captions):
            self.im_div = 5
        else:
            self.im_div = 1

        self.length = len(self.captions) // self.im_div
        # the development set for coco is large and so validation would be slow
        if data_split == "dev":
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index
        caption_id = index * self.im_div + randint(0, self.im_div-1)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[caption_id]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower()) # drop decode
        tokens = tokens[:max_length]
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        target = torch.Tensor(caption)
        return image, target, caption_id, img_id, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, img_cls = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    # targets = torch.zeros(len(captions), max(lengths)).long()
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = torch.Tensor(lengths)
    img_cls = torch.Tensor(img_cls).long()
    return images, targets, lengths, ids, img_cls


def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn,
                      distributed=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if "coco" in data_name:
        # COCO custom dataset
        dataset = CocoDataset(root=root,
                              json=json,
                              vocab=vocab,
                              transform=transform, ids=ids)
    elif "f8k" in data_name or "f30k" in data_name:
        dataset = FlickrDataset(root=root,
                                split=split,
                                json=json,
                                vocab=vocab,
                                transform=transform)
    elif "CUHK" in data_name:
        dataset = CUHKDataset(root=root,
                              split=split,
                              json=json,
                              vocab=vocab,
                              transform=transform)

    # Data loader
    if distributed:
        data_sampler = DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=False,
                                                  num_workers=num_workers,
                                                  sampler=data_sampler,
                                                  collate_fn=collate_fn)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  num_workers=num_workers,
                                                  collate_fn=collate_fn)
    
    return data_loader


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == "train":
        t_list = [transforms.RandomSizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == "val":
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        #t_list = [transforms.Resize((224, 224))]
    elif split_name == "test":
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        #t_list = [transforms.Resize((224, 224))]

    """if "CUHK" in data_name:
        t_end = [transforms.ToTensor()]
    else:"""
    t_end = [transforms.ToTensor(), normalizer]
    
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith("_precomp"):
        train_loader = get_precomp_loader(dpath, "train", vocab, opt,
                                          batch_size, True, workers)
        val_loader = get_precomp_loader(dpath, "dev", vocab, opt,
                                        batch_size, False, workers)
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, "train", opt)
        train_loader = get_loader_single(opt.data_name, "train", # !!!
                                         roots["train"]["img"],
                                         roots["train"]["cap"],
                                         vocab, transform, ids=ids["train"],
                                         batch_size=batch_size, shuffle=True,
                                         num_workers=workers,
                                         collate_fn=collate_fn,
                                         distributed=opt.distributed)

        transform = get_transform(data_name, "val", opt)
        val_loader = get_loader_single(opt.data_name, "test", # !!!
                                       roots["val"]["img"],
                                       roots["val"]["cap"],
                                       vocab, transform, ids=ids["val"],
                                       batch_size=16, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn,
                                       distributed=False)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if opt.data_name.endswith("_precomp"):
        test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                         batch_size, False, workers)
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, split_name, opt)
        test_loader = get_loader_single(opt.data_name, split_name,
                                        roots[split_name]["img"],
                                        roots[split_name]["cap"],
                                        vocab, transform, ids=ids[split_name],
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=workers,
                                        collate_fn=collate_fn,
                                        distributed=False)

    return test_loader
