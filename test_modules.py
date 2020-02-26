print("work start!")
import torch
print(torch.__version__)
#import tensorboard_logger as tb_logger
print("import logger OK!")
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import yaml
from easydict import EasyDict

print("import all torch OK!")

#from transformer.Models import Encoder as self_attention_encoder
#from transformer.Layers import EncoderLayer as attention_layer
#print("import transformer OK!")

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import logging
import torch.backends.cudnn as cudnn
import pickle
import os
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data


import data
from model import ImageSelfAttention
import model
from vocab import Vocabulary
import argparse
from fusion_module import *


def test_img_self_att():
    fake_feature = Variable(torch.randn(16, 32*7*7))
    fake_feature = fake_feature.view(16, -1, 7, 7)
    img_self_attention = ImageSelfAttention(32)
    out = img_self_attention(fake_feature)
    print(out.size())

def test_f30k_dataloader():
    data_name = "f30k"
    data_path = "./data/f30k"
    vocab_path = "./vocab/"

    vocab = pickle.load(open(os.path.join(vocab_path,
            '%s_vocab.pkl' % data_name), 'rb'))
    roots, ids = data.get_paths(data_path, data_name, False)
    transform = transforms.Compose([transforms.RandomSizedCrop(224),
                                    transforms.ToTensor()])
    print (roots, ids)
    train_loader = data.get_loader_single(data_name, "train", # !!!
                                     roots["train"]["img"],
                                     roots["train"]["cap"],
                                     vocab, transform, ids=ids["train"],
                                     batch_size=16, shuffle=False,
                                     num_workers=1,
                                     collate_fn=data.collate_fn,
                                     distributed=False)
    print ("f30k dataloader output:", train_loader.dataset.img_num)
    #for (id, x) in enumerate(train_loader):
        #if id > 0 : break
        #print (id, x)

def test_text_encoder():

    data_name = "f30k_precomp"
    data_path = "./data/"
    vocab_path = "./vocab/"

    vocab = pickle.load(open(os.path.join(vocab_path,
            '%s_vocab.pkl' % data_name), 'rb'))
    vocab_size = len(vocab)

    print(vocab_size)

    word_dim = 10
    embed_size = 20
    num_layers = 1

    txt_enc = model.EncoderText(vocab_size, word_dim, embed_size, num_layers, 
                                bi_gru=True, self_attention=True)
    txt_enc = nn.DataParallel(txt_enc)
    txt_enc.cuda()
    
    fake_text = Variable(torch.ones(16, 50).long())
    fake_lengths = Variable(torch.Tensor([16-i for i in range(16)]).long())

    out = txt_enc(fake_text, fake_lengths)
    print ("txt_enc output:", out.size())

def test_img_encoder():
    embed_size = 20
    img_enc = model.EncoderImage("f30k_precomp", 20, 20, False, self_attention=True)
    img_enc = nn.DataParallel(img_enc)
    img_enc.cuda()

    fake_img = Variable(torch.ones(16, 3, 20))
    out = img_enc(fake_img)
    print ("img_enc output:", out.size())

def test_stack_fusion():
    fusion_module = CrossAttention(32, 2, -1)
    print("CrossAttention init success!")
    fake_img = Variable(torch.randn(16, 49, 32))
    fake_txt = Variable(torch.randn(8, 14, 32))
    score = fusion_module(fake_img, fake_txt, get_score=True)
    print(score.size())

    print("----CrossAttention module success!----")


def test_stack_fusion_new():
    fusion_module = CrossAttentionNew(32, 2, -1)
    print("CrossAttention init success!")
    fake_img = Variable(torch.randn(16, 49, 32))
    fake_txt = Variable(torch.randn(8, 14, 32))
    score = fusion_module(fake_img, fake_txt, get_score=True)
    print(score.size())

    print("----CrossAttention module success!----")


def test_gate_fusion():
    fusion_module = GatedFusion(32, 2, 0.0)
    print("FusionModule init success!")
    fake_img = Variable(torch.randn(16, 49, 32))
    fake_txt = Variable(torch.randn(8, 14, 32))
    score = fusion_module(fake_img, fake_txt, get_score=True)
    print(score.size())

    print("----GatedFusion module success!----")


def test_gate_fusion_new():
    fusion_module = GatedFusionNew(32, 2, 0.0)
    print("FusionModule init success!")
    fake_img = Variable(torch.randn(16, 49, 32))
    fake_txt = Variable(torch.randn(8, 14, 32))
    score = fusion_module(fake_img, fake_txt, get_score=True)
    print(score.size())

    print("----GatedFusion module success!----")

def test_CAMP_model(config_path):
    print("OK!")
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    #config_path = "./experiments/f30k_cross_attention/config_test.yaml"
    with open(config_path) as f:
        opt = yaml.load(f)
    opt = EasyDict(opt['common'])


    vocab = pickle.load(open(os.path.join(opt.vocab_path,
            '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)

    train_logger = LogCollector()

    print("----Start init model----")
    CAMP = model.CAMP(opt)
    CAMP.logger = train_logger

    if opt.resume is not None:
       ckp = torch.load(opt.resume)
       CAMP.load_state_dict(ckp["model"])

    CAMP.train_start()
    print("----Model init success----")

    """
    fake_img = torch.randn(16, 36, opt.img_dim)
    fake_text = torch.ones(16, 32).long()
    fake_lengths = torch.Tensor([32] * 16)
    fake_pos = torch.ones(16, 32).long()
    fake_ids = torch.ones(16).long()

    CAMP.train_emb(fake_img, fake_text, fake_lengths,
                   instance_ids=fake_ids)
    print("----Test train_emb success----")
    """
    
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.crop_size, 128, 4, opt)

    test_loader = data.get_test_loader("test", opt.data_name, vocab, opt.crop_size, 128, 4, opt)

    CAMP.val_start()
    img_embs, cap_embs, cap_masks = encode_data(
        CAMP, test_loader, opt.log_step, logging.info)


    (r1, r5, r10, medr, meanr), (r1i, r5i, r10i, medri, meanri), score_matrix= i2t(img_embs, cap_embs, cap_masks, measure=opt.measure,
                                     model=CAMP, return_ranks=True)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    
def main():
    #test_f30k_dataloader()
    #test_text_encoder()
    #test_img_encoder()
    #test_stack_fusion()
    #test_gate_fusion()
    #test_stack_fusion_new()
    #test_gate_fusion_new()
    test_CAMP_model("./experiments/f30k_cross_attention/config_test.yaml")

if __name__ == '__main__':
    main()
