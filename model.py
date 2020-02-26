import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict

#from transformer.Models import Encoder as self_attention_encoder
#from transformer.Layers import EncoderLayer as attention_layer
from resnet import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import logging
import torch.backends.cudnn as cudnn
import pickle
from fusion_module import *

max_length = 47

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='resnet152', no_imgnorm=False,
                 self_attention=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(img_dim, embed_size, no_imgnorm,
                                      self_attention)
    else:
        img_enc = EncoderImageFull(embed_size, finetune, cnn_type, no_imgnorm,
                                   self_attention, fusion)

    return img_enc



class ImageSelfAttention(nn.Module):
    """ Self-attention module for CNN's feature map.
    Inspired by: Zhang et al., 2018 The self-attention mechanism in SAGAN.
    """
    def __init__(self, planes):
        super(ImageSelfAttention, self).__init__()
        inner = planes // 8
        self.conv_f = nn.Conv1d(planes, inner, kernel_size=1, bias=False)
        self.conv_g = nn.Conv1d(planes, inner, kernel_size=1, bias=False)
        self.conv_h = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        f = self.conv_f(x)
        g = self.conv_g(x)
        h = self.conv_h(x)
        sim_beta = torch.matmul(f.transpose(1, 2), g)
        beta = nn.functional.softmax(sim_beta, dim=1)
        o = torch.matmul(h, beta)
        return o

# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='resnet152',
                 no_imgnorm=False, self_attention=False, fusion=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.self_attention = self_attention
        self.fusion = fusion
        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True, fusion)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with new structures
        if self_attention:
            self.cnn.avgpool = nn.Sequential()
            self.attention_layer = ImageSelfAttention(2048)
            self.AvgPool2d = nn.AvgPool2d(7, stride=1)
        
        if fusion:
            self.cnn.avgpool = nn.Sequential()
            self.fc = nn.Linear(2048, embed_size)
    
        else:
            self.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        
        self.cnn.fc = nn.Sequential()
        self.init_weights()

    def get_cnn(self, arch, pretrained, fusion):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if arch == "resnet152":
            if pretrained:
                print("=> using pre-trained model '{}'".format(arch))
                model = resnet152(pretrained=True, fusion=fusion)
            else:
                print("=> creating model '{}'".format(arch))
                model = resnet152(pretrained=False, fusion=fusion)
        
        else:
            if pretrained:
                print("=> using pre-trained model '{}'".format(arch))
                model = models.__dict__[arch](pretrained=True)
            else:
                print("=> creating model '{}'".format(arch))
                model = models.__dict__[arch]()
    
        return model

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        if self.self_attention:
            features = features.view(images.size(0), -1, 7, 7)
            features = self.attention_layer(features)
            features = features.view(images.size(0), -1, 7, 7)
            features = self.AvgPool2d(features)

        # linear projection to the joint embedding space
        if self.fusion:
            features = features.view(features.size(0), features.size(1), -1)
            features = features.transpose(1, 2)
        else:    
            features = features.view(features.size(0), -1)
        
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            if self.fusion:
                features = l2norm(features, dim=2)
            else:
                features = l2norm(features, dim=1)

        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False,
                 self_attention=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.self_attention = self_attention

        self.fc = nn.Linear(img_dim, embed_size)
        if self_attention:
            self.attention_layer = SummaryAttn(embed_size, 1, -1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        if self.self_attention:
            features = self.attention_layer(features, features)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 bi_gru=False, no_txtnorm=False,
                 self_attention=False, embed_weights=''):
        super(EncoderText, self).__init__()
        self.no_txtnorm = no_txtnorm
        self.embed_size = embed_size
        self.self_attention = self_attention
        self.bi_gru = bi_gru

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=bi_gru)


        if self_attention:
            self.attention_layer = SummaryAttn(embed_size, 1, -1)

        self._init_weights(embed_weights)

    def _init_weights(self, embed_weights=''):
        if embed_weights:
            w = np.load(embed_weights)
            w = torch.from_numpy(w)
            self.embed.load_state_dict({'weight': w})
            print("Load Word Embedding Weights Successfully.")
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)

        # Mask the attention weights of emtpy token
        l_list = [int(i) for i in lengths.data]
        mask = Variable(torch.ByteTensor([i*[1] + (max_length+3-i)*[0] for i in l_list])).cuda()


        # Forward propagate RNN
        packed = pack_padded_sequence(x, l_list, batch_first=True)
        self.rnn.flatten_parameters()
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        
        if self.bi_gru:
            out, cap_len = padded
            out = (out[:,:,:out.size(2)//2] + out[:,:,out.size(2)//2:])/2
            I = Variable(torch.zeros(out.size(0),
                max_length+3-out.size(1), out.size(2))).cuda()
            if not len(I.size()) < 3:
                out = torch.cat((out, I), dim=1)

        else:
            I = torch.LongTensor(l_list).view(-1, 1, 1)
            I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
            out = torch.gather(padded[0], 1, I).squeeze(1)

        if self.self_attention:
            out = self.attention_layer(out, out, mask=mask)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            out = l2norm(out, dim=-1)

        return out



def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def attention_sim(im, s):
    im_expanded = im.unsqueeze(1).expand(
                  im.size(0),s.size(0),s.size(1))
    no_attention_score = im_expanded * s
    im_to_s_attention = nn.functional.softmax(no_attention_score, dim=2)
    score = (im_to_s_attention*no_attention_score).sum(dim=2)
    return score

class InstanceLoss(nn.Module):
    """
    Compute instance loss
    """

    def __init__(self):
        super(InstanceLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img_cls, txt_cls, labels):
        cost_im = self.loss(img_cls, labels)
        cost_s = self.loss(txt_cls, labels)
        return cost_im + cost_s


class SimLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, inner_dim=0, loss_func="BCE"):
        super(SimLoss, self).__init__()
        self.margin = margin
        self.measure = measure
        if measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'attention':
            self.sim = attention_sim
        elif measure == 'cross_attention':
            self.sim = CrossAttention(inner_dim, 4, -1)
        elif measure == 'cross_attention_new':
            self.sim = CrossAttentionNew(inner_dim, 4, -1)
        elif measure == 'gate_fusion':
            self.sim = GatedFusion(inner_dim, 4, 0.0)
        elif measure == 'gate_fusion_new':
            self.sim = GatedFusionNew(inner_dim, 4, 0.0)
        else:
            self.sim = cosine_sim

        self.loss_func = loss_func
        self.max_violation = max_violation

    def forward(self, im, s, get_score=False, keep="words", mask=None):
        # compute image-sentence score matrix
        if self.measure == 'cosine':
            cur_im = im
            cur_s = s
            drive_num = torch.cuda.device_count()

            if keep == "words":
                cur_s = s.unsqueeze(0).expand(drive_num, -1, -1, -1)
            elif keep == "regions":
                cur_im = im.unsqueeze(0).expand(drive_num, -1, -1, -1)

            scores = self.sim(cur_im, cur_s, keep=keep, ret_dot=True)
            
            if keep == "regions":
                scores = scores.transpose(0, 1)
        elif self.measure == 'cross_attention' or self.measure == 'cross_attention_new':
            cur_im = im
            cur_s = s
            cur_mask = mask
            drive_num = torch.cuda.device_count()

            if keep == "words":
                cur_s = s.unsqueeze(0).expand(drive_num, -1, -1, -1)
                cur_mask = mask.unsqueeze(0).expand(drive_num, -1, -1)
            elif keep == "regions":
                cur_im = im.unsqueeze(0).expand(drive_num, -1, -1, -1)

            scores = self.sim(cur_im, cur_s, keep=keep, mask=cur_mask)
            
            if keep == "regions":
                scores = scores.transpose(0, 1)
        elif self.measure == 'gate_fusion' or self.measure == 'gate_fusion_new':
            cur_im = im
            cur_s = s
            cur_mask = mask
            drive_num = torch.cuda.device_count()

            if keep == "words":
                cur_s = s.unsqueeze(0).expand(min(im.size(0), drive_num), -1, -1, -1)
                cur_mask = mask.unsqueeze(0).expand(min(im.size(0), drive_num), -1, -1)
            elif keep == "regions":
                cur_im = im.unsqueeze(0).expand(drive_num, -1, -1, -1)

            scores = self.sim(cur_im, cur_s, keep=keep, mask=cur_mask)

            if keep == "regions":
                scores = scores.transpose(0, 1)
        else:
            scores = self.sim(im, s)

        if get_score:
            return scores


        if self.loss_func == 'BCE':
            eps = 0.000001

            scores = scores.clamp(min=eps, max=(1.0-eps))
            de_scores = 1.0 - scores

            label = Variable(torch.eye(scores.size(0))).cuda()
            de_label = 1 - label
        
            scores = torch.log(scores) * label
            de_scores = torch.log(de_scores) * de_label

            if self.max_violation:
                le = -(scores.sum() + scores.sum() + de_scores.min(1)[0].sum() + de_scores.min(0)[0].sum())
            else:
                le = -(scores.diag().mean() + de_scores.mean())

            return le
        else:
            
            diagonal = scores.diag().view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)
            # compare every diagonal score to scores in its column
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            # compare every diagonal score to scores in its row
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            # clear diagonals
            mask = torch.eye(scores.size(0)) > .5
            I = Variable(mask)
            if torch.cuda.is_available():
                I = I.cuda()
            cost_s = cost_s.masked_fill_(I, 0)
            cost_im = cost_im.masked_fill_(I, 0)

            # keep the maximum violating negative for each query
            if self.max_violation:
                cost_s = cost_s.max(1)[0]
                cost_im = cost_im.max(0)[0]

            return cost_s.sum() + cost_im.sum()


class CAMP(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    no_imgnorm=opt.no_imgnorm,
                                    self_attention=opt.self_attention)

        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   no_txtnorm=opt.no_txtnorm, 
                                   self_attention=opt.self_attention,
                                   embed_weights=opt.word_embed, 
                                   bi_gru=opt.bi_gru)

        # Loss and Optimizer
        if opt.cross_model:
            self.criterion = SimLoss(margin=opt.margin,
                                             measure=opt.measure,
                                             max_violation=opt.max_violation,
                                             inner_dim=opt.embed_size)
        else:
            self.criterion = SimLoss(margin=opt.margin,
                                             measure=opt.measure,
                                             max_violation=opt.max_violation)

        if torch.cuda.is_available():
            self.img_enc = nn.DataParallel(self.img_enc)
            self.txt_enc = nn.DataParallel(self.txt_enc)
            self.img_enc.cuda()
            self.txt_enc.cuda()
            if opt.cross_model:
                self.criterion.sim = nn.DataParallel(self.criterion.sim)
                self.criterion.sim.cuda()
            cudnn.benchmark = True

        print("Encoders init OK!")
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.module.fc.parameters())
        if opt.self_attention:
            params += list(self.img_enc.module.attention_layer.parameters())

        if opt.finetune:
            params += list(self.img_enc.module.cnn.parameters())

        if opt.cross_model:
            params += list(self.criterion.sim.parameters())
        
        if opt.measure == "gate_fusion" and not opt.finetune_gate:
            print("Only fc layers and final aggregation layers optimized.")
            params = list(self.criterion.sim.module.fc_1.parameters())
            params += list(self.criterion.sim.module.fc_2.parameters())
            params += list(self.criterion.sim.module.fc_out.parameters())
            params += list(self.criterion.sim.module.reduce_layer_1.parameters())
            params += list(self.criterion.sim.module.reduce_layer_2.parameters())

        if opt.measure == "gate_fusion_new" and not opt.finetune_gate:
            print("Only fc layers and final aggregation layers optimized.")
            params = list(self.criterion.sim.module.fc_1.parameters())
            params += list(self.criterion.sim.module.fc_2.parameters())
            #params += list(self.criterion.sim.module.fc_gate_1.parameters())
            #params += list(self.criterion.sim.module.fc_gate_2.parameters())
            params += list(self.criterion.sim.module.fc_out.parameters())
            params += list(self.criterion.sim.module.final_reduce_1.parameters())
            params += list(self.criterion.sim.module.final_reduce_2.parameters())

        if opt.embed_mask:
            self.embed_mask = np.load(opt.embed_mask)
        else:
            self.embed_mask = None
        
        
        self.params = params

        if opt.optimizer.type == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer.type == "SGD":
            self.optimizer = torch.optim.SGD(params, lr=opt.learning_rate,
                                             momentum=opt.optimizer.momentum,
                                             weight_decay=opt.optimizer.weight_decay,
                                             nesterov=opt.optimizer.nesterov)
        else:
            raise NotImplementedError('Only support Adam and SGD optimizer.')

        self.Eiters = 0
        print("Model init OK!")

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        if self.opt.cross_model:
            state_dict += [self.criterion.sim.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict[0].items():
            new_state_dict[k] = v
        self.img_enc.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[1].items():
            #name = k.replace('module.', '') # remove `module.`
            new_state_dict[k] = v
        self.txt_enc.load_state_dict(new_state_dict, strict=True)
        new_state_dict = OrderedDict()

        if len(state_dict)>2:
            new_state_dict = OrderedDict()
            for k, v in state_dict[2].items():
                #name = k.replace('module.', '') # remove `module.`
                new_state_dict[k] = v
            self.criterion.sim.load_state_dict(new_state_dict, strict=False)
            new_state_dict = OrderedDict()

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        if self.opt.cross_model:
            self.criterion.sim.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        if self.opt.cross_model:
            self.criterion.sim.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        lengths = Variable(lengths, volatile=volatile)

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, instance_ids, mask=None, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, mask=mask)
        loss = loss #/ self.opt.batch_size
        self.logger.update('Le', loss.data, img_emb.size(0)) 
        return loss

    def train_emb(self, images, captions, lengths, ids=None,
                  instance_ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)
        # measure accuracy and record loss
        self.optimizer.zero_grad()

        l_list = [int(i) for i in lengths]
        mask = Variable(torch.ByteTensor([i*[1] + (max_length+3-i)*[0] for i in l_list])).cuda()
        loss = self.forward_loss(img_emb, cap_emb, instance_ids, mask)
        # compute gradient and do optimization
        loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)

        if self.embed_mask is not None:
            for i, mask in enumerate(self.embed_mask):
                if mask:
                    self.txt_enc.module.embed.weight.grad.data[i].zero_()
        self.optimizer.step()
