import pickle
import os
import time
import shutil

import torch
import yaml
from easydict import EasyDict

import data
from vocab import Vocabulary  # NOQA
from model import CAMP
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

import logging
import tensorboard_logger as tb_logger

import argparse


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='',
                        help='Config path.')
    args = parser.parse_args()
    with open(args.config) as f:
        opt = yaml.load(f)
    opt = EasyDict(opt['common'])
    opt.learning_rate = opt.learning_rate * (128.0/opt.batch_size)
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)
    opt.distributed = False

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)

    print(len(train_loader), len(val_loader), opt.batch_size)

    # Construct the model
    model = CAMP(opt)

    # Train the Model
    best_rsum = 0

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader, tb_logger)

        if epoch % opt.val_epoc == 0:
            # evaluate on validation set
            rsum = validate(opt, val_loader, model, tb_logger)

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_'+ str(epoch) +'.pth.tar', prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader, tb_logger):
    print("start to train")
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    # switch to train mode
    model.train_start()

    end = time.time()
    print("start loading data...")
    for i, train_data in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
    
        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        #if model.Eiters % opt.val_step == 0:
        #    validate(opt, val_loader, model, tb_logger)
            # switch to train mode
        #    model.train_start()


def validate(opt, val_loader, model, tb_logger):
    # compute the encoding for all the validation images and captions
    print("start validate")
    model.val_start()


    img_embs, cap_embs, cap_masks = encode_data(
        model, val_loader, opt.log_step, logging.info)

    # caption retrieval
    (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr), (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr) = i2t(img_embs, cap_embs, cap_masks, measure=opt.measure, model=model)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr))
    # image retrieval
    #(r1i, r5i, r10i, medri, meanr) = t2i(
    #    img_embs, cap_embs, measure=opt.measure, model=model)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr))
    # sum of recalls to be used for early stopping
    currscore = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10

    # record metrics in tensorboard
    tb_logger.log_value('i2t_r1', i2t_r1, step=model.Eiters)
    tb_logger.log_value('i2t_r5', i2t_r5, step=model.Eiters)
    tb_logger.log_value('i2t_r10', i2t_r10, step=model.Eiters)
    tb_logger.log_value('i2t_medr', i2t_medr, step=model.Eiters)
    tb_logger.log_value('i2t_meanr', i2t_meanr, step=model.Eiters)
    tb_logger.log_value('t2i_r1', t2i_r1, step=model.Eiters)
    tb_logger.log_value('t2i_r5', t2i_r5, step=model.Eiters)
    tb_logger.log_value('t2i_r10', t2i_r10, step=model.Eiters)
    tb_logger.log_value('t2i_medr', t2i_medr, step=model.Eiters)
    tb_logger.log_value('t2i_meanr', t2i_meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
