import logging
import os 
import argparse
import time
import gluoncv as gcv
import mxnet as mx
import numpy as np
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
from mxnet import gluon, autograd
from Trainer.trainer_ssd import (load_data_VOC, get_pretrained_model,
                                 get_train_dataloader, get_val_dataloader,
                                 validate, save_params, get_ctx)
parser = argparse.ArgumentParser(
    description="do training"
)

parser.add_argument(
    "--epoch", default=15, dest="epoch", help="number of training epoch"
)

parser.add_argument(
    "--save_prefix",default='/logs/ssd_512', dest="save_prefix", help="number of training epoch"
)

parser.add_argument(
    "--start_epoch", default=0, dest="start_epoch", help="number of training epoch"
)

parser.add_argument(
    "--save_interval", default=5, dest="save_interval", help="interval of epoch for saving models"
)

args = parser.parse_args()


def train(net, ctx, train_data, val_data, classes=['cheminee', 'eaf']):
    #set up logging
    
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    map_list = []
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'adam',
        {'learning_rate': 0.001, 'wd': 0.0005})
    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')
    ce_metric_val = mx.metric.Loss('CrossEntropy')
    smoothl1_metric_val = mx.metric.Loss('SmoothL1')
    epochs = np.arange(args.epoch)
    for epoch in range(args.start_epoch, args.epoch):
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            logging.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()
        eval_metric = VOCMApMetric(iou_thresh=0.5, class_names=classes)
        map_name, mean_ap = validate(net, val_data,ctx, eval_metric)
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        current_map = mean_ap[1]
        save_params(net, best_map, current_map, epoch, args.save_interval)
        print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
        logging.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))


if __name__== '__main__':
    classes = ['cheminee', 'eaf']
    train_dataset, val_dataset = load_data_VOC()
    print('taille données train : {len(train_dataset)}', 'taille données val: {len(val_dataset)}')
    train_data, val_data = get_train_dataloader(net, train_dataset), get_val_dataloader(val_dataset)
    net = get_pretrained_model(classes)
    ctx = get_ctx()

    train(net, ctx, train_data, val_data, classes)