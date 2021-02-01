import matplotlib.pyplot as plt
import gluoncv as gcv
import mxnet as mx
import numpy as np
import cv2
import os
import logging
import time
import sys
import json
from mxnet import autograd, gluon
from gluoncv import model_zoo
from gluoncv import utils
from mxnet import nd
from sklearn.metrics import roc_curve, auc
try:
    sys.path.append('../detector_utils')
    from trainer import (VOCLike, get_pretrained_model,
                                    ssd_train_dataloader, ssd_val_dataloader,
                                    validate, save_params, get_ctx, val_loss)
except:
    from detector_utils.trainer import (VOCLike, get_pretrained_model,
                                    ssd_train_dataloader, ssd_val_dataloader,
                                    validate, save_params, get_ctx, val_loss)
CLASSES = ['cheminee', 'eaf']

class BaseDetector(object):
    def __init__(self):
        super().__init__()
        self.y_true=None
        self.y_scores=None
        self.thresh=None
        self.save_prefix=None


    def eval(self, taux_fp, save_plot):
        fpr,tpr,threshholds= roc_curve(self.y_true, self.y_scores)
        arg = np.argmax(tpr[np.argwhere(fpr < taux_fp)])
        fn = 1-fpr[arg]
        opti_thresh = threshholds[arg]
        print(f'seuil de confiance optimal : {opti_thresh:.3f}, \n avec un taux de faux positif de: {fpr[arg]} \n avec un taux de vrai positif de: {tpr[arg]} \n pour la condition taux de faux positif  < {taux_fp}')
        roc_auc = auc(fpr, tpr)
        if save_plot:
            f =plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title("Courbe ROC pour la détection de l'EAF")
            plt.legend(loc="lower right")
            f.savefig('results_ROC/' + self.save_prefix + '_ROC_curve.png')
        if not os.path.exists('logs/eval.json'):
            results =  {'model':[self.save_prefix],
                        'description': [self.description_train],
                        'taux_faux_positif': [fpr],
                        'taux_vrai_positif':[tpr],
                        'taux_faux_positif_fixe': [taux_fp],
                        'taux_faux_negatif':[fn],
                        'taux_vrai_p': [tpr[arg]],
                        'seuil_optimal':[opti_thresh]}
            with open('logs/eval.json', 'w') as json_file:
                json.dump(results, json_file)
        else:
            with open('eval.json') as f:
                data = json.load(f)
            data['model'].append(self.save_prefix)
            data['description'].append(self.description_train)
            data['taux_faux_positif'].append(fpr)
            data['taux_vrai_positif'].append(tpr)
            data['taux_faux_positif_fixe'].append(taux_fp)
            data['taux_faux_negatif'].append(fn)
            data['taux_vrai_p'].append([tpr[arg]])
            data['seuil_optimal'].append(opti_thresh)
            with open('eval.json', 'w') as json_file:
                json.dump(data, json_file)
            
        with open('logs/'+'eval.log', 'a') as log:
            log.write(f"modèle: {self.save_prefix} \n description de l'entrainement: {self.description_train} \n seuil de confiance optimal : {opti_thresh:.3f}, \n on a un taux de faux positif de: {fpr[arg]} \n on a un taux de vrai positif de: {tpr[arg]} \n on a un taux de faux négatif de: {fn} \n pour la condition taux de faux positif  < {taux_fp} \n")
        self.thresh = opti_thresh

class ModelBasedDetector(BaseDetector):
    def __init__(self, net=None, thresh=None, save_prefix='ssd_512_test',data_path='../Data/EAF_real',
                 data_path_test='../Data/EAF_real',train_dataloader=ssd_train_dataloader,
                 val_dataloader=ssd_val_dataloader, batch_size=10):
        super().__init__()
        self.net = net
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_dataset = None
        self.val_dataset = None
        self.train_data = None
        self.val_data = None
        self.loss_val_data = None
        self.thresh = thresh
        self.save_prefix = save_prefix
        self.tests_set = None
        self.data_path = data_path
        self.data_path_test = data_path_test
        self.mean_iou = None
        self.ctx = None
        self.batch_size = batch_size
        self.description_train = None
    
    @classmethod
    def from_pretrained(cls, data_path, batch_size=10, base_model='ssd_512_mobilenet1.0_custom', save_prefix='ssd_512_test2'):
        net = model_zoo.get_model(base_model, classes=CLASSES, pretrained_base=False, transfer='voc')
        return cls(net=net, data_path=data_path, save_prefix=save_prefix, batch_size=batch_size)

    @classmethod
    def from_finetuned(cls, name_model, data_path_test, batch_size=10, base_model='ssd_512_mobilenet1.0_custom', thresh=0.3, save_prefix='ssd_512_test2'):
        net = model_zoo.get_model(base_model, classes=CLASSES, pretrained_base=False, transfer='voc')
        net.load_parameters(name_model)
        return cls(net=net, data_path_test=data_path_test, save_prefix=save_prefix, batch_size=batch_size, thresh=thresh)

    def set_dataset(self, split=2021):
        self.train_dataset = VOCLike(root=self.data_path, splits=[(split, 'train')])
        self.val_dataset = VOCLike(root=self.data_path, splits=[(split, 'val')])
        self.train_data = self.train_dataloader(self.net, self.train_dataset, batch_size=self.batch_size)
        self.val_data = self.val_dataloader(self.val_dataset, batch_size=self.batch_size)
        self.loss_val_data = self.train_dataloader(self.net, self.val_dataset, batch_size=self.batch_size)

    def set_test_dataset(self, split=2021):
        self.test_dataset = VOCLike(root=self.data_path_test, splits=[(split, 'test')])

    def plot_predict(self):
        self.set_test_dataset()
        transf = gcv.data.transforms.presets.rcnn.FasterRCNNDefaultValTransform(512)
        test_true = self.test_dataset.transform(transf)
        items = self.test_dataset._items
        img_paths = [p[0]+'/JPEGImages/'+p[1]+'.jpg' for p in items]
        x_list_test, img_list_test = gcv.data.transforms.presets.ssd.load_test(img_paths, 512)

        for i, (x, orig_img, data) in enumerate(zip(x_list_test, img_list_test, test_true)):
            _,label,_ = data
            true_box_ids, true_scores, true_bboxes= filter_eaf(nd.array([label[:, :4]]),nd.array([label[:, 4:5]]), nd.array(np.ones(shape=(1,2,1))))
            box_ids, scores, bboxes = self.net(x)
            n_box_ids, n_scores, n_bboxes = filter_eaf(bboxes,box_ids,scores)
            inter_bboxes = mx.nd.concat(true_bboxes,n_bboxes)
            inter_box_ids = mx.nd.concat(true_box_ids,n_box_ids)
            inter_scores = mx.nd.concat(true_scores,n_scores)
            ax = gcv.utils.viz.plot_bbox(orig_img, inter_bboxes[0], inter_scores[0], inter_box_ids[0], class_names=detector.net.classes,thresh=thresh)

        plt.show()
    def set_tests(self):
        path_test = self.data_path_test + '/VOC2021/ImageSets/Main/test.txt'
        path_image = self.data_path_test + '/VOC2021/JPEGImages/'
        img_list = []
        with open(path_test, 'r') as f:
            readlines = f.read()
            img_list = readlines.split('\n')
        pather = lambda x: path_image + x +'.jpg'
        img_list = list(map(pather, img_list))
        print(img_list)
        self.tests_set = img_list

    def set_labels_and_scores(self):
        self.set_test_dataset()
        transf = gcv.data.transforms.presets.rcnn.FasterRCNNDefaultValTransform(512)
        test_true = self.test_dataset.transform(transf)
        y_true= np.zeros((len(self.test_dataset)))
        y_scores = np.zeros((len(self.test_dataset)))
        iou_list = np.zeros((len(self.test_dataset)))
        x_list_test, _ = gcv.data.transforms.presets.ssd.load_test(self.tests_set, 512)
        for i, (x, data) in enumerate(zip(x_list_test, test_true)):
            _, label, _ = data
            _, _, true_bboxes= ModelBasedDetector.filter_eaf(
                nd.array([label[:, :4]]),nd.array([label[:, 4:5]]), nd.array(np.ones(shape=(1,2,1))))
            box_ids, scores, bboxes = self.net(x)
            _, n_scores, n_bboxes = ModelBasedDetector.filter_eaf(bboxes,box_ids,scores)
            iou = utils.bbox_iou(true_bboxes[0].asnumpy(), n_bboxes[0].asnumpy())
            iou_list[i] = iou
            if 1 in label[:, 4:5]:
                y_true[i] = 1
            y_scores[i] = n_scores.asnumpy()[0][0][0]
        mean_iou = np.mean(iou_list[iou_list > 0])
        with open('logs/'+'eval.log', 'a') as f:
            f.write(f"L'intersection over union moyen est : {mean_iou:.3f}\n")
        self.y_true, self.y_scores, self.mean_iou = y_true, y_scores, mean_iou

    def set_ctx(self):
        try:
            a = mx.nd.zeros((1,), ctx=mx.gpu(0))
            self.ctx = [mx.gpu(0)]
        except:
            self.ctx = [mx.cpu()]

    def train(self, start_epoch, epoch, description):
        self.description_train=description
        print(self.batch_size)
        self.set_ctx()
        self.set_dataset()
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_file_path = 'logs/'+ self.save_prefix + '_train.log'
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        logger.info(f'save_prefix={self.save_prefix}, start_epoch={start_epoch}, epoch={epoch}')
        logger.info('Start training from [Epoch {}]'.format(start_epoch))
        best_map = [0]
        epochs = np.arange(int(epoch))
        ce_loss_list = []
        ce_loss_val = []
        smooth_loss_list = []
        smooth_loss_val = []
        map_list = []
        self.net.collect_params().reset_ctx(self.ctx)
        trainer = gluon.Trainer(
            self.net.collect_params(), 'adam',
            {'learning_rate': 0.001, 'wd': 0.0005})
        mbox_loss = gcv.loss.SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')
        ce_metric_val = mx.metric.Loss('CrossEntropy')
        smoothl1_metric_val = mx.metric.Loss('SmoothL1')
        for epoch in range(int(start_epoch), int(epoch)):
            ce_list = np.zeros(shape=(len(self.train_data)))
            smooth_list = np.zeros(shape=(len(self.train_data)))
            ce_metric.reset()
            smoothl1_metric.reset()
            tic = time.time()
            btic = time.time()
            self.net.hybridize(static_alloc=True, static_shape=True)
            for i, batch in enumerate(self.train_data):
                batch_size = batch[0].shape[0]
                data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)
                cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx, batch_axis=0)
                box_targets = gluon.utils.split_and_load(batch[2], ctx_list=self.ctx, batch_axis=0)
                with autograd.record():
                    cls_preds = []
                    box_preds = []
                    for x in data:
                        cls_pred, box_pred, _ = self.net(x)
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
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
                btic = time.time()
                ce_list[i] = loss1
                smooth_list[i] = loss2
            loss1_val, loss2_val = val_loss(self.net, self.loss_val_data, self.ctx)
            ce_loss_val.append(loss1_val)
            smooth_loss_val.append(loss2_val)
            if len(ce_loss_list) > 1 and epoch > 5:
                if ce_loss_val[-1]>ce_loss_val[-2]:
                    print('Early stopping')
                    return
            ce_loss_list.append(np.mean(ce_list))
            logger.info('[Epoch {}] Validation, {}={:.3f}, {}={:.3f}'.format(
                epoch, name1, loss1_val, name2, loss2_val))
            smooth_loss_list.append(np.mean(smooth_list))
            eval_metric = gcv.utils.metrics.voc_detection.VOCMApMetric(iou_thresh=0.4, class_names=CLASSES)
            map_name, mean_ap = validate(self.net, self.val_data, self.ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            current_map = mean_ap[1]
            save_params(self.net, best_map, current_map, epoch, self.save_prefix)
            map_list.append(current_map)
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
        return epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list

    def predict(self, image_path, output_path, close=True):
        x, img = gcv.data.transforms.presets.ssd.load_test(image_path, 512)
        path, filename = os.path.split(image_path)
        box_ids, scores, bboxes = self.net(x)
        n_box_ids, n_scores, n_bboxes = ModelBasedDetector.filter_eaf(bboxes, box_ids, scores)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = utils.viz.plot_bbox(img, n_bboxes[0], n_scores[0],
                                n_box_ids[0], class_names=self.net.classes, ax=ax, thresh=self.thresh)
        plt.axis('off')
        fig.savefig(output_path+filename)
        if close:
            plt.close(fig)
        else:
            plt.show()
        score = n_scores.asnumpy()[0][0][0]
        prediction = score > self.thresh
        return score, prediction


    @staticmethod
    def filter_eaf(bboxes, box_ids, scores):
        new_bboxes, new_box_ids, new_scores = nd.array(np.zeros(shape=(1,1,4))),nd.array(np.zeros(shape=(1,1,1))),nd.array(np.zeros(shape=(1,1,1)))
        found = False
        for index, (box,ids,score)  in enumerate(zip(bboxes[0], box_ids[0], scores[0])):
            if not found:
                if ids[0]==float(1):
                    new_bboxes[0][0] = box
                    new_box_ids[0][0] = ids
                    new_scores[0][0] = score
                    found = True
        return new_box_ids, new_scores, new_bboxes

