import matplotlib.pyplot as plt
import gluoncv as gcv
from gluoncv import utils
from mxnet import nd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


CLASSES = ['cheminee', 'eaf']


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


def get_ouput_model(image_path, name_model, thresh):
    x, img = gcv.data.transforms.presets.ssd.load_test(image_path, 512)
    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=CLASSES, pretrained_base=False)
    net.load_parameters(name_model)
    box_ids, scores, bboxes = net(x)
    n_box_ids, n_scores, n_bboxes = filter_eaf(bboxes, box_ids, scores)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = utils.viz.plot_bbox(img, n_bboxes[0], n_scores[0],
                            n_box_ids[0], class_names=net.classes, ax=ax, thresh=thresh)
    plt.axis('off')
    fig.savefig('/ouputs/test.png')
    plt.close(fig)
    return n_scores.asnumpy()[0][0][0]


def get_test_set():
    path_test = '../Data/EAF_2labels/VOC2021/ImageSets/Main/test.txt'
    path_image = '../Data/EAF_2labels/VOC2021/JPEGImages/'
    img_list = []
    with open(path_test, 'r') as f:
        readlines = f.read()
        img_list = readlines.split('\n')
    pather = lambda x: path_image + x +'.jpg'
    img_list = list(map(pather, img_list))
    return img_list

def ROC_curve_thresh(y_true, y_scores, taux_fp):
    fpr,tpr,threshholds= roc_curve(y_true, y_scores)
    arg = np.argmax(tpr[np.argwhere(fpr <taux_fp)])
    opti_thresh = threshholds[arg]
    print(f'seuil de confiance optimal : {opti_thresh}, \n avec un taux de faux positif de: {fpr[arg]} \n avec un taux de vrai positif de: {tpr[arg]} \n pour la condition taux de faux positif  < {taux_fp}')
    roc_auc = auc(fpr, tpr)
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
    f.savefig('/results/ROC_curve.png')
    return opti_thresh


def get_labels_and_scores(val_dataset, test_img_list, net):
    transf = gcv.data.transforms.presets.rcnn.FasterRCNNDefaultValTransform(512)
    test_true = val_dataset.transform(transf)
    y_true= np.zeros((len(val_dataset)))
    y_scores = np.zeros((len(val_dataset)))
    x_list_test, _ = gcv.data.transforms.presets.ssd.load_test(test_img_list, 512)
    for i, (x, data) in enumerate(zip(x_list_test, test_true)):
        _, label, _ = data
        box_ids, scores, bboxes = net(x)
        _, n_scores, _ = filter_eaf(bboxes,box_ids,scores)
        if 1 in label[:, 4:5]:
            y_true[i] = 1
        y_scores[i] = n_scores.asnumpy()[0][0][0]
    return y_true, y_scores


def get_predictions(score, thresh):
    if score >= thresh:
        return True
    return False


def plot_train(epochs, ce_loss_list, ce_loss_val,
               smooth_loss_list, smooth_loss_val, map_list):
    f = plt.figure(figsize=(5,10))
    ax = f.add_subplot(311)

    ax.plot(epochs, ce_loss_list)
    ax.plot(epochs, ce_loss_val)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Loss ce for training')
    plt.legend(['train', 'val'], loc='upper left')
    ax1 = f.add_subplot(312)
    ax1.plot(epochs, smooth_loss_list)
    ax1.plot(epochs, smooth_loss_val)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Loss smooth for training')
    plt.legend(['train', 'val'], loc='upper left')
    ax2 = f.add_subplot(313)
    ax2.plot(epochs, map_list)
    plt.ylabel('MAP')
    plt.xlabel('epoch')
    plt.title('MAP for eaf')