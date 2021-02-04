import matplotlib.pyplot as plt
import gluoncv as gcv
from gluoncv import utils
import cv2
import os
from mxnet import nd
import numpy as np
from sklearn.metrics import roc_curve, auc


CLASSES = ["cheminee", "eaf"]


def process_output_img(name_path):
    path, filename = os.path.split(name_path)
    img = cv2.imread(name_path)  # Read in the image and convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)  # To invert the text to white
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = img[
        y : y + h, x : x + w
    ]  # Crop the image - note we do this on the original image
    output_image_path = path + "/corrected_" + filename
    cv2.imwrite(output_image_path, rect)
    os.remove(name_path)
    return output_image_path


def ROC_curve_thresh(y_true, y_scores, taux_fp, save_plot):
    fpr, tpr, threshholds = roc_curve(y_true, y_scores)
    arg = np.argmax(tpr[np.argwhere(fpr < taux_fp)])
    opti_thresh = threshholds[arg]
    print(
        f"seuil de confiance optimal : {opti_thresh}, \n avec un taux de faux positif de: {fpr[arg]} \n avec un taux de vrai positif de: {tpr[arg]} \n pour la condition taux de faux positif  < {taux_fp}"
    )
    roc_auc = auc(fpr, tpr)
    if save_plot:
        f = plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Courbe ROC pour la détection de l'EAF")
        plt.legend(loc="lower right")
        f.savefig("results/ROC_curve.png")
    return opti_thresh


def get_test_set():
    path_test = "../Data/EAF_2labels/VOC2021/ImageSets/Main/test.txt"
    path_image = "../Data/EAF_2labels/VOC2021/JPEGImages/"
    img_list = []
    with open(path_test, "r") as f:
        readlines = f.read()
        img_list = readlines.split("\n")
    pather = lambda x: path_image + x + ".jpg"
    img_list = list(map(pather, img_list))
    return img_list


def get_labels_and_scores(val_dataset, test_img_list, net):
    transf = gcv.data.transforms.presets.rcnn.FasterRCNNDefaultValTransform(512)
    test_true = val_dataset.transform(transf)
    y_true = np.zeros((len(val_dataset)))
    y_scores = np.zeros((len(val_dataset)))
    iou_list = np.zeros((len(val_dataset)))
    x_list_test, _ = gcv.data.transforms.presets.ssd.load_test(test_img_list, 512)
    for i, (x, data) in enumerate(zip(x_list_test, test_true)):
        _, label, _ = data
        _, _, true_bboxes = filter_eaf(
            nd.array([label[:, :4]]),
            nd.array([label[:, 4:5]]),
            nd.array(np.ones(shape=(1, 2, 1))),
        )
        box_ids, scores, bboxes = net(x)
        _, n_scores, n_bboxes = filter_eaf(bboxes, box_ids, scores)
        iou = utils.bbox_iou(true_bboxes[0].asnumpy(), n_bboxes[0].asnumpy())
        iou_list[i] = iou
        if 1 in label[:, 4:5]:
            y_true[i] = 1
        y_scores[i] = n_scores.asnumpy()[0][0][0]
    mean_iou = np.mean(iou_list[iou_list > 0])
    return y_true, y_scores, mean_iou


def plot_train(
    epochs,
    ce_loss_list,
    ce_loss_val,
    smooth_loss_list,
    smooth_loss_val,
    map_list,
    save_prefix,
    save_plot=True,
):
    '''
    plot the outputs from training
    '''
    f = plt.figure(figsize=(5, 10))
    ax = f.add_subplot(311)

    ax.plot(epochs, ce_loss_list)
    ax.plot(epochs, ce_loss_val)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Loss ce for training")
    plt.legend(["train", "val"], loc="upper left")
    ax1 = f.add_subplot(312)
    ax1.plot(epochs, smooth_loss_list)
    ax1.plot(epochs, smooth_loss_val)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Loss smooth for training")
    plt.legend(["train", "val"], loc="upper left")
    ax2 = f.add_subplot(313)
    ax2.plot(epochs, map_list)
    plt.ylabel("MAP")
    plt.xlabel("epoch")
    plt.title("MAP for eaf")
    if save_plot:
        f.savefig("results_train/" + save_prefix + "_train_curves.png")