import matplotlib.pyplot as plt
import gluoncv as gcv
from gluoncv import utils
import cv2
import os
from mxnet import nd
import numpy as np
from sklearn.metrics import roc_curve, auc
from pathlib import Path

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


def plot_train(
    epochs,
    ce_loss_list,
    ce_loss_val,
    smooth_loss_list,
    smooth_loss_val,
    map_list,
    save_prefix,
    train_results_folder,
    save_plot=True,
):
    '''
    plot the outputs from training
    '''
    Path(train_results_folder).mkdir(parents=True, exist_ok=True)
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
        f.savefig(results_train_folder + save_prefix + "_train_curves.png")