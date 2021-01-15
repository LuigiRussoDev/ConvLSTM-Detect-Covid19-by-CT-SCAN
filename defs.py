import imageio
import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, BatchNormalization, Activation, Conv2D, Flatten, Dense, add, Dropout, \
    AveragePooling2D,Conv3D,MaxPooling3D,GlobalAveragePooling3D

from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.models import Model, Sequential
import matplotlib.pyplot as plt
import numpy as np
import itertools
from itertools import cycle
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy import interp
import tifffile as tiff
import os
import skimage
from skimage.transform import resize
from tqdm import tqdm

def MiniModel(input_shape):
    images = Input(input_shape)


    net = ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu',
                     return_sequences=True, use_bias=False, data_format='channels_last')(images)
    net = BatchNormalization()(net)
    net = Flatten()(net)
    net = Dense(1,activation='sigmoid')(net)
    model = Model(inputs=images, outputs=net)
    return model

def roc_each_classes(test_y,y_pred):
    n_classes = 2
    # Plot linewidth.
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(7)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes ), colors):
        if i == 0:
            cl = "Normal"
        if i==1:
            cl = "Covid"


        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f}) %s'
                       ''.format(i,roc_auc[i]) %cl )

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('Roc_each_classes.jpg')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(10)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    y = np.repeat(np.arange(0, 2), 75)
    plt.xlim(-0.5, len(np.unique(y)) - 0.5)  # ADD THIS LINE
    plt.ylim(len(np.unique(y)) - 0.5, -0.5)  # ADD THIS LINE
    plt.savefig("confusion_matrix_big.png")


def get_data(folder):
    X = []
    y = []

    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['normal']:
                label = 0
            else: #caso covid folder
                label = 1
            for image_filename in tqdm(os.listdir(folder + folderName)):
                #img_file = cv2.imread(folder + folderName + '/' + image_filename,-1)
                #img_file = tiff.imread(folder + folderName + '/' + image_filename)
                img_file = imageio.imread(str(folder + folderName + '/' + image_filename))
                if img_file is not None:
                    #img_file = skimage.transform.resize(img_file, (256,256, 1))
                    img_file = skimage.transform.resize(img_file, (256,256,1))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


