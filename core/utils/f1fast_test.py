from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import skimage
import sys

import os
import time
from glob import glob
import skimage
import sklearn
start_time = time.time()

def count_raw(ground_truth, predict):
    """
    Count and return values needed for precision and recall calculation
    via the formulas:
    precision = true_positive / (true_positive + false_positive) =
    true_positive / selected_elements,
    recall = true_positive / (true_positive + false_negative) =
    true_positive / relevant_elements.
    These values can be summed for all images in the dataset.
    true_positive and selected_elements depend on a threshold,
    while relevant_elements does not.
    
    For each threshold of 0-255 we apply it to the predict array and
    count the sum of intersection with gt and sum of the thresholded predict.
    As pixels are discretized into 0-255 values, there are many
    efficient ways to do this compared to the sklearn
    precision-recall function.
    """
    predict = (predict * 255).astype(np.int16) # Integers are faster
    
    true_positive = np.zeros((257)) # Equivalence to sklearn.metrics.precision_recall_curve
    selected_elements = np.zeros((257))
    relevant_elements = np.sum(ground_truth)
    
    unique, counts = np.unique(predict, return_counts=True)
    counts_thr = np.zeros((256))
    for i, pixel_val in enumerate(unique):
        counts_thr[pixel_val] = counts[i]
        
    unique, counts = np.unique(ground_truth * (predict + 1), return_counts=True) # +1 makes difference between gt pixels and all other pixels
    counts_inter = np.zeros((256))
    for i, pixel_val in enumerate(unique[1:]):
        counts_inter[pixel_val - 1] = counts[i + 1]
    
    for i in range(255, -1, -1):
        true_positive[i] = true_positive[i + 1] + counts_inter[i]
        selected_elements[i] = selected_elements[i + 1] + counts_thr[i]
        
    return (true_positive, selected_elements, relevant_elements)

def precision_recall_raw(true_positive, selected_elements, relevant_elements):
    """
    Calculate precision and recall from summed true_positive,
    selected_elements and relevant_elements. This function
    (almost) matches output from sklearn.metrics.precision_recall_curve.
    """
    precision = np.zeros((257))
    recall = np.zeros((257))
    threshold = np.zeros((257))
    
    for i in range(257):
        if selected_elements[i] == 0: # Equivalence to sklearn.metrics.precision_recall_curve
            precision[i] = 1
            recall[i] = 0
        elif relevant_elements == 0:
            precision[i] = 0
            recall[i] = 1
        else:
            precision[i] = true_positive[i] / selected_elements[i]
            recall[i] = true_positive[i] / relevant_elements
        threshold[i] = 1.0 / 255 * i

    return (precision, recall, threshold)

# This function can be used when all samples are concatenated in memory,
# which I don't do currently as much memory is needed
def precision_recall(ground_truth, predict):
    a = time.time()
    predict = (predict * 255).astype(np.int16)
    gt = ground_truth
    
    precision = np.zeros((257))
    recall = np.zeros((257))
    threshold = np.zeros((257))

    gt_pixels = np.sum(gt)
    for i in range(257):
        #print(i)
        predict_thr = predict >= i
        true_positive = np.sum(gt * predict_thr)
        selected_pixels = np.sum(predict_thr)
        if selected_pixels == 0:
            precision[i] = 1
            recall[i] = 0
        elif gt_pixels == 0:
            precision[i] = 0
            recall[i] = 1
        else:
            precision[i] = true_positive / selected_pixels
            recall[i] = true_positive / gt_pixels
        threshold[i] = 1.0 / 255 * i
        
        predict = predict[predict_thr]
        gt = gt[predict_thr]

    print(time.time() - a)
    return (precision, recall, threshold)

class F1Accumulator:
    '''
    Feed it with (gt, pred) images, and call get_result to calculate F1-measure for each threshold
    '''
    def __init__(self):
        self.true_positive_sum = np.zeros((257))
        self.selected_elements_sum = np.zeros((257))
        self.relevant_elements_sum = 0

    def add(self, y_test, y_score):
        '''
        y_test: h*w boolean image, ground truth values
        y_score: h*w float image, prediction
        '''
        tp, se, re = count_raw(y_test, y_score)
        self.true_positive_sum += tp
        self.selected_elements_sum += se
        self.relevant_elements_sum += re

    def get_result(self):
        return precision_recall_raw(self.true_positive_sum, self.selected_elements_sum, self.relevant_elements_sum)

if __name__ == '__main__':
    #img_rows = 218
    #img_cols = 512
    img_rows = 436
    img_cols = 1024


    #MODEL = 'ALEXANDRA'
    #MODEL = 'MFFocc'
    #MODEL = 'IRR-PWC'
    MODEL = 'RAFT'

    #res_dir = 'anzina-soft' # Probabilty of occlusions
    #res_dir = 'anzina-hard' # 0/1 thresholded occlusions
    #res_dir = 'cssr1'
    res_dir = 'res_fold'
    #res_dir = 'cssr-hard1'
    #res_dir = 'irr-soft1'
    #res_dir = 'content1'
    #res_dir = 'irr-chairs'

    #gt_dir = 'sintel-occ1'
    gt_dir = 'occlusions_rev'

    # Sum tp, se and re for all images.
    # F1 = 2 * (precision * recall) / (precision + recall)
    # precision = tp / se
    # recall = tp / re
    accumulator = F1Accumulator()
    for i, path in enumerate(sorted(glob(res_dir + '/*/*.png'))):
        print(i)
        name = '/'.join(path.split('/')[1:])
        test_name = os.path.join(gt_dir, name)
        score_name = os.path.join(res_dir, name)

        # OpenCV is faster than skimage
        y_test = cv2.cvtColor(cv2.imread(test_name), cv2.COLOR_BGR2GRAY)
        y_test = cv2.resize(y_test, (img_cols, img_rows)).flatten() / 255
        y_score = cv2.cvtColor(cv2.imread(score_name), cv2.COLOR_BGR2GRAY)
        y_score = cv2.resize(y_score, (img_cols, img_rows)).flatten() / 255

        y_test = np.asarray(y_test, dtype=np.bool)
        y_score = np.asarray(y_score, dtype=np.float16)
        
        accumulator.add(y_test, y_score)
        

    precision, recall, thresholds = accumulator.get_result()

    # Max f-score and figure drawing
    max_f1 = 0
    pr = rc = th = 0
    for i, j in zip(range(len(precision)), thresholds):
        f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        if f1 > max_f1:
            max_f1 = f1
            pr = precision[i]
            rc = recall[i]
            th = j

    print(max_f1, pr, rc, th)
    plt.scatter(rc, pr, s=100)
    plt.step(recall, precision, label=MODEL + ' Fscore={0:0.4f}'.format(max_f1), linewidth=2)

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve (sintel)')
    plt.legend(loc='lower left', prop={'size': 14})
    plt.tight_layout()
    plt.savefig('naive3_sintel.png')
    #plt.show()
