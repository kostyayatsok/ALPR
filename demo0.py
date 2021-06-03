# Import all necessary libraries.
import os
import numpy as np
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import copy
from PIL import Image
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

DIR = "C:/projects/python/ANPR"
sys.path.append(DIR)

from YoloV5Detector import Detector
detector_tf = Detector()
detector_tf.loadTf()

from CornersSegmentationDetector import CornersSegmentationDetector
corners_detector = CornersSegmentationDetector()

from OCR import OCR
class Test(OCR):
    def __init__(self):
        OCR.__init__(self)
        # only for usage model
        # in train generate automaticly
        self.letters = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "E", "H", "K", "M", "O", "P", "T", "X", "Y"]
        self.EPOCHS = 500
textDetector = Test()

textDetector.load('./weights/ocr.hdf5'); fld = 'img1'

# models_path = "C:\\projects\\python\\ANPR\\models\\"
# detector_tf.save_model(models_path + 'bboxes')
# corners_detector.save_model(models_path + 'corners')
# textDetector.save_model(models_path + 'ocr')

import glob
jpgFilenamesList = glob.glob('C:/projects/python/nomeroff-net-master/datasets/mine/**/*.jpg')

import ntpath
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

correct_pred, error_pred, detector_errors = 0, 0, 0

import matplotlib
matplotlib.use('Qt5Agg')

directory = 'C:/projects/python/nomeroff-net-master/datasets/'
if not os.path.exists(directory + 'mistakes/' + fld):
    os.makedirs(directory + 'mistakes/' + fld)
if not os.path.exists(directory + 'mistakes/' + fld + '-full'):
    os.makedirs(directory + 'mistakes/' + fld + '-full')

total_time, n = 0, 0

re_file = open('re.txt', 'w')
for img_path in jpgFilenamesList:
    # try:
    if True:
        img = mpimg.imread(img_path)
        if img_path.endswith('.png'):
            img = np.asarray(Image.open(img_path).convert('RGB'))

        truenum = "".join(path_leaf(img_path).split('.')[0].split())
        truenum = truenum.replace("0", "O")

        start = time.time()
        targetBoxes = detector_tf.detect_bbox_tf(copy.deepcopy(img), min_accuracy=0)
        zones = corners_detector.detect(copy.deepcopy(img), targetBoxes)
            
        found, foundTop3 = False, False
        print("***", truenum, "[]")
        for zone in zones:
            textArr, probs = textDetector.predict(zone, return_acc=True)
            textArr, probs = textArr.flatten(), probs.flatten()
            
            top_candidates = np.argsort(probs)[::-1]
            textArr = textArr[top_candidates]
            probs = probs[top_candidates]
            _, uniq_idxs = np.unique(textArr, return_index=True)
            
            textArr = textArr[np.sort(uniq_idxs)]
            probs = probs[np.sort(uniq_idxs)]
            
            _foundTop3 = truenum in textArr[:3]
            _found = truenum in textArr[:1]
            print(
                "..." if _found else ("---" if _foundTop3 else "***"),
                truenum, np.hstack((textArr.reshape(-1, 1), probs.reshape(-1, 1))).tolist())

            foundTop3 |= _foundTop3
            found |= _found
        

        end = time.time()
        total_time += end - start
        n += 1
        if found:
            correct_pred += 1
        else:
            if len(targetBoxes) == 0 or len(zones) == 0:
                detector_errors += 1
            else:
                error_pred += 1

        print(
            "..." if found else ("---" if foundTop3 else "***"),
            truenum, correct_pred + error_pred + detector_errors,
            correct_pred, error_pred, detector_errors,
            int(10000*correct_pred/(correct_pred+error_pred+detector_errors))/100, file=sys.stderr)

    # except Exception as e:
    #     print(e, file=re_file)
    #     print('Ошибка проверки.', sys.exc_info()[0], file=re_file)

print('Elapsed time: ', total_time, ' for ', n, 'images')
print('Seconds per image: ', total_time / n)
print('Images per second: ', n / total_time)