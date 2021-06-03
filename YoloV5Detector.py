# load default packages
import os
import time
import sys
import pathlib
# import torch
import numpy as np

# download and append to path yolo repo
NOMEROFF_NET_DIR = os.path.join(pathlib.Path(__file__).parent.absolute(), "../")
YOLOV5_DIR       = os.environ.get("YOLOV5_DIR", os.path.join(NOMEROFF_NET_DIR, 'yolov5'))
YOLOV5_URL       = "https://github.com/ultralytics/yolov5.git"
if not os.path.exists(YOLOV5_DIR):
    from git import Repo
    Repo.clone_from(YOLOV5_URL, YOLOV5_DIR)
sys.path.append(YOLOV5_DIR)

# load yolo packages
# from models.experimental import attempt_load
# from utils.datasets import letterbox
# from utils.general import non_max_suppression, scale_coords
# from utils.torch_utils import select_device, load_classifier, time_synchronized

# load NomerooffNet packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Base')))

import onnx
from onnx_tf.backend import prepare

from tensorflow.keras.models import load_model
import tensorflow as tf
class Detector:
    """

    """
    @classmethod
    def get_classname(cls):
        return cls.__name__

    # def loadModel(self,
    #              weights,
    #              device='cuda'):
    #     device = select_device(device)
    #     model = attempt_load(weights, map_location=device)  # load FP32 model
    #     stride = int(model.stride.max())
        
    #     half = device.type != 'cpu'  # half precision only supported on CUDA
    #     if half:
    #         model.half()  # to FP16
        
    #     self.model  = model
    #     self.device = device
    #     self.half   = half

    def loadTf(self, path='./weights/bboxes'):
        # model = onnx.load(path)
        # model = prepare(model)
        # model.export_graph(path[:-5])
        self.model = load_model(path, compile=False)

    def detect_bbox_tf(self, img, img_size=640, stride=32, min_accuracy=0.):
        """
        TODO: input img in BGR format, not RGB; To Be Implemented in release 2.2
        """
        
        import imgaug as ia
        import imgaug.augmenters as iaa

        img_orig = img.copy()
        # normalize
        img_shape = img.shape
        
        seq = iaa.Sequential([
                    iaa.PadToAspectRatio(1, position='center'),
                    iaa.Resize({"height": img_size, "width": img_size}),
                ])
        img = seq(image=img)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img.astype(np.float32), axis=0)

        pred = self.model(images=img)['561'][0]
        # print(pred.keys())
        # print(pred)
        # print(pred.shape)
        # print(pred[0, 0, 0, 0:3, :])
        scores = pred[:, 4].numpy()
        x_min = pred[:,0] - pred[:,2]/2 
        x_max = pred[:,0] + pred[:,2]/2
        y_min = pred[:,1] - pred[:,3]/2
        y_max = pred[:,1] + pred[:,3]/2

        if img_shape[0] < img_shape[1]:
            scale = img_shape[1] / img_size
            x_pad = 0
            y_pad = (img_shape[1] - img_shape[0]) / 2
        else:
            scale = img_shape[0] / img_size
            x_pad = (img_shape[0] - img_shape[1]) / 2
            y_pad = 0
        x_min = x_min.numpy() * scale - x_pad
        x_max = x_max.numpy() * scale - x_pad
        y_min = y_min.numpy() * scale - y_pad
        y_max = y_max.numpy() * scale - y_pad

        boxes = np.hstack((
            y_min.reshape(-1, 1),
            x_min.reshape(-1, 1),
            y_max.reshape(-1, 1),
            x_max.reshape(-1, 1))).astype(np.float32)
        # print(boxes)
        # print(scores)
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=10, score_threshold=1e-5)#float(min_accuracy))
        selected_boxes = boxes[selected_indices,:]
        selected_scores = np.take(scores, selected_indices)
        # print(selected_indices)
        # print(selected_scores)
        # print(np.argmax(scores))
        # import matplotlib.pyplot as plt
        # plt.imshow(img_orig)
        # plt.scatter(selected_boxes[:,1], selected_boxes[:,0])
        # plt.scatter(selected_boxes[:,3], selected_boxes[:,2])
        # plt.show()

        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # from PIL import Image
        # # Create figure and axes
        # fig, ax = plt.subplots()
        # # Display the image
        # ax.imshow(img_orig)
        # # Create a Rectangle patch
        # rect = patches.Rectangle((selected_boxes[0,1], selected_boxes[0,2]),
        #  selected_boxes[0,3] - selected_boxes[0,1],
        #  selected_boxes[0,0] - selected_boxes[0,2],
        #   linewidth=1, edgecolor='r', facecolor='none')

        # # Add the patch to the Axes
        # ax.add_patch(rect)
        # plt.show()
        # # Apply NMS
        # pred = non_max_suppression(pred)
        # res = []
        # for i, det in enumerate(pred): 
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_shape).round()
        #         res.append(det.cpu().detach().numpy())
        result = []
        for i, box in enumerate(selected_boxes):
            if selected_scores[i] >= min_accuracy:
                result.append([box[1], box[0], box[3], box[2], selected_scores[i], 0])
        return result

    def save_model(self, path):
        self.model.save(path, save_format='tf')