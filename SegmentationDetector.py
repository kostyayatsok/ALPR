import tensorflow as tf
import random
import numpy as np
import os
import json
from cv2 import cv2
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import math
from imgaug.augmentables import Keypoint, KeypointsOnImage
from segmentation_models import Unet
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import Model, Input
from tensorflow.keras.applications.resnet import preprocess_input
import tensorflow.keras.backend as K
import glob
from segmentation_models.losses import categorical_focal_loss, dice_loss, bce_dice_loss, DiceLoss

class SegmentationDetector:
    def __init__(self, NOMEROFF_NET_DIR='./', **args):
        self.IMG_W = 256
        self.IMG_H = 256
        self.IMG_SHAPE = (self.IMG_W, self.IMG_H, 3)
        self.BATCH_SIZE = 32
        self.JSON_FILENAME = 'via_region_data.json'
        self.model = None
        self.buff_weights = NOMEROFF_NET_DIR + '/segmentation_buff_weights.h5'
        
    def generator(self, dir_path, aug=True, val=False, verbose=False):        
        file_list = glob.glob(dir_path + '/*.jpg')
        val_size = len(file_list) // 10
        if val:
            file_list = file_list[-val_size:]
        else:
            file_list = file_list[:-val_size]
        file_idx = 0
        while True:
            X_batch, y_batch = [], []
            for _ in range(self.BATCH_SIZE):
                if file_idx == len(file_list):
                    file_idx = 0
                    random.shuffle(file_list)
                filename = file_list[file_idx]
                image = cv2.imread(filename, flags=cv2.IMREAD_IGNORE_ORIENTATION+cv2.IMREAD_COLOR)
                image = np.flip(image, axis=-1)
                filename = filename[:-3] + 'png'
                mask = cv2.imread(filename, flags=cv2.IMREAD_IGNORE_ORIENTATION+cv2.IMREAD_GRAYSCALE)
            
                if verbose:
                    plt.imshow(image)
                    # plt.imshow(mask, alpha=0.5)
                    plt.show()
                file_idx += 1
                if aug:
                    # https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html
                    seq = iaa.Sequential([
                        iaa.PadToAspectRatio(self.IMG_W/self.IMG_H),
                        iaa.AverageBlur(k=((0, 7), (0, 3))), 
                        iaa.MultiplyBrightness((0.5, 1.5)),
                        iaa.Affine(
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                            rotate=(-15, 15),
                            shear=(-5, 5)
                        ),
                        iaa.Resize({"height": self.IMG_H, "width": self.IMG_W}),
                    ])

                    segmap = SegmentationMapsOnImage(mask, shape=image.shape)
                    image, segmap = seq(image=image, segmentation_maps=segmap)
                    mask = segmap.get_arr()
                else:
                    seq = iaa.Sequential([
                        iaa.PadToAspectRatio(self.IMG_W/self.IMG_H, position='center'),
                        iaa.Resize({"height": self.IMG_H, "width": self.IMG_W}),
                    ])

                    segmap = SegmentationMapsOnImage(mask, shape=image.shape)
                    image, segmap = seq(image=image, segmentation_maps=segmap)
                    mask = segmap.get_arr()
                
                mask = mask / 255.
                if verbose:
                    print(image)
                    print(np.unique(mask))
                    plt.imshow(image)
                    plt.imshow(mask, alpha=0.5)
                    plt.show()
                X_batch.append(image)
                y_batch.append(mask.astype(float))
                    
            yield np.array(X_batch), np.array(y_batch)

    def defineModel(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        
        inputs = Input(self.IMG_SHAPE)
        x = preprocess_input(inputs)
        unet = Unet(
            'resnet34', classes=1, activation='sigmoid',
             encoder_weights='imagenet', encoder_freeze=True)(x)
        self.model = Model(inputs, unet)
        print(self.model.summary())


    def train(self, train_path, val_path=None):
        if self.model is None:
            self.defineModel() 
            # self.model = tf.keras.models.load_model(self.buff_weights, compile=False)

        file_list = glob.glob(train_path + '/*.jpg')
        total_size = len(file_list)
        val_size = total_size // 10
        train_size = total_size - val_size

        from tensorflow.keras import callbacks
        CALLBACKS_LIST = [
            callbacks.ModelCheckpoint(
                filepath=self.buff_weights,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.33,
                patience=50,
                min_lr=1e-15,
                verbose=1,
            )
        ]
        from boundary_loss import surface_loss_keras
        DiceLoss(beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=1e-05)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-4),
            loss=surface_loss_keras,
        )
        
        self.model.load_weights(self.buff_weights)

        self.model.fit(
            self.generator(train_path, aug=True),
            steps_per_epoch=(train_size+self.BATCH_SIZE-1)//self.BATCH_SIZE,
            validation_data=self.generator(train_path, aug=False, val=True),
            validation_steps=(val_size+self.BATCH_SIZE-1)//self.BATCH_SIZE,
            callbacks=CALLBACKS_LIST, 
            epochs=1024,
            verbose=1)

        self.model.load_weights(self.buff_weights)
    
    def detect_in_folder(self, PATH, verbose=False):
        if self.model is None:
            # self.model = tf.keras.models.load_model(self.buff_weights, compile=False)
            # self.model = tf.keras.models.load_model('segmentation_weights_1.h5', compile=False)
            self.model = tf.keras.models.load_model('weights_num.hdf5', compile=False)
            print(self.model.summary())
        for filename in os.listdir(PATH):
            image = cv2.imread(PATH + '\\' + filename, flags=cv2.IMREAD_IGNORE_ORIENTATION+cv2.IMREAD_COLOR)
            image = np.flip(image, axis=-1)
            # self.detect(image, verbose)
            self.silly_detect(image, verbose)
            
    def detect(self, img, verbose=False):
        import tensorflow_addons as tfa
        if self.model is None:
            # self.model = tf.keras.models.load_model(self.buff_weights, compile=False)
            weights_path = 'C:\\projects\\python\\nomeroff-net\\NomeroffNet\\weights_num.hdf5'
            self.model = tf.keras.models.load_model(weights_path, compile=False)
        plates = []
        # img = np.flip(img, axis=-1)
        for sz in [64, 128, 256, 512, 768, 1024, 1280]:
            seq = iaa.Sequential([
                        iaa.PadToAspectRatio(1, position='center'),
                        iaa.Resize({"height": sz, "width": sz}),
                    ])
            small_img = seq(image=img)
            mask = self.model(np.expand_dims((small_img - 127.5)/128., axis=0))
            mask = (mask.numpy()[0,:,:,0] > 0.5).astype(int)
            
            if verbose:
                plt.imshow(small_img)
                plt.imshow(mask, alpha=0.5)
                plt.show()
            
            mask = tfa.image.connected_components(mask)
            classes = np.unique(mask)
            for c in classes:
                if c == 0: continue
                try:
                    from scipy.spatial import ConvexHull
                    points = np.argwhere(mask == c)
                    convex_hull_idx = ConvexHull(points).vertices
                    points = points[convex_hull_idx]
                    
                    # if verbose:
                    #     plt.imshow(small_img)
                    #     plt.scatter(points[:, 1], points[:, 0])
                    #     plt.show()

                    while points.shape[0] > 4:
                        edges = points - np.roll(points, 1, axis=0)
                        edges = edges / np.linalg.norm(edges, axis=1)[:,None]
                        angles = np.abs(np.dot(edges, np.roll(edges, -1, axis=0).T).diagonal())
                        retired = np.argmax(angles)
                        points = np.delete(points, retired, 0)


                    corners = points
                    orig_h, orig_w, _ = img.shape
                    orig_h, orig_w = orig_h, orig_w
                    if orig_w < orig_h:
                        corners = corners * orig_h / sz
                        corners[:, 1] -= (orig_h - orig_w) / 2
                    else:
                        corners = corners * orig_w / sz
                        corners[:, 0] -= (orig_w - orig_h) / 2
                    
                    # if verbose:
                    #     plt.imshow(img)
                    #     plt.scatter(corners[:, 1], corners[:, 0])
                    #     plt.show()

                    corners = np.hstack((corners[:,1].reshape(-1, 1), corners[:,0].reshape(-1, 1)))
                    plate = self.cropFromImage(img, corners)
                    plates.append(plate)

                    # if verbose:
                    #     plt.imshow(plate)
                    #     plt.show()
                    # plt.imshow(small_img)
                    # plt.imshow(mask, alpha=0.5)
                    # plt.scatter(points[:, 1], points[:, 0], s=20, marker = 'X')
                    # plt.scatter(points[corners, 1], points[corners, 0], s=50, marker = 'o')
                    # for i, _ in enumerate(points):
                    #     plt.gca().annotate(i, (points[i, 1], points[i, 0]))
                    # plt.show()
                except:
                    pass

        return plates
    def silly_detect(self, img, verbose=False):
        import tensorflow_addons as tfa
        if self.model is None:
            # self.model = tf.keras.models.load_model(self.buff_weights, compile=False)
            weights_path = 'C:\\projects\\python\\nomeroff-net\\NomeroffNet\\weights_num.hdf5'
            self.model = tf.keras.models.load_model(weights_path, compile=False)
        plates = []
        for sz in [256, 512, 1024]:
            seq = iaa.Sequential([
                        iaa.PadToAspectRatio(1, position='center'),
                        iaa.Resize({"height": sz, "width": sz}),
                    ])
            small_img = seq(image=img)
            mask = self.model(np.expand_dims((small_img - 127.5)/128., axis=0))
            mask = (mask.numpy()[0,:,:,0] > 0.5).astype(int)
            
            if verbose:
                plt.imshow(small_img)
                plt.imshow(mask, alpha=0.5)
                plt.show()
            
            mask = tfa.image.connected_components(mask)
            classes = np.unique(mask)
            for c in classes:
                if c == 0: continue
                from scipy.spatial import ConvexHull
                try:
                    points = np.argwhere(mask == c)
                    x_min = np.min(points[:, 1])
                    x_max = np.max(points[:, 1])
                    y_min = np.min(points[:, 0])
                    y_max = np.max(points[:, 0])

                    orig_h, orig_w, _ = img.shape

                    if orig_w < orig_h:
                        x_min *= orig_h / sz
                        x_max *= orig_h / sz
                        y_min *= orig_h / sz
                        y_max *= orig_h / sz
                        x_min -= (orig_h - orig_w) / 2
                        x_max -= (orig_h - orig_w) / 2
                    else:
                        x_min *= orig_w / sz
                        x_max *= orig_w / sz
                        y_min *= orig_w / sz
                        y_max *= orig_w / sz
                        y_min -= (orig_w - orig_h) / 2
                        y_max -= (orig_w - orig_h) / 2
                    x_min = int(x_min)
                    x_max = int(x_max)
                    y_min = int(y_min)
                    y_max = int(y_max)
                    plate = img[y_min:y_max, x_min:x_max]
                    
                    plates.append(plate)

                    if verbose:
                        plt.imshow(plate)
                        plt.show()
                except:
                    pass
        return plates

    def cropFromImage(self, img, points):
        inputs = np.float32(sorted(points, key=lambda x: x[0]))
        if inputs[0, 1] > inputs[1, 1]:
            inputs[[0,1]] = inputs[[1,0]]
        if inputs[2, 1] < inputs[3, 1]:
            inputs[[2,3]] = inputs[[3,2]]
        
        maxHeight = 50
        maxWidth = 200
        outputs = np.float32([[0, 0],
                [0, maxHeight - 1],
                [maxWidth - 1, maxHeight - 1],
                [maxWidth - 1, 0]])
                
        outputs = np.float32(sorted(outputs,key=lambda x: x[0]))
        if outputs[0, 1] > outputs[1, 1]:
            outputs[[0,1]] = outputs[[1,0]]
        if outputs[2, 1] < outputs[3, 1]:
            outputs[[2,3]] = outputs[[3,2]]

        M = cv2.getPerspectiveTransform(inputs,outputs)
        return cv2.warpPerspective(img, M, (200,50))
    
    def npToJson(self, path):
        json_dict = {}
        for filename in os.listdir(path):
            if filename.endswith(".npy"):
                regions = np.load(path + "\\" + filename)
                imgname = filename[:-4] + '.png'
                
                regions_list = []
                for region in regions:
                    regions_list.append({"shape_attributes": {"all_points_x": region[:, 0].tolist(), "all_points_y": region[:, 1].tolist()}})                                    
                desc = {"filename": imgname, "regions": regions_list}
                json_dict[imgname] = desc
        json_dict = {"_via_img_metadata": json_dict}
        print(json_dict)
        with open(path + "\\" + self.JSON_FILENAME, "w") as write_file:
            json.dump(json_dict, write_file)
detector = SegmentationDetector()
import matplotlib.image as mpimg

img = mpimg.imread('C:\\projects\\python\\nomeroff-net\\NomeroffNet\\test1.jpg')

detector.detect(img, verbose=True)

# TRAIN_PATH = 'C:\projects\python\\nomeroff-net-master\datasets\split'
# import sys
# TEST_PATH = 'C:\projects\python\\nomeroff-net-master\datasets\mine\\0000'
# if sys.argv[1] == "train":
#     detector.train(TRAIN_PATH)
# elif sys.argv[1] == "test":
#     detector.detect_in_folder("C:\\projects\\python\\nomeroff-net-master\\datasets\\mistakes\\img4-full", verbose=True)
# elif sys.argv[1] == "generate":
#     for _ in detector.generator(TRAIN_PATH, aug=False, verbose=True): pass
# elif sys.argv[1] == "test-image":
#     filename = 'C:\projects\python\\nomeroff-net-master\datasets\mine\\0000\B 067 AE 54.jpg'
#     image = cv2.imread(filename, flags=cv2.IMREAD_IGNORE_ORIENTATION+cv2.IMREAD_COLOR)
#     print(image.shape)
#     detector.detect(image)
    