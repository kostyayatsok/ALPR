import tensorflow as tf
import random
import numpy as np
import os
import json
from cv2 import cv2
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
import math
from imgaug.augmentables import Keypoint, KeypointsOnImage

class ResNetDetector:
    def __init__(self, **args):
        self.IMG_W = 112
        self.IMG_H = 112
        self.IMG_SHAPE = (self.IMG_W, self.IMG_H, 3)
        self.BATCH_SIZE = 32
        self.JSON_FILENAME = 'via_region_data.json'
        self.model = None

    def defineModel(self):
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, AveragePooling2D, Lambda, GlobalMaxPool2D, DepthwiseConv2D
        from tensorflow.keras.applications import ResNet50, EfficientNetB0
        from tensorflow.keras.applications.resnet50 import preprocess_input
        '''
        https://www.tensorflow.org/tutorials/images/transfer_learning
        '''

        # base_model = ResNet50(include_top=False, input_shape=self.IMG_SHAPE)
        # base_model.trainable = False

        # model = tf.keras.Sequential()
        # model.add(Input(shape=self.IMG_SHAPE))
        # model.add(Lambda(preprocess_input))
        # model.add(base_model)
        # model.add(GlobalAveragePooling2D())
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dense(256, activation='relu'))
        # model.add(Dense(8))

        # self.model = model

        base_model = ResNet50(include_top=False, weights='imagenet',input_shape=self.IMG_SHAPE)
        global_average_layer = DepthwiseConv2D((4,4))
        prediction_layer = Dense(8)
        
        inputs = Input(shape=self.IMG_SHAPE)
        # x = data_augmentation(inputs)
        x = preprocess_input(inputs)
        x = base_model(x)
        x = global_average_layer(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        outputs = prediction_layer(x)
        self.model = Model(inputs, outputs)

        print(self.model.summary())


    def loadLandmarks(self, dir_path):
        from tqdm import tqdm
        json_filepath = os.path.join(dir_path, self.JSON_FILENAME)
        json_file = open(json_filepath, 'r')
        landmarks_data = json.load(json_file)["_via_img_metadata"]
        landmarks_data = {v['filename'] : v for k, v in landmarks_data.items()}

        files = []
        for filename, landmarks in tqdm(landmarks_data.items()): 
            img_filepath = os.path.join(dir_path, filename)
            image = cv2.imread(img_filepath, flags=cv2.IMREAD_IGNORE_ORIENTATION+cv2.IMREAD_COLOR )
            files.append((filename, image))
            for region in landmarks["regions"]:
                x_points = np.array(region["shape_attributes"]["all_points_x"])
                y_points = np.array(region["shape_attributes"]["all_points_y"])
                
                if len(x_points) != 4: continue

                srt = np.array(sorted(zip(x_points, y_points)))
                if srt[0, 1] > srt[1, 1]:
                    srt[[0,1]] = srt[[1,0]]
                if srt[2, 1] < srt[3, 1]:
                    srt[[2,3]] = srt[[3,2]]
                region["shape_attributes"]["all_points_x"] = srt[:, 0]
                region["shape_attributes"]["all_points_y"] = srt[:, 1]
                
        return landmarks_data, files

    def generator(self, dir_path, aug=True):        
        landmarks_data, file_list = self.loadLandmarks(dir_path)

        #print('generator')
        #file_list = os.listdir(dir_path)
        #file_list.remove(self.JSON_FILENAME)

        file_idx = 0
        while True:
            X_batch, y_batch = [], []
            for _ in range(self.BATCH_SIZE):
                if file_idx == len(file_list):
                    file_idx = 0
                    random.shuffle(file_list)
                image = file_list[file_idx][1]
                filename = file_list[file_idx][0]
                file_idx += 1

                for region in landmarks_data[filename]["regions"]:
                    x_points = np.array(region["shape_attributes"]["all_points_x"])
                    y_points = np.array(region["shape_attributes"]["all_points_y"])

                    # print(x_points)
                    if len(x_points) != 4: continue
                    # print('?')
                    # plt.imshow(image)
                    # plt.scatter(x_points, y_points, s=50, marker='X', color='red')
                    # plt.show()

                    x_min, x_max = np.min(x_points), np.max(x_points)
                    y_min, y_max = np.min(y_points), np.max(y_points)
                    
                    numberplate_w = x_max - x_min
                    numberplate_h = y_max - y_min

                    len_v = math.sqrt((x_points[0]-x_points[1])**2+(y_points[0]-y_points[1])**2)
                    len_h = math.sqrt((x_points[1]-x_points[2])**2+(y_points[1]-y_points[2])**2)

                    if numberplate_w < numberplate_h:
                        continue

                    #print(len_v, len_h)

                    x_center = int((x_max + x_min) / 2)
                    y_center = int((y_max + y_min) / 2)

                    span = int(max(numberplate_w, numberplate_h))

                    left_edge, right_edge = int(x_center - span), int(x_center + span)
                    up_edge, down_edge = int(y_center - span), int(y_center + span)
                
                    left_edge, right_edge = max(0, left_edge), min(right_edge, image.shape[1])
                    up_edge, down_edge = max(0, up_edge), min(down_edge, image.shape[0])
                

                    cropped = image[up_edge:down_edge, left_edge:right_edge]
    
                    x_points_cropped = x_points - left_edge
                    y_points_cropped = y_points - up_edge
                    
                    # plt.imshow(cropped)
                    # plt.scatter(x_points_cropped, y_points_cropped, s=50, marker='X', color='red')
                    # plt.show()

                    if aug:
                        # https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
                        kps = KeypointsOnImage(
                            [Keypoint(x=x, y=y) for x, y in zip(x_points_cropped, y_points_cropped)],
                            shape=cropped.shape)

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

                        image_aug, kps_aug = seq(image=cropped, keypoints=kps)
                        
                        # plt.imshow(image_aug)
                        # plt.scatter(x_points_aug, y_points_aug, s=50, marker='X', color='red')
                        # plt.show()
                    else:
                        kps = KeypointsOnImage(
                            [Keypoint(x=x, y=y) for x, y in zip(x_points_cropped, y_points_cropped)],
                            shape=cropped.shape)

                        # https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html
                        seq = iaa.Sequential([
                            iaa.PadToAspectRatio(self.IMG_W/self.IMG_H),
                            iaa.Resize({"height": self.IMG_H, "width": self.IMG_W}),
                        ])

                        image_aug, kps_aug = seq(image=cropped, keypoints=kps)
                        
                    x_points_aug = np.array([kp.x for kp in kps_aug])
                    y_points_aug = np.array([kp.y for kp in kps_aug])
                    xy_points = np.hstack((x_points_aug, y_points_aug)).astype(float)/self.IMG_W
                    '''
                    print(filename)
                    plt.imshow(image_aug)
                    plt.scatter(x_points_aug[0], y_points_aug[0], s=50, marker='X', color='red')
                    plt.scatter(x_points_aug[1], y_points_aug[1], s=50, marker='X', color='blue')
                    plt.scatter(x_points_aug[2], y_points_aug[2], s=50, marker='X', color='green')
                    plt.scatter(x_points_aug[3], y_points_aug[3], s=50, marker='X', color='black')
                    plt.show()
                    '''
                    X_batch.append(image_aug)
                    y_batch.append(xy_points)
                    
            yield np.array(X_batch), np.array(y_batch)


    def train(self, train_path, val_path):
        if self.model is None:
            self.defineModel() 
            #self.model = tf.keras.models.load_model('./buff_weights.h5')

        train_size = len(os.listdir(train_path))
        val_size = len(os.listdir(val_path))

        from tensorflow.keras import callbacks
        CALLBACKS_LIST = [
            callbacks.ModelCheckpoint(
                filepath='./buff_weights.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.33,
                patience=50,
                min_lr=1e-7,
                verbose=1,
            )
        ]

        def acc(tr, pr):
            elements_equal_to_value = tf.less(tf.abs(tr-pr), 0.01)
            as_ints = tf.cast(elements_equal_to_value, tf.int32)
            return tf.reduce_mean(as_ints)

        self.model.compile(
            #optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov = True, clipnorm=0.3),
            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            metrics=[tf.keras.metrics.RootMeanSquaredError(), acc],
            loss='mae')
        self.model.fit(
            self.generator(train_path),
            steps_per_epoch=(train_size+self.BATCH_SIZE-1)//self.BATCH_SIZE,
            validation_data=self.generator(val_path, aug=False),
            validation_steps=(val_size+self.BATCH_SIZE-1)//self.BATCH_SIZE,
            callbacks=CALLBACKS_LIST, 
            epochs=1024,
            verbose=1)

        self.model.load_weights('./buff_weights.h5')
    
    def detect(self, TRAIN_PATH):
        if self.model is None:
            self.model = tf.keras.models.load_model('./buff_weights.h5')
        for batch in self.generator(TRAIN_PATH):
            print(batch[0].shape, batch[1].shape)
            points_preds = self.model.predict(batch[0])
            print(points_preds.shape)
            for image, points_true, points_pred in zip(batch[0], batch[1], points_preds):
                print(image.shape, points_true.shape)
                plt.imshow(image)
                plt.scatter(points_true[0] * self.IMG_W, points_true[4] * self.IMG_W, s=50, marker='<', color='green')
                plt.scatter(points_true[1] * self.IMG_W, points_true[5] * self.IMG_W, s=50, marker='v', color='green')
                plt.scatter(points_true[2] * self.IMG_W, points_true[6] * self.IMG_W, s=50, marker='>', color='green')
                plt.scatter(points_true[3] * self.IMG_W, points_true[7] * self.IMG_W, s=50, marker='^', color='green')
                plt.scatter(points_pred[0] * self.IMG_W, points_pred[4] * self.IMG_W, s=50, marker='<', color='red')
                plt.scatter(points_pred[1] * self.IMG_W, points_pred[5] * self.IMG_W, s=50, marker='v', color='red')
                plt.scatter(points_pred[2] * self.IMG_W, points_pred[6] * self.IMG_W, s=50, marker='>', color='red')
                plt.scatter(points_pred[3] * self.IMG_W, points_pred[7] * self.IMG_W, s=50, marker='^', color='red')
                plt.show()

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

detector = ResNetDetector()
TRAIN_PATH = 'C:\\projects\\python\\nomeroff-net-master\\datasets\\autoriaNumberplateDataset-2020-12-17\\train\\'
VAL_PATH = 'C:\\projects\\python\\nomeroff-net-master\\datasets\\autoriaNumberplateDataset-2020-12-17\\val\\'

#for _ in detector.generator(TRAIN_PATH, aug=True): pass

detector.train(TRAIN_PATH, VAL_PATH)

detector.detect(TRAIN_PATH)
