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
from segmentation_models import Unet
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import Model, Input
from tensorflow.keras.applications.resnet import preprocess_input
import tensorflow.keras.backend as K
from scipy.spatial import ConvexHull

class CornersSegmentationDetector:
    def __init__(self, DIR='./weights/', **args):
        self.IMG_W = 4*32
        self.IMG_H = 4*32
        self.IMG_SHAPE = (self.IMG_W, self.IMG_H, 3)
        self.BATCH_SIZE = 32
        self.JSON_FILENAME = 'via_region_data.json'
        self.model = None
        self.buff_weights = DIR + '/corners.h5'
        self.magic_dict = dict(zip(np.linspace(0.5, 1.5, 10), np.zeros(10)))
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

                    if len(x_points) != 4: continue

                    # plt.imshow(image)
                    # plt.scatter(x_points, y_points, s=50, marker='X', color='red')
                    # plt.show()

                    x_min, x_max = np.min(x_points), np.max(x_points)
                    y_min, y_max = np.min(y_points), np.max(y_points)
                    
                    numberplate_w = x_max - x_min
                    numberplate_h = y_max - y_min

                    # len_v = math.sqrt((x_points[0]-x_points[1])**2+(y_points[0]-y_points[1])**2)
                    # len_h = math.sqrt((x_points[1]-x_points[2])**2+(y_points[1]-y_points[2])**2)

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
                        
                        plt.imshow(image_aug)
                        # plt.scatter(x_points_aug, y_points_aug, s=50, marker='X', color='red')
                        plt.show()
                        plt.imshow(cropped)
                        # plt.scatter(x_points_aug, y_points_aug, s=50, marker='X', color='red')
                        plt.show()
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
                        
                    x_points_aug = np.array([kp.x for kp in kps_aug]).astype(int)
                    y_points_aug = np.array([kp.y for kp in kps_aug]).astype(int)
                    
                    mask = np.zeros((self.IMG_W, self.IMG_W, 1))
                    layer = -1
                    for x, y in zip(x_points_aug, y_points_aug):
                        layer += 1
                        if y < 0 or y >= self.IMG_W: continue
                        if x < 0 or x >= self.IMG_H: continue
                       
                        mask[y, x, 0] = 1

                    # soft_mask = np.zeros((self.IMG_W, self.IMG_W))
                    # for x, y in zip(x_points_aug, y_points_aug):
                    #     for i in range(y-2, y+3):
                    #         for j in range(x-2, x+3):
                    #             if i < 0 or i >= self.IMG_W: continue
                    #             if j < 0 or j >= self.IMG_H: continue
                                
                    #             soft_mask[i, j] = 1
                    # plt.imshow(soft_mask)
                    # plt.scatter(x_points_aug, y_points_aug, s=50, marker='X', color='red')
                    # plt.show()
                    # xy_points = np.hstack((x_points_aug, y_points_aug)).astype(float)/self.IMG_W
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
                    y_batch.append(mask)
                    
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
        #self.model = Unet(input_shape=self.IMG_SHAPE)
        inputs = Input(self.IMG_SHAPE)
        x = preprocess_input(inputs)
        unet = Unet('resnet34', classes=1, activation=None, encoder_weights='imagenet', input_shape=self.IMG_SHAPE)(x)
        self.model = Model(inputs, unet)
        print(self.model.summary())


    def train(self, train_path, val_path):
        if self.model is None:
            #self.defineModel() 
            self.model = tf.keras.models.load_model(self.buff_weights, compile=False)

        train_size = len(os.listdir(train_path))
        val_size = len(os.listdir(val_path))

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
                patience=20,
                min_lr=1e-7,
                verbose=1,
            )
        ]

        from tensorflow.keras.layers import MaxPooling2D


        def custom_loss(y_true, y_pred):
            y_pred = K.sigmoid(y_pred)
            loss2 = K.mean(K.square((1-tf.nn.max_pool(y_true, ksize = [7,7], strides = [1,1], padding = 'SAME'))*y_pred))

            where = tf.not_equal(y_true, 0) 
            a = tf.boolean_mask(y_pred,where)
            loss1 = tf.cond(tf.equal(tf.size(a), 0), lambda : tf.constant(0.0), lambda: K.mean(K.square(1-a)))

            #alpha = 0.1
            #return alpha * loss1 + (1 - alpha) * loss2
            return loss1 + 1000*loss2

        def f(y_true, y_pred):
            y_pred = tf.sigmoid(y_pred)
            where = tf.not_equal(y_true, 0) 
            a = tf.boolean_mask(y_pred,where)
            loss1 = tf.cond(tf.equal(tf.size(a), 0), lambda : tf.constant(0.0), lambda: K.mean(K.square(1-a)))
            return loss1

        def t(y_true, y_pred):
            y_pred = tf.sigmoid(y_pred)
            loss2 = tf.reduce_mean(tf.math.multiply(1-tf.nn.max_pool(y_true, ksize = [7,7], strides = [1,1], padding = 'SAME'), tf.round(y_pred)))
            return loss2

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.01),
            loss=custom_loss,
            metrics = [f,t]
        )
        
        # self.model.load_weights(self.buff_weights)

        self.model.fit(
            self.generator(train_path),
            steps_per_epoch=(train_size+self.BATCH_SIZE-1)//self.BATCH_SIZE,
            validation_data=self.generator(val_path, aug=False),
            validation_steps=(val_size+self.BATCH_SIZE-1)//self.BATCH_SIZE,
            callbacks=CALLBACKS_LIST, 
            epochs=1024,
            verbose=1)

        self.model.load_weights(self.buff_weights)
    
    def detect_in_folder(self, PATH):
        if self.model is None:
            self.model = tf.keras.models.load_model(self.buff_weights, compile=False)
        for batch in self.generator(PATH, aug=False):
            points_preds = tf.nn.relu(self.model(batch[0]))
            print(points_preds)
            maxpool = tf.nn.max_pool(points_preds, ksize = [5,5], strides = [1,1], padding = 'SAME')
            faces = tf.logical_and(tf.equal(maxpool, points_preds), tf.greater(points_preds, 0))
            centers = tf.where(faces)
            print(centers)
            an = tf.gather_nd(maxpool, centers)
            print(an)

            k = 0
            for i in range(len(batch[0])):
                image = batch[0][i]
                cnt = 0
                while k < centers.shape[0] and centers[k, 0] == i:
                    cnt += 1
                    k += 1
                ang = centers[k-cnt:k]
                print(ang)
                points = points_preds[i].numpy().reshape((self.IMG_W, self.IMG_H))  
                print(points[ang[:,1], ang[:,2]])
                plt.imshow(image, interpolation='none')
                # idxs = np.dstack(np.unravel_index(np.argsort(points, axis=None), points.shape))
                # print(idxs[0, -8:])
                plt.scatter(ang[:,2], ang[:,1], s=50, marker='X', color='red')
                plt.imshow(points_preds[i], 'jet', interpolation='none', alpha=0.5)
                #plt.imshow(mask_pred[:, :, 1], 'gray', interpolation='none', alpha=0.1)
                #plt.imshow(mask_pred[:, :, 2], 'gray', interpolation='none', alpha=0.1)
                #plt.imshow(mask_pred[:, :, 3], 'gray', interpolation='none', alpha=0.1)
                plt.show()
    def detect(self, img, zones):
        if self.model is None:
            self.model = tf.keras.models.load_model(self.buff_weights, compile=False)
        
        plates = []
        for x0, y0, x1, y1, p, _ in zones:
            try:
                cur_plates = []
                x_c = (x0 + x1) / 2
                y_c = (y0 + y1) / 2

                h = x1 - x0
                w = y1 - y0
                

                for nh, nw in [[32, 128], [None, None]]:
                    orig_points = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
                    plate = self.cropFromImage(img, orig_points, nh, nw)
                    cur_plates.append(plate)

                cropped_batch = []
                for magic in [0.8, 0.9, 1]:
                    span = magic * max(h, w)
                    up_edge = max(int(y_c - span), 0) 
                    left_edge = max(int(x_c - span), 0) 
                    down_edge = min(int(y_c + span), img.shape[0]) 
                    right_edge = min(int(x_c + span), img.shape[1])

                    cropped = img[up_edge:down_edge,left_edge:right_edge]
                    
                    seq = iaa.Sequential([
                        iaa.PadToAspectRatio(self.IMG_W/self.IMG_H, position='center'),
                        iaa.Resize({"height": self.IMG_H, "width": self.IMG_W}),
                    ])
                    cropped = seq(image=cropped)
                    cropped_batch.append(cropped)
                
                # mask = tf.nn.relu(self.model(np.expand_dims(cropped, axis=0)))
                mask_batch = tf.nn.relu(self.model(np.array(cropped_batch)))
                for i, magic in enumerate([0.8, 0.9, 1]):
                    span = magic * max(h, w)
                    up_edge = max(int(y_c - span), 0)
                    left_edge = max(int(x_c - span), 0)
                    down_edge = min(int(y_c + span), img.shape[0])
                    right_edge = min(int(x_c + span), img.shape[1])

                    mask = np.expand_dims(mask_batch[i], axis=0)
                    cropped = cropped_batch[i]
                    # plt.imshow(cropped)
                    # plt.imshow(mask[0,:,:,0], alpha=0.5)
                    # plt.show()
                    maxpool = tf.nn.max_pool(mask, ksize = [5,5], strides = [1,1], padding = 'SAME')
                    corners = tf.logical_and(tf.equal(maxpool, mask), tf.greater(mask, 0))
                    points = tf.where(corners)
                    # print(points)
                    if points.shape[0] > 0:
                        points = points.numpy()[:, 1:3].astype(float)
                        # print(points)
                        # plt.imshow(cropped)
                        # plt.scatter(points[:,1], points[:,0], s=50, marker='X')
                        # plt.show()
                        orig_w = right_edge - left_edge
                        orig_h = down_edge - up_edge
                        
                        if orig_w < orig_h:
                            points = points * orig_h / self.IMG_H
                            points[:, 1] -= (orig_h - orig_w) / 2
                        else:
                            points = points * orig_w / self.IMG_W
                            points[:, 0] -= (orig_w - orig_h) / 2
                        points[:, 1] += left_edge
                        points[:, 0] += up_edge

                        # plt.imshow(img)
                        # plt.scatter(points[:,1], points[:,0], s=50, marker='X')
                        # plt.show()

                        points = np.hstack((points[:,1].reshape(-1, 1), points[:,0].reshape(-1, 1)))

                        # plt.imshow(img)
                        # plt.scatter(points[:,0], points[:,1], s=50, marker='o')

                        if points.shape[0] <= 3:
                            # plt.show()
                            continue

                        convex_hull_idx = ConvexHull(points).vertices
                        points = points[convex_hull_idx]
                        # plt.scatter(points[:,0], points[:,1], s=50, marker='.')

                        if len(convex_hull_idx) <= 3:
                            # plt.show()
                            continue

                        while points.shape[0] > 4:
                            edges = points - np.roll(points, 1, axis=0)
                            edges = edges / np.linalg.norm(edges, axis=1)[:,None]
                            angles = np.abs(np.dot(edges, np.roll(edges, -1, axis=0).T).diagonal())
                            retired = np.argmax(angles)
                            points = np.delete(points, retired, 0)

                        # plt.scatter(points[:,0], points[:,1], s=50, marker='x')
                        # plt.show()
                        for nh, nw in [[32, 128], [None, None]]:
                            plate = self.cropFromImage(img, points, nh, nw)
                            cur_plates.append(plate)
                            # plt.imshow(plate)
                            # plt.show()
                        plates.append(cur_plates)
            except Exception as e:
                pass

        return plates

    def cropFromImage(self, img, points, maxHeight=None, maxWidth=None):
        inputs = np.float32(sorted(points, key=lambda x: x[0]))
        if inputs[0, 1] > inputs[1, 1]:
            inputs[[0,1]] = inputs[[1,0]]
        if inputs[2, 1] < inputs[3, 1]:
            inputs[[2,3]] = inputs[[3,2]]
        
        if maxHeight == None:
            maxHeight = int(np.round(np.sqrt((inputs[1, 1] - inputs[0, 1])**2 + (inputs[1, 0] - inputs[0, 0])**2)))
        if maxWidth == None:
            maxWidth = int(np.round(np.sqrt((inputs[1, 1] - inputs[2, 1])**2 + (inputs[1, 0] - inputs[2, 0])**2)))
        

        outputs = np.float32([[0, 0],
                [0, maxHeight],
                [maxWidth, maxHeight],
                [maxWidth, 0]])
                
        outputs = np.float32(sorted(outputs,key=lambda x: x[0]))
        if outputs[0, 1] > outputs[1, 1]:
            outputs[[0,1]] = outputs[[1,0]]
        if outputs[2, 1] < outputs[3, 1]:
            outputs[[2,3]] = outputs[[3,2]]

        M = cv2.getPerspectiveTransform(inputs,outputs)
        return cv2.warpPerspective(img, M, (maxWidth,maxHeight))
    
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

    def save_model(self, path):
        if self.model is None:
            self.model = tf.keras.models.load_model(self.buff_weights, compile=False)

        self.model.save(path, save_format='tf')    
# detector = CornersSegmentationDetector()
# TRAIN_PATH = 'C:\\projects\\python\\nomeroff-net-master\\datasets\\autoriaNumberplateDataset-2020-12-17\\train\\'
# # VAL_PATH = 'C:\\projects\\python\\nomeroff-net-master\\datasets\\autoriaNumberplateDataset-2020-12-17\\val\\'
# PATH = "C:\projects\python\\nomeroff-net-master\datasets\detector-mistakes\img3"
# detector.npToJson(PATH)
# for _ in detector.generator(TRAIN_PATH, aug=True): pass

# # # detector.train(TRAIN_PATH, VAL_PATH)
# detector.detect_in_folder(PATH)