# import labaris
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

import os
from os.path import join
import json
import numpy as np
import sys

from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, Dropout
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.layers import Reshape, Lambda, LeakyReLU
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import PReLU, ReLU,LeakyReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers

from collections import Counter

from tensorflow.keras.layers import GRU

from TextImageGenerator import TextImageGenerator

import time

from tensorflow.keras import backend as K

def MyDecay():
    return regularizers.l1_l2(1e-9, 1e-8)

def MyActivation():
    return Activation('relu')

'''
def vgg_block(x, filters, layers):
    inval = x = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l1(weight_decay))(x)
    x = MyActivation()(x)
    for _ in range(layers):
        x = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l1(weight_decay))(x)
        x = DepthwiseConv2D((3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l1(weight_decay))(x)
        x = BatchNormalization()(x)
        x = MyActivation()(x)
    return x +inval
'''
def vgg_block(x, filters, layers, stride):
    inpt = x = tf.keras.layers.Conv2D(filters*2, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=MyDecay())(x)
    for _ in range(layers):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=MyDecay())(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x) 
        x = tf.concat([x,-x], axis = -1)
        #tf.exp(-x*x)
        # #PReLU(shared_axes=[1, 2])(x)
        #x = PReLU(shared_axes=[1, 2])(x)
        
#    x = add([x,inpt])
    x = tf.concat([x,inpt], axis = -1)
    x = tf.keras.layers.MaxPool2D(pool_size = stride)(x)
    #x = BatchNormalization()(x)
    #x = PReLU(shared_axes=[1, 2])(x)

    return x

class IntervalEvaluation(Callback):
    def __init__(self, ocr_module, interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.accuracy = 0.91
        self.count = 0
        self.ocr = ocr_module

    def on_epoch_end(self, epoch, logs={}):
        self.count = self.count + 1

        try:
            net_inp = self.model.get_layer(name='{}'.format(self.model.layers[0].name)).input
            net_out = self.model.get_layer(name='{}'.format(self.model.layers[-5].name)).output
            self.ocr.MODEL = Model(inputs=net_inp, outputs=net_out)
            err_c1,succ_c1 = self.ocr.test(verbose=0) 
            err_c2,succ_c2 = self.ocr.validate(verbose=0)
            acc = logs['lfw'] = (succ_c1+succ_c2)/(err_c1+succ_c1+err_c2+succ_c2)
            if acc >= 0.9975 or self.accuracy <= acc:
                self.accuracy = acc
                print("Save, model accuracy is ", acc)
                self.ocr.MODEL.save('lwf2'+str(acc)+'.hdf5')
        except:
            print('Ошибка проверки.', sys.exc_info()[0])


class OCR(TextImageGenerator):
    @classmethod
    def get_classname(cls):
        return cls.__name__

    def __init__(self):
        # Input parameters
        self.IMG_H = 32
        self.IMG_W = 128
        self.IMG_C = 1

        # Train parameters
        self.BATCH_SIZE = 128
        self.EPOCHS = 1

        # Network parameters
        self.CONV_FILTERS = 16
        self.KERNEL_SIZE = (3, 3)
        self.POOL_SIZE = 2
        self.TIME_DENSE_SIZE = 32
        self.RNN_SIZE = 512
        self.ACTIVATION = 'relu'
        self.DOWNSAMPLE_FACROT = self.POOL_SIZE * self.POOL_SIZE

        self.INPUT_NODE = "the_input_{}:0".format(type(self).__name__)
        self.OUTPUT_NODE = "softmax_{}".format(type(self).__name__)
        
        # callbacks hyperparameters
        self.REDUCE_LRO_N_PLATEAU_PATIENCE = 50
        self.REDUCE_LRO_N_PLATEAU_FACTOR   = 0.3

    def get_counter(self, dirpath, verbose=1):
        dirname = os.path.basename(dirpath)
        ann_dirpath = join(dirpath, 'ann')
        letters = ''
        lens = []
        for filename in os.listdir(ann_dirpath):
            json_filepath = join(ann_dirpath, filename)
            description = json.load(open(json_filepath, 'r'))['description']
            description = description.replace('0','O')
            lens.append(len(description))
            letters += description
        max_plate_length = max(Counter(lens).keys())
        if verbose:
            print('Max plate length in "%s":' % dirname, max_plate_length)
        return Counter(letters), max_plate_length

    def get_alphabet(self, train_path, test_path, val_path, verbose=1):
        c_val, max_plate_length_val     = self.get_counter(val_path)
        c_train, max_plate_length_train = self.get_counter(train_path)
        c_test, max_plate_length_test   = self.get_counter(test_path)

        letters_train = set(c_train.keys())
        letters_val = set(c_val.keys())
        letters_test = set(c_test.keys())
        if verbose:
            print("Letters train ", letters_train)
            print("Letters val ", letters_val)
            print("Letters test ", letters_test)

        if max_plate_length_val == max_plate_length_train:
            if verbose:
                print('Max plate length in train, test and val do match')
        else:
            raise Exception('Max plate length in train, test and val do not match')

        if letters_train == letters_val:
            if verbose:
                print('Letters in train, val and test do match')
        else:
            raise Exception('Letters in train, val and test do not match')

        self.letters = sorted(list(letters_train))
        self.max_text_len = max_plate_length_train
        if verbose:
            print('Letters:', ' '.join(self.letters))
        return self.letters, self.max_text_len

    def explainTextGenerator(self, train_dir, letters, max_plate_length, verbose=1):
        tiger = TextImageGenerator(train_dir, self.IMG_W, self.IMG_H, 1, self.POOL_SIZE * self.POOL_SIZE, letters, max_plate_length, cname=type(self).__name__)
        tiger.build_data()

        for inp, out in tiger.next_batch():
            print('Text generator output (data which will be fed into the neutral network):')
            print('1) the_input (image)')
            if K.image_data_format() == 'channels_first':
                img = inp['the_input_{}'.format(type(self).__name__)][0, 0, :, :]
            else:
                img = inp['the_input_{}'.format(type(self).__name__)][0, :, :, 0]
            '''
            try:
                import matplotlib.pyplot as plt
                plt.imshow(img.T, cmap='gray')
                plt.show()
            except Exception as e:
                print("[WARN]", "Can not display image")
            '''
            print('2) the_labels (plate number): %s is encoded as %s' %
                  (tiger.labels_to_text(inp['the_labels_{}'.format(type(self).__name__)][0]), list(map(int, inp['the_labels_{}'.format(type(self).__name__)][0]))))
            print('3) input_length (width of image that is fed to the loss function): %d == %d / 4 - 2' %
                  (inp['input_length_{}'.format(type(self).__name__)][0], tiger.img_w))
            print('4) label_length (length of plate number): %d' % inp['label_length_{}'.format(type(self).__name__)][0])
            break

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        #y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def save(self, path, verbose=1):
        if self.MODEL:
            self.MODEL.save(path)
            if verbose:
                print("SAVED TO {}".format(path))

    def validate(self, verbose=1, random_state=0):
        if verbose:
            print("\nRUN TEST")
            start_time = time.time()

        err_c = 0
        succ_c = 0

        names_input = [self.MODEL.layers[0].name, 'the_input_'+type(self).__name__,'the_input_ru','the_input_Test', 'input_1', 'input_2']
        names_output = [self.MODEL.layers[-1].name, 'the_labels_'+type(self).__name__,'the_labels_ru','the_labels_Test']
        save_batch = self.tiger_val.batch_size
        self.tiger_val.batch_size = self.tiger_val.n
        for inp_value, _ in self.tiger_val.next_batch(random_state, input_name=self.MODEL.layers[0].name, output_name=self.MODEL.layers[-1].name):
            X_data = None
            for name in names_input:
                try:
                    bs = inp_value[name].shape[0]
                    X_data = inp_value[name]
                    break
                except:
                    continue
            
            net_out_value = self.MODEL.predict(np.array(X_data))
            pred_texts = self.decode_batch(net_out_value)
                
            for name in names_output:
                try:
                    labels = inp_value[name]
                    break
                except:
                    continue

            texts = []
            for label in labels:
                text = self.tiger_test.labels_to_text(label)
                texts.append(text)

            for i in range(bs):
                if (pred_texts[i] != texts[i]):
                    if verbose:
                        print('\nPredicted: \t\t %s\nTrue: \t\t\t %s' % (pred_texts[i], texts[i]))
                    err_c += 1
                else:
                    succ_c += 1
            break

        self.tiger_val.batch_size = save_batch
        if verbose:
            print("Test processing time: {} seconds".format(time.time() - start_time))
        print("acc: {}".format(succ_c/(err_c+succ_c)))
        return err_c,succ_c

    def test(self, verbose=1, random_state=0):
        if verbose:
            print("\nRUN TEST")
            start_time = time.time()

        err_c = 0
        succ_c = 0

        names_input = [self.MODEL.layers[0].name, 'the_input_'+type(self).__name__,'the_input_ru','the_input_Test', 'input_1', 'input_2']
        names_output = [self.MODEL.layers[-1].name, 'the_labels_'+type(self).__name__,'the_labels_ru','the_labels_Test']
        for inp_value, _ in self.tiger_test.next_batch(random_state, input_name=self.MODEL.layers[0].name, output_name=self.MODEL.layers[-1].name):
            X_data = None
            for name in names_input:
                try:
                    bs = inp_value[name].shape[0]
                    X_data = inp_value[name]
                    break
                except:
                    continue
            
            net_out_value = self.MODEL.predict(np.array(X_data))
            pred_texts = self.decode_batch(net_out_value)
                
            for name in names_output:
                try:
                    labels = inp_value[name]
                    break
                except:
                    continue

            texts = []
            for label in labels:
                text = self.tiger_test.labels_to_text(label)
                texts.append(text)

            for i in range(bs):
                if (pred_texts[i] != texts[i]):
                    if verbose:
                        print('\nPredicted: \t\t %s\nTrue: \t\t\t %s' % (pred_texts[i], texts[i]))
                    err_c += 1
                else:
                    succ_c += 1
            break
        if verbose:
            print("Test processing time: {} seconds".format(time.time() - start_time))
        print("acc: {}".format(succ_c/(err_c+succ_c)))
        return err_c,succ_c
   
    def test_pb(self, verbose=1, random_state=0):
        if verbose:
            print("\nRUN TEST")
            start_time = time.time()

        err_c = 0
        succ_c = 0
        for X_data, labels in self.tiger_test.next_batch_pb(random_state):
            tensorX = tf.convert_to_tensor(np.array(X_data).astype(np.float32))
            net_out_value = self.PB_MODEL(tensorX)[self.OUTPUT_NODE]
            pred_texts = self.decode_batch(net_out_value)
            
            texts = []
            for label in labels:
                text = self.tiger_test.labels_to_text(label)
                texts.append(text)
            
            bs = len(labels)
            for i in range(bs):
                if (pred_texts[i] != texts[i]):
                    if verbose:
                        print('\nPredicted: \t\t %s\nTrue: \t\t\t %s' % (pred_texts[i], texts[i]))
                    err_c += 1
                else:
                    succ_c += 1
            break
        if verbose:
            print("Test processing time: {} seconds".format(time.time() - start_time))
        print("acc: {}".format(succ_c/(err_c+succ_c)))

    def predict(self, imgs, return_acc=True):
        Xs = []
        for img in imgs:
            x = self.normalize(img)
            Xs.append(x)
        pred_texts = []
        probs = []
        if bool(Xs):
            if len(Xs) == 1:
                net_out_value = self.MODEL.predict_on_batch(np.array(Xs))
            else:
                net_out_value = self.MODEL(np.array(Xs), training=False)
            pred_texts, probs = self.decode_batch(net_out_value)
        if return_acc:
            return np.array(pred_texts), np.array(probs)
        return pred_texts
    
    def predict_pb(self, imgs, return_acc=False):
        Xs = []
        for img in imgs:
            x = self.normalize_pb(img)
            Xs.append(x)
        pred_texts = []
        if bool(Xs):
            tensorX = tf.convert_to_tensor(np.array(Xs).astype(np.float32))
            net_out_value = self.PB_MODEL(tensorX)[self.OUTPUT_NODE]
            #print(net_out_value)
            pred_texts = self.decode_batch(net_out_value)
        if return_acc:
            return pred_texts, net_out_value
        return pred_texts

    def load(self, path_to_model, mode="cpu", verbose = 0):
        self.MODEL = load_model(path_to_model, compile=False, custom_objects={'ctc': self.ctc_lambda_func})

        net_inp = self.MODEL.get_layer(name='{}'.format(self.MODEL.layers[0].name)).input
        net_out = self.MODEL.get_layer(name='{}'.format(self.MODEL.layers[-1].name)).output

        self.MODEL = Model(inputs=net_inp, outputs=net_out)

        if verbose:
            self.MODEL.summary()

        return self.MODEL
    
    def load_pb(self, model_dir, mode="cpu", verbose = 0):
        pb_model = tf.saved_model.load(model_dir)
        self.PB_MODEL = pb_model.signatures["serving_default"]

    def prepare_test(self, path_to_dataset, use_aug=False, verbose=1, aug_debug=False, aug_suffix = 'aug', aug_seed_num = 42):
        test_path  = path_to_dataset
        #self.letters, max_plate_length = self.get_alphabet(test_path, test_path, test_path, verbose=verbose)
        max_plate_length = 9
        self.tiger_test = TextImageGenerator(test_path, self.IMG_W, self.IMG_H, len(os.listdir(os.path.join(test_path, "img"))), self.DOWNSAMPLE_FACROT, self.letters, max_plate_length, cname=type(self).__name__)
        self.tiger_test.build_data()

    def prepare(self, path_to_dataset, use_aug=False, verbose=1, aug_debug=True, aug_suffix = 'aug', aug_seed_num = 42):
        train_path = os.path.join(path_to_dataset, "train")
        test_path  = os.path.join(path_to_dataset, "test")
        val_path   = os.path.join(path_to_dataset, "val")

        if verbose:
            print("GET ALPHABET")
        self.letters, max_plate_length = self.get_alphabet(train_path, test_path, val_path, verbose=verbose)

        #if verbose:
        #    print("\nEXPLAIN DATA TRANSFORMATIONS")
        #    self.explainTextGenerator(train_path, self.letters, max_plate_length)

        if verbose:
            print("START BUILD DATA")
        self.tiger_train = TextImageGenerator(train_path, self.IMG_W, self.IMG_H, self.BATCH_SIZE, self.DOWNSAMPLE_FACROT, self.letters, max_plate_length, cname=type(self).__name__)
        self.tiger_train.build_data(use_aug=use_aug, aug_debug=aug_debug, aug_suffix = aug_suffix, aug_seed_num = aug_seed_num)
        self.tiger_val = TextImageGenerator(val_path,  self.IMG_W, self.IMG_H, self.BATCH_SIZE, self.DOWNSAMPLE_FACROT, self.letters, max_plate_length, cname=type(self).__name__)
        self.tiger_val.build_data()

        self.tiger_test = TextImageGenerator(test_path, self.IMG_W, self.IMG_H,
         #self.BATCH_SIZE,
         len(os.listdir(os.path.join(test_path, "img"))), 
         self.DOWNSAMPLE_FACROT, self.letters, max_plate_length, cname=type(self).__name__)
        self.tiger_test.build_data()
        if verbose:
            print("DATA PREPARED")

    def set_use_aug(self, val):
        self.tiger_train.set_use_aug(val)

    def train(self, is_random=1, load_trained_model_path=None, load_last_weights=True, verbose=1, log_dir="./"):
        if verbose:
            print("\nSTART TRAINING")
        input_data = Input(shape=(128, 32, 1))
        x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=MyDecay())(input_data)
        x = MyActivation()(x)
        x = vgg_block(x, 16, 2, (2,2))
        x = vgg_block(x, 32, 2, (2,2))
        x = vgg_block(x, 48, 2, (1,2))
        x = vgg_block(x, 64, 2, (1,2))
        x = vgg_block(x, 64, 2, (1,2))
        #x = Conv2D(64, (1, 4), padding='valid', kernel_initializer='he_normal',
        #            kernel_regularizer=MyDecay())(x) #32x1x4
        x = MyActivation()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(5, (1, 1), padding='valid', kernel_initializer='he_normal',
                    kernel_regularizer=MyDecay())(x) #32x1x4

        x = BatchNormalization()(x)
        inner = Reshape(target_shape=(32,5), name='reshape')(x)
        #inner = tf.math.l2_normalize(inner, axis=-1)
        inner = Dense(self.tiger_train.get_output_size(), kernel_initializer='he_normal',
                        name='dense2', use_bias=False)(inner)
        y_pred = Activation('softmax', name='softmax_{}'.format(type(self).__name__))(inner)
        Model(inputs=input_data, outputs=y_pred).summary()

        labels = Input(name='the_labels_{}'.format(type(self).__name__), shape=[self.tiger_train.max_text_len], dtype='float32')
        input_length = Input(name='input_length_{}'.format(type(self).__name__), shape=[1], dtype='int64')
        label_length = Input(name='label_length_{}'.format(type(self).__name__), shape=[1], dtype='int64')
        
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        
        # clipnorm seems to speeds up convergence
        adam = tf.keras.optimizers.Adam(lr=0.003)
        #adam = tf.keras.optimizers.SGD(lr=0.03) #, nesterov=True, momentum = 0.5)

        if load_trained_model_path is not None:
            model = load_model(load_trained_model_path, compile=False)
        else:
            model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)


        def puer(t,p):
            return p
        model.compile(loss={'{}'.format(model.layers[-1].name): lambda y_true, y_pred: y_pred}, optimizer=adam,  metrics=[puer])

        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])
        
        # traine callbacks
        self.CALLBACKS_LIST = [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'buff_weights.h5'),
                monitor='val_puer',
                save_best_only=True,
                verbose = 1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_puer',
                factor=self.REDUCE_LRO_N_PLATEAU_FACTOR,
                patience=self.REDUCE_LRO_N_PLATEAU_PATIENCE,
                verbose = 1
            ),
            IntervalEvaluation(self)
        ]
        
        if load_last_weights:
            print("Loading last weights....")
            try:
                model.load_weights(os.path.join(log_dir, 'buff_weights.h5'))
            except:
                print("Loading failed.")
        
        model.fit_generator(generator=self.tiger_train.next_batch(is_random, input_name=model.layers[0].name, output_name=model.layers[-1].name),
                            steps_per_epoch=self.tiger_train.n//self.BATCH_SIZE,
                            epochs=self.EPOCHS,
                            callbacks=self.CALLBACKS_LIST,
                            validation_data=self.tiger_val.next_batch(is_random, input_name=model.layers[0].name, output_name=model.layers[-1].name),
                            validation_steps=self.tiger_val.n//self.BATCH_SIZE)
        # load best model
        model.load_weights(os.path.join(log_dir, 'buff_weights.h5'))
        
        net_inp = model.get_layer(name='{}'.format(model.layers[0].name)).input
        net_out = model.get_layer(name='{}'.format(model.layers[-5].name)).output
        if load_trained_model_path is not None:
            net_inp = model.get_layer(name='{}'.format(model.layers[0].name)).input
            net_out = model.get_layer(name='{}'.format(model.layers[-1].name)).output

        self.MODEL = Model(inputs=net_inp, outputs=net_out)
        return self.MODEL

    def get_acc(self, predicted, decode):
        labels = []
        for text in decode:
            labels.append(self.text_to_labels(text))
        loss = tf.keras.backend.ctc_batch_cost(
            np.array(labels),
            np.array(predicted)[:, 2:, :],
            np.array([[self.label_length] for label in labels]),
            np.array([[self.max_text_len] for label in labels])
        )
        return  1 - tf.keras.backend.eval(loss)