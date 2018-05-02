from __future__ import print_function

import os
import cv2
import numpy as np
import h5py
from  sklearn.metrics import jaccard_similarity_score
from keras.models import Model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, MaxPooling2D, Concatenate
from keras.layers import Dropout, core , Activation
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from IPython.core.debugger import set_trace
import random

## augmentation functions
def augment_data_ltrb(imgs_train):
    rotation_angles = [0,90,180,270]
    aug_img_train = np.zeros((np.shape(imgs_train)[0]*4,np.shape(imgs_train)[1],np.shape(imgs_train)[2],np.shape(imgs_train)[3]))
    #set_trace()
    img_ctr = 0
    for i in range(np.shape(imgs_train)[0]):
        img = imgs_train[i,:,:,:]
        aug_img_train[img_ctr,:,:,:] = rotate_image(img, rotation_angles[0])
        img_ctr+=1
        aug_img_train[img_ctr,:,:,:] = rotate_image(img, rotation_angles[1])
        img_ctr+=1
        aug_img_train[img_ctr,:,:,:] = rotate_image(img, rotation_angles[2])
        img_ctr+=1
        aug_img_train[img_ctr,:,:,:] = rotate_image(img, rotation_angles[3])
        img_ctr+=1
    return aug_img_train

def rotate_image( im, angle):
    # if parameter for image is string load image
    if isinstance(im, str):
        im = cv2.imread(im)
    rows,cols,_ = im.shape
    rotmat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    im_rotated = cv2.warpAffine(im,rotmat,(cols,rows))
    if np.size(np.shape(im_rotated))<3:
        im_rotated = im_rotated[:,:,np.newaxis]
    return im_rotated

def split_data(imgs_train, train_split):
    total_train = np.asarray(range(np.shape(imgs_train)[0]))
    rand_train = np.asarray(random.sample(range(np.shape(imgs_train)[0]), int(np.ceil(np.shape(imgs_train)[0]*train_split))))
    rand_val = np.setdiff1d(total_train, rand_train)
    return rand_train,rand_val

#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

# data_path = 'IZW_RGB/'
#data_path = 'D:/KID/'
n_ch = 3
bn=True
test_only=False
use_salient_images = True
results_home = 'D:/US-NET/set3c/'
salient_image_home = 'D:/US-NET/set3c'
data_path = 'D:/US-NET/set3c/'
#Image Dimensions
image_rows = 256
image_cols = 160
train_split = 0.85

smooth = np.finfo(float).eps

def jaccard_index(y_true, y_pred):
    y_true_f = y_true.flatten('F')
    y_pred_f = y_pred.flatten('F')
    intersection = sum(y_true_f*y_pred_f)
    dice = (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)
    return ((dice)/ (2 - dice))



def dice_coef_test(y_true, y_pred):
    y_true_f = y_true.flatten('F')
    y_pred_f = y_pred.flatten('F')
    intersection = sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred):
   y_true_f = K.flatten(y_true)
   y_pred_f = K.flatten(y_pred)
   intersection = K.sum(y_true_f * y_pred_f)
   return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
   return -dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((image_rows, image_cols, n_ch))

    conv1 = Conv2D(32, (3 ,3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization(axis=-1, momentum=0.99)(conv1)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis=-1, momentum=0.99)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis=-1, momentum=0.99)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1, momentum=0.99)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1, momentum=0.99)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=-1, momentum=0.99)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1, momentum=0.99)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1, momentum=0.99)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=-1, momentum=0.99)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis=-1, momentum=0.99)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1, momentum=0.99)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1, momentum=0.99)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=-1, momentum=0.99)(conv5)


    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=2, padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1, momentum=0.99)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1, momentum=0.99)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=-1, momentum=0.99)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1, momentum=0.99)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1, momentum=0.99)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=-1, momentum=0.99)(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)

    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=-1, momentum=0.99)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=-1, momentum=0.99)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=-1, momentum=0.99)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=2, padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)

    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 =  BatchNormalization(axis=-1, momentum=0.99)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization(axis=-1, momentum=0.99)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    epochs = 200
    learning_rate = 0.001
    decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

    model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()
    plot_model(model, to_file='model.png')
    return model


if __name__ == '__main__':
    datasets = ['set1','set2','set3','set4','set5']
    if n_ch==3:
        #datasets = ['set1c','set2c','set3c','set4c','set5c']
        datasets = ['set3c']
    #if bn==True & n_ch == 3:
    #    datasets = ['set1cbn','set2cbn','set3cbn','set4cbn','set5cbn']

    for iDs in range(len(datasets)):
        data_path = './'+datasets[iDs]+'/'
        if test_only==False:
            # load training data

            train_data_path = os.path.join(data_path, 'train')
            images = os.listdir(train_data_path)
            total = len(images) // 2


            imgs_train = np.ndarray((total, image_rows, image_cols, n_ch), dtype=np.uint8)
            imgs_train_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)

            i = 0
            print('-'*30)
            print("iteration "+str(iDs))
            print('Creating training images...')
            print('-'*30)
            for image_name in images:
                if "jpg" not in image_name:
                    continue
                if 'mask' in image_name:
                    continue
                image_mask_name = image_name.split('.')[0] + '_mask.jpg'
                #print(image_mask_name)

                img = cv2.imread(os.path.join(train_data_path, image_name),-1)
                #if use_salient_images == True:
                #    img = cv2.imread(os.path.join(salient_image_home, image_name),-1)
                # img = cv2.equalizeHist(img) # histogram equalized image
                img = cv2.resize(img,(image_cols, image_rows))
                img = np.reshape(img,(image_rows, image_cols,n_ch))
                img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name),cv2.IMREAD_GRAYSCALE)
                img_mask = cv2.resize(img_mask,(image_cols, image_rows))
                img_mask = np.reshape(img_mask,(image_rows, image_cols,1))
        #        plt.imshow(cv2.cvtcolor(img, cv2.color_BGR2RGB))

                img = np.array([img])
                img_mask = np.array([img_mask])

                imgs_train[i] = img
                imgs_train_mask[i] = img_mask

                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1
            print('Loading done for training data.')

            imgs_train = imgs_train.astype('float32')
            #imgs_train /= 255.
            imgs_train_mask = imgs_train_mask.astype('float32')
            #imgs_train_mask /= 255.

            # split data into train val
            rand_train,rand_val = split_data(imgs_train, train_split)
            imgs_train_sp = imgs_train[rand_train,:,:,:]
            imgs_val_sp = imgs_train[rand_val,:,:,:]
            masks_train_sp = imgs_train_mask[rand_train,:,:,:]
            masks_val_sp = imgs_train_mask[rand_val,:,:,:]

            # augment them
            imgs_train_sp_aug = augment_data_ltrb(imgs_train_sp)
            imgs_val_sp_aug = augment_data_ltrb(imgs_val_sp)
            masks_train_sp_aug =( augment_data_ltrb(masks_train_sp))
            masks_val_sp_aug = (augment_data_ltrb(masks_val_sp))
            masks_train_sp_aug/=255
            masks_val_sp_aug/=255
            masks_train_sp_aug = np.round(masks_train_sp_aug)
            masks_val_sp_aug=np.round(masks_val_sp_aug)
            # imgs_train = (imgs_train.astype('float32') - 127.5)/127.5
            # imgs_train_mask = (imgs_train_mask.astype('float32') - 127.5)/127.5
            # set_trace()



            print('-'*30)
            print('Creating and compiling model...')
            print('-'*30)
            model = get_unet()
            # use_weights = True;
            model_checkpoint = ModelCheckpoint(data_path+'/US_NET3New.hdf5', monitor='val_loss', save_best_only=True)

            print('-'*30)
            print('Fitting model...')
            print('-'*30)
            history = model.fit(imgs_train_sp_aug, masks_train_sp_aug, batch_size=8, epochs=2,verbose=1, shuffle=True,
                                validation_data=(imgs_val_sp_aug, masks_val_sp_aug),
                                callbacks=[model_checkpoint])
            '''
            history = model.fit(imgs_train_sp_aug, masks_train_sp_aug, batch_size=4, epochs=200,verbose=1, shuffle=True,
                                validation_split=0.05,
                                callbacks=[model_checkpoint])
            '''

            print ("*"*30)
            print("iteration "+str(iDs))
            print("train_dice_coeff "+str(history.history['dice_coef']))
            print("val_dice_coeff "+str(history.history['val_dice_coef']))
            print("val_loss "+str(history.history['val_loss']))
            # list all data in history
            print(history.history.keys())
            #set_trace()
            # summarize history for accuracy
            plt.plot(history.history['dice_coef'])
            plt.plot(history.history['val_dice_coef'])
            plt.plot(history.history['val_loss'])
            plt.title('model accuracy')
            plt.ylabel('Dice')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
            print ("*"*30)


    #%%

        if test_only==True:
            model = get_unet()
            model.load_weights(data_path+'/US_NET3New.hdf5')
        datasets_save = ['set3c']
        # load testing data
        test_data_path = os.path.join(data_path, 'test')
        images_test = os.listdir(test_data_path)
        total = len(images_test) // 2

        imgs_test = np.ndarray((total, image_rows, image_cols, n_ch), dtype=np.uint8)
        imgs_test_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)

        i = 0
        print('-'*30)
        print('Creating testing images...')
        print('-'*30)
        valid_image_name = []
        for image_name in images_test:
            if 'mask' in image_name:
                continue
            image_mask_name = image_name.split('.')[0] + '_mask.jpg'
            image_mask_name = image_name.split('.')[0] + '_mask.jpg'
            #set_trace()
            valid_image_name.append(image_name)

            img = cv2.imread(os.path.join(test_data_path, image_name),-1)
            #if use_salient_images == True:
            #img = cv2.imread(os.path.join(salient_image_home, image_name),-1)
            #img = cv2.equalizeHist(img)  # histogram equalized image
            img_mask = cv2.imread(os.path.join(test_data_path, image_mask_name),cv2.IMREAD_GRAYSCALE)
            #set_trace()
            img = cv2.resize(img,(image_cols, image_rows))
            img = np.reshape(img,(image_rows, image_cols, n_ch))
            img_mask = cv2.resize(img_mask,(image_cols, image_rows))
            img_mask = np.reshape(img_mask,(image_rows, image_cols, 1))
    #        plt.imshow(cv2.cvtcolor(img, cv2.color_BGR2RGB))

            img = np.array([img])
            img_mask = np.array([img_mask])
            imgs_test[i] = img
            imgs_test_mask[i] = img_mask

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        print('Loading done for test data.')

        imgs_test = imgs_test.astype('float32')
        #imgs_test /= 255.  # scale masks to [0, 1]

        imgs_test_mask = imgs_test_mask.astype('float32')
        imgs_test_mask /= 255.  # scale masks to [0, 1]
        imgs_test_mask = np.round(imgs_test_mask)
        #set_trace()

        print('-'*30)
        print('Predicting masks on test data...')
        print('-'*30)
        imgs_mask_predict = model.predict(imgs_test, batch_size=4, verbose=1)
        #set_trace()
        dice_coeff_test = dict()
        jaccard_index_test = dict()
        # get some test numbers & possibly figures
        for jj in range(np.shape(imgs_mask_predict)[0]):
            ypred = np.squeeze(imgs_mask_predict[jj,:,:,:])
            ytrue = np.squeeze(imgs_test_mask[jj,:,:,:])
            dice_coeff_test[valid_image_name[jj]] = dice_coef_test(ytrue, ypred)
            jaccard_index_test[valid_image_name[jj]] = jaccard_index(ytrue, ypred)
        #set_trace()

        result_path = os.path.join(data_path,'results')


        # save coeff file
        if test_only==True:
             np.save(results_home+'/test_dice_coeff_'+datasets_save[iDs]+'.npy',dice_coeff_test)
             np.save(results_home +'/jaccard_index_' + datasets_save[iDs] + '.npy',jaccard_index_test)

        # to load
        # dice_coeff_test = npy.load(results_home+'/test_dice_coeff_'+datasets[iDs]+'.npy').item()

        lp=0
        for image_name in valid_image_name:
            if 'mask' in image_name:
                continue

            img_pred = np.reshape(imgs_mask_predict[lp] * 255., (image_rows, image_cols, 1))
            image_pred_name = image_name.split('.')[0] + '_pred.png'
            cv2.imwrite(os.path.join(result_path, image_pred_name), img_pred)
            lp += 1

