import numpy as np
import pandas as pd
import os
import math
from math import ceil
import cv2
import glob
import csv
import tqdm
from tqdm import tqdm
from model import *
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(os.path.join('./ori_data', batch_sample[0]))
                left_image = cv2.imread(os.path.join('./ori_data', batch_sample[1].strip(' ')))
                right_image = cv2.imread(os.path.join('./ori_data', batch_sample[2].strip(' ')))
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3]) + 0.1
                right_angle = float(batch_sample[3]) - 0.1
                images.extend([center_image, left_image, right_image,
                               cv2.flip(center_image, 1), cv2.flip(left_image, 1), cv2.flip(right_image, 1)])
                angles.extend([center_angle, left_angle, right_angle,
                               -center_angle, -left_angle, -right_angle])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

if __name__ == '__main__':


    """ Process CSVs """
    # Load CSVs
    # data_dirs = glob.glob("./data/*")
    csv_fn = "driving_log.csv"
    # csv_fns = [os.path.join(data_dirs[i], csv_fn) for i in range(len(data_dirs))]
    csv_fns = [os.path.join('./ori_data', csv_fn)]

    sheet = []
    for fn in csv_fns:
        f = open(fn, 'r')
        reader = csv.reader(f)
        for line in reader:
            sheet.append(line)
    sheet = np.array(sheet)
    # Images from center, left and right mounted cameras
    # center_img_fns, left_img_fns, right_img_fns = sheet[:, 0], sheet[:, 1], sheet[:, 2]

    # [Steering, Throttle, Brake, Speed]
    # control_info = np.float32(sheet[:, 3:])

    """ Read images """
    # n_imgs = center_img_fns.shape[0]

    # imgs_cen = np.array([cv2.imread(center_img_fns[i][68:])
    #                      for i in tqdm(range(n_imgs))])
    # imgs_left = np.array([cv2.imread(left_img_fns[i][68:])
    #                       for i in tqdm(range(n_imgs))])
    # imgs_right = np.array([cv2.imread(right_img_fns[i][68:])
    #                        for i in tqdm(range(n_imgs))])

    # imgs_cen = np.array([cv2.imread(os.path.join('./ori_data', center_img_fns[i]))
    #                      for i in tqdm(range(n_imgs))])
    # print(os.path.join('/ori_data', left_img_fns[0].strip(' ')))
    # imgs_left = np.array([cv2.imread(os.path.join('./ori_data',  left_img_fns[i].strip(' ')))
    #                       for i in tqdm(range(n_imgs))])
    # imgs_right = np.array([cv2.imread(os.path.join('./ori_data', right_img_fns[i].strip(' ')))
    #                        for i in tqdm(range(n_imgs))])
    #
    # print("cen:", imgs_cen.shape, " left:", imgs_left.shape, " right", imgs_right.shape)
    #
    # imgs_ori = np.concatenate((imgs_cen, imgs_left, imgs_right))
    # print("img ori:", imgs_ori.shape)
    # angles_ori = np.concatenate((control_info[:, 0],
    #                              np.add(control_info[:, 0], 0.1),
    #                              np.subtract(control_info[:, 0], 0.1)))
    # print("angle ori:", angles_ori.shape)
    #
    # imgs_flip = np.array([cv2.flip(imgs_ori[i], 1) for i in range(imgs_ori.shape[0])])
    # print("img flip:", imgs_flip.shape)
    # angles_flip = np.array([-angles_ori[i] for i in range(angles_ori.shape[0])])
    # print("angles flip:", angles_flip.shape)
    #
    # imgs = np.concatenate((imgs_ori, imgs_flip))
    # angles = np.concatenate((angles_ori, angles_flip))
    # print("imgs:", imgs.shape)
    # print("angles:", angles.shape)

    # Input data to model

    train_sheet, valid_sheet = train_test_split(sheet, test_size=0.2)
    train_generator = generator(train_sheet)
    valid_generator = generator(valid_sheet)

    # train_X, valid_X, train_y, valid_y = train_test_split(imgs, angles, test_size=0.2)
    # print("Training shape:{}, {}".format(train_X.shape, train_y.shape))
    # print("Validation shape:{}, {}".format(valid_X.shape, valid_y.shape))

    """ Load model """
    model = model()
    model.summary()

    print("Start training ...")
    n_epochs = 5
    batch_size = 32
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    # hist = model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=[valid_X, valid_y])
    hist = model.fit_generator(train_generator,
                               epochs=n_epochs,
                               steps_per_epoch=ceil(len(train_sheet) / batch_size),
                               validation_data=valid_generator,
                               validation_steps=(len(valid_sheet) / batch_size),
                               verbose=1)

    model_name = 'model.h5'
    model.save(model_name)
    print("Model {} saved.".format(model_name))