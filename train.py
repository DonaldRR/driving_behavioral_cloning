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
import re
from re import finditer


def generator(samples, batch_size=32):

    """
    Samples generator
    """

    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                left_image = cv2.imread(batch_sample[1].strip(' '))
                right_image = cv2.imread(batch_sample[2].strip(' '))
                # center_image = cv2.imread(os.path.join('./ori_data', batch_sample[0]))
                # left_image = cv2.imread(os.path.join('./ori_data', batch_sample[1].strip(' ')))
                # right_image = cv2.imread(os.path.join('./ori_data', batch_sample[2].strip(' ')))
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


def process_sheet(folder_name):

    """
    Process all running CSV files to one sheet
    """

    data_dirs = glob.glob(folder_name)
    csv_fn = "driving_log.csv"
    csv_fns = [os.path.join(data_dirs[i], csv_fn) for i in range(len(data_dirs))]

    sheet = []
    for i in range(len(csv_fns)):
        f = open(fn, 'r')
        reader = csv.reader(f)
        for line in tqdm(reader):
            for j in range(3):
                line[j] = os.path.join(data_dirs[i],
                                       line[j][[t.span()[0] for t in finditer(r'IMG', line[j])][0]:])
            sheet.append(line)

    return np.array(sheet)

import argparse
if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument(name='-e', default=5, type=int, dest='epochs')
    parser.add_argument(name='-o', default='model.h5', dest='output')
    parser.add_argument(name='-f', default='ori_data', dest='folder')
    parser.add_argument(name='-b', default=16, type=int, dest='batch_size')
    args = parser.parse_args()

    """ Variables"""
    n_epochs = args.epochs
    batch_size = args.batch_size
    model_name = args.output

    """ Load model """
    model = model()
    model.summary()

    """ Load data"""
    sheet = process_sheet('./', args.folder)
    train_sheet, valid_sheet = train_test_split(sheet, test_size=0.2)
    train_generator = generator(train_sheet, batch_size)
    valid_generator = generator(valid_sheet, batch_size)

    """ Training """
    print("Start training ...")
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
    hist = model.fit_generator(train_generator,
                               epochs=n_epochs,
                               steps_per_epoch=ceil(len(train_sheet) / batch_size),
                               validation_data=valid_generator,
                               validation_steps=(len(valid_sheet) / batch_size),
                               verbose=1)

    """ Save model"""
    model.save(model_name)
    print("Model {} saved.".format(model_name))