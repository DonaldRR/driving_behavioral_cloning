import numpy as np
import pandas as pd
import os
import cv2
import glob
import csv
import tqdm
from tqdm import tqdm
from model import *
import sklearn
from sklearn.model_selection import train_test_split

if __name__ == '__main__':


    """ Process CSVs """
    # Load CSVs
    data_dirs = glob.glob("./data/*")
    csv_fn = "driving_log.csv"
    csv_fns = [os.path.join(data_dirs[i], csv_fn) for i in range(len(data_dirs))]

    sheet = []
    for fn in csv_fns:
        f = open(fn, 'r')
        reader = csv.reader(f)
        for line in reader:
            sheet.append(line)
    sheet = np.array(sheet)
    # Images from center, left and right mounted cameras
    center_img_fns, left_img_fns, right_img_fns = sheet[:, 0], sheet[:, 1], sheet[:, 2]

    # [Steering, Throttle, Brake, Speed]
    control_info = np.float32(sheet[:, 3:])

    """ Read images """
    n_imgs = center_img_fns.shape[0]
    imgs = [cv2.cvtColor(cv2.imread(center_img_fns[i]), cv2.COLOR_BGR2RGB) for i in tqdm(range(n_imgs))]
    imgs = np.array(imgs)

    """ Get training and vlidation data """
    # Input data to model
    train_X, valid_X, train_y, valid_y = train_test_split(imgs, control_info[:, 0], test_size=0.2)
    print("Train size: {}".format(len(train_X)))
    print("Validation size: {}".format(len(valid_y)))

    """ Load model """
    model = model()
    model.summary()

    n_epochs = 10
    batch_size = 128
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    hist = model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=[valid_X, valid_y])