import csv
from PIL import Image
import numpy as np

class C100Dataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    ts_x = None  # X (data) of test set.
    ts_y = None  # Y (label) of test set.

    def __init__(self, train_csv, test_csv):
        with open(train_csv, encoding='utf-8') as f:
            reader = csv.reader(f)
            x = [x for x in reader]
            x = x[:49999]  # x[50000:] datas are outlier
            train_data = {x[0]: x[1] for x in x}

        with open(test_csv, encoding='utf-8') as f:
            reader = csv.reader(f)
            test_data = {rows[0]: rows[1] for rows in reader}

        ## read the csv for dataset (cifar100.csv, cifar100_lt.csv or cifar100_nl.csv),
        #
        # Format:
        #   image file path,classname

        ### TODO: Read the csv file and make the training and testing set
        image_test = list(test_data.keys())
        label_train = np.unique(list(train_data.values()))

        self.labels_map = dict((idx, value) for idx, value in enumerate(label_train))

        x_train = np.zeros((49999, 32, 32, 3))
        y_train = np.empty((0, 1))
        x_test = np.zeros((9999, 32, 32, 3))
        y_test = np.empty((0, 1))

        n = 0
        for key, value in train_data.items():
            img = Image.open(key)
            img = np.array(img)
            x_train[n, :, :, :] = img
            n += 1
            y = np.where(label_train == value)
            y_train = np.vstack((y_train, y))

        n = 0
        for key, value in test_data.items():
            img = Image.open(key)
            img = np.array(img)
            x_test[n, :, :, :] = img
            n += 1
            y = np.where(label_train == value)
            y_test = np.vstack((y_test, y))

        ts_x = x_test.astype(np.uint8)
        ts_y = y_test.squeeze().astype(np.uint8)
        
        tr_x = x_train.astype(np.uint8)
        tr_y = y_train.squeeze().astype(np.uint8)

        ### TODO: assign each dataset
        self.tr_x = tr_x  ### TODO: YOUR CODE HERE
        self.tr_y = tr_y  ### TODO: YOUR CODE HERE
        self.ts_x = ts_x  ### TODO: YOUR CODE HERE
        self.ts_y = ts_y  ### TODO: YOUR CODE HERE

    def getDataset(self):
        return [self.tr_x, self.tr_y, self.ts_x, self.ts_y]

    def getLabels_map(self):
        return self.labels_map


if __name__ == "__main__":
    train_csv = 'data/cifar100_nl.csv'
    test_csv = 'data/cifar100_nl_test.csv'
    Dataloader = C100Dataset(train_csv, test_csv)
    tr_x, tr_y, ts_x, ts_y = Dataloader.getDataset()
    labels_map = Dataloader.getLabels_map()

    print(tr_x.shape)
    print(tr_y.shape)
    print(ts_x.shape)
    print(ts_y.shape)
    print(labels_map)