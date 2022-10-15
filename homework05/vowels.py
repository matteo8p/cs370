import numpy as np
from sklearn import preprocessing


def read_data(file_name):
    first, xs, ys = True, [], []
    with open(file_name, 'r') as file:
        for line in file:
            if first:
                first = False
                continue
            items = line.split(',')
            x = [float(i) for i in items[2:]]
            y = int(items[1])
            xs.append(x)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys) - 1  # Convert to zero-based labels
    return xs, ys


def vowel_dataset():
    x_train, y_train = read_data('vowel.train')
    x_test, y_test = read_data('vowel.test')
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    meaning = ['heed', 'hid', 'head', 'had', 'hard',
               'hud', 'hod', 'hoard', 'hood', "who'd", 'heard']
    d = {'train': {'x': x_train, 'y': y_train},
         'test': {'x': x_test, 'y': y_test},
         'label meaning': meaning}
    return d
