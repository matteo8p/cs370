from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

x_digits, y_digits = datasets.load_digits(return_X_y=True)
scaler = MinMaxScaler()
x_digits = scaler.fit_transform(x_digits)


def pack_parameters(coefficients, intercepts):
    p = []
    for c in zip(coefficients, intercepts):
        for item in c:
            p.extend(item.flatten())
    return np.array(p)


def get_parameters(mlp):
    return pack_parameters(mlp.coefs_, mlp.intercepts_)


def set_parameters(p, mlp):
    j = 0
    for layer, c in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        k = j + c[0].size
        mlp.coefs_[layer] = np.reshape(p[j:k], c[0].shape)
        j = k
        k = j + c[1].size
        mlp.intercepts_[layer] = np.reshape(p[j:k], c[1].shape)
        j = k


def backprop(mlp, x, y, cats):
    layer_units = [mlp.n_features_in_] + list(mlp.hidden_layer_sizes) + [mlp.n_outputs_]
    activations = [x] + [None] * (len(layer_units) - 1)
    deltas = [None] * (len(activations) - 1)
    weight_grads = [
        np.empty((fan_in, fan_out), dtype=x.dtype)
        for fan_in, fan_out in zip(layer_units[:-1], layer_units[1:])
    ]
    bias_grads = [
        np.empty(n_fan_out_, dtype=x.dtype) for n_fan_out_ in layer_units[1:]
    ]
    y = label_binarize(y, classes=cats).astype(bool)
    risk, weight_grads, bias_grads = mlp._backprop(
        x,
        y,
        activations,
        deltas,
        weight_grads,
        bias_grads
    )
    grad = pack_parameters(weight_grads, bias_grads)
    return risk, grad


def ensure_data_sizes(x, y):
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    return x, y


def risk_and_gradient(p, x, y, mlp):
    set_parameters(p, mlp)
    cats = np.unique(y_digits)
    x, y = ensure_data_sizes(x, y)
    risk, grad = backprop(mlp, x, y, cats)
    return risk, grad


def reset_network():
    # Initial run with one data point to get a starting parameter vector
    classes = np.unique(y_digits)
    mlp = MLPClassifier(batch_size=1, random_state=0)
    mlp.partial_fit(x_digits[:1], y_digits[:1], classes=classes)
    p = get_parameters(mlp)

    # Reinitialize with default hyper-parameters
    mlp = MLPClassifier(random_state=0)
    n_classes = len(classes)
    n_features = x_digits.shape[1]
    layer_units = [n_features] + list(mlp.hidden_layer_sizes) + [n_classes]
    mlp._validate_input(x_digits, y_digits, False, reset=True)
    mlp._random_state = np.random.RandomState(mlp.random_state)
    mlp._initialize(np.zeros((1, n_classes)), layer_units, x_digits.dtype)

    return mlp, p


def plot_risk_curves(training_risks, batch_risks, title=None, fs=16):
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    values = np.reshape(np.array(batch_risks), (-1, 2))
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    training_risks = np.array(training_risks)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.plot(training_risks[:, 0], training_risks[:, 1],
            color=colors[0], marker='.', ms=4)
    segments = mc.LineCollection(batch_risks, colors=colors[1],
                                 linewidths=2)
    ax.add_collection(segments)
    ax.legend(['training risks', 'batch risks'], fontsize=fs)
    if title is not None:
        plt.title(title, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.show()
