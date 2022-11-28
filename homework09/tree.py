import numpy as np


class Node:
    def __init__(self, j=None, t=None, p=None, left=None, right=None):
        self.j = j
        self.t = t
        self.p = p
        self.left = left
        self.right = right

    def __str__(self):
        for key, value in self.__dict__.items():
            if value is not None:
                return '{}: {}'.format(key, value)


def is_leaf(tau):
    return tau.left is None and tau.right is None


def split(x, tau):
    return tau.left if x[tau.j] < tau.t else tau.right


def ok_to_split(samples, depth, config):
    s_min, d_max, impurity = config['min samples'], config['max depth'], config['impurity']
    return impurity(samples, config) > 0. and len(samples['y']) > s_min and depth < d_max


def impurity_change(i_samples, left, right, samples, config):
    impurity = config['impurity']
    i_left, i_right = impurity(left, config), impurity(right, config)
    n_left, n_right, n_samples = len(left['y']), len(right['y']), len(samples['y'])
    delta = i_samples - (n_left * i_left + n_right * i_right) / n_samples
    return delta


def find_split(samples, config):
    impurity = config['impurity']
    current_impurity = impurity(samples, config)
    delta_opt, left_opt, right_opt, j_opt, t_opt = -1., None, None, None, None
    for j in range(samples['x'].shape[1]):
        xj = samples['x'][:, j]
        u = np.unique(xj)
        thresholds = (u[:-1] + u[1:]) / 2.
        for t in thresholds:
            below, above = xj <= t, xj > t
            left = {'x': samples['x'][below], 'y': samples['y'][below]}
            right = {'x': samples['x'][above], 'y': samples['y'][above]}
            delta = impurity_change(current_impurity, left, right, samples, config)
            if delta > delta_opt:
                delta_opt, left_opt, right_opt, j_opt, t_opt = delta, left, right, j, t

    return left_opt, right_opt, j_opt, t_opt, delta_opt


def train_tree(samples, depth, config):
    if ok_to_split(samples, depth, config):
        left, right, j, t, delta = find_split(samples, config)
        return Node(j=j, t=t, left=train_tree(left, depth + 1, config),
                    right=train_tree(right, depth + 1, config))
    else:
        distribution = config['distribution']
        return Node(p=distribution(samples, config))


def predict(x, tau, summary):
    return summary(tau.p) if is_leaf(tau) else predict(x, split(x, tau), summary)
