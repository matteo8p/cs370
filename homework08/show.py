import numpy as np
import matplotlib.pyplot as plt


def show_split(h, samples, marker='none'):
    sx, sy = samples['x'], samples['y']
    p, n = sy == 1, sy != 1
    y_hat = h.predict(sx)
    ep, en = y_hat == 1, y_hat != 1
    tp, fp = ep & p, ep & n
    tn, fn = en & n, en & p
    t_pos, f_pos = sx[tp], sx[fp]
    t_neg, f_neg = sx[tn], sx[fn]
    marker_good, marker_bad = 'o', 'v'
    fill_style = 'full' if marker == 's' else 'none'
    marker_size = 5
    for points, marker, color in ((t_pos, marker_good, 'b'),
                                  (t_neg, marker_good, 'r'),
                                  (f_pos, marker_bad, 'r'),
                                  (f_neg, marker_bad, 'b')):
        plt.plot(points[:, 0], points[:, 1], marker + color,
                 fillstyle=fill_style, markersize=marker_size)


def show_classification(h, data_set, data_set_type, h_name):
    title = '{} on the {} set'.format(h_name, data_set_type)
    plt.figure(figsize=(8, 8), tight_layout=True)
    show_split(h, data_set)
    if data_set_type == 'training':
        if hasattr(h, 'support_'):
            supp = {'x': data_set['x'][h.support_], 'y': data_set['y'][h.support_]}
            show_split(h, supp, marker='s')
            print('\t{} support vectors found'.format(len(h.support_)))
    step, margin = 0.02, 0.1
    dx = data_set['x']
    x1_min, x1_max = dx[:, 0].min() - margin, dx[:, 0].max() + margin
    x2_min, x2_max = dx[:, 1].min() - margin, dx[:, 1].max() + margin
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, step),
                         np.arange(x2_min, x2_max, step))
    grid = {'x': np.stack((x1.ravel(), x2.ravel()), axis=1), 'y': data_set['y']}
    cat = h.predict(grid['x']).reshape(x1.shape)
    plt.contourf(x1, x2, cat, cmap=plt.cm.RdBu, alpha=0.3)
    if hasattr(h, 'support_'):
        z = h.decision_function(grid['x']).reshape(x1.shape)
        plt.contour(x1, x2, z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                    linestyles=['--', '-', '--'])
    plt.axis('equal')
    plt.axis('off')
    plt.title(title, fontsize=18)
    plt.show()
