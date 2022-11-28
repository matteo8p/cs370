import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap


def colormap(value_range, cmap=cm.viridis):
    norm = colors.Normalize(vmin=float(value_range[0]), vmax=float(value_range[1]))
    cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    return cmap


def sample_plot(samples, cmap, marker_size):
    y = samples['y'].astype(float)
    plt.scatter(samples['x'][:, 0], samples['x'][:, 1],
                s=marker_size, c=cmap.to_rgba(y))


class Box:
    def __init__(self, x=None, left=None, right=None, down=None, up=None, cmap=None, margin=0.03):
        bounds = all((b is not None for b in (left, right, down, up)))
        assert bounds ^ (x is not None),\
            'Either give data points or all bounds, but not both'
        if bounds:
            self.left = left
            self.right = right
            self.down = down
            self.up = up
        else:
            left = np.amin(x[:, 0])
            right = np.amax(x[:, 0])
            down = np.amin(x[:, 1])
            up = np.amax(x[:, 1])
            margin *= max(right - left, up - down)
            self.left = left - margin
            self.right = right + margin
            self.down = down - margin
            self.up = up + margin
        self.cmap = cmap

    def but_with(self, left=None, right=None, down=None, up=None):
        box = Box(left=self.left, right=self.right, down=self.down, up=self.up, cmap=self.cmap)
        if left is not None:
            box.left = left
        if right is not None:
            box.right = right
        if down is not None:
            box.down = down
        if up is not None:
            box.up = up
        return box

    def paint(self, value):
        width = self.right - self.left
        height = self.up - self.down
        color = self.cmap.to_rgba(value)
        plt.gca().add_patch(
            Rectangle((self.left, self.down), width, height,
                      facecolor=color, edgecolor=None, fill=True, alpha=.25))


def draw_spiral(spiral):
    font_size = 14
    plt.figure(figsize=(7, 7), tight_layout=True)
    for m, color in enumerate(spiral['labels']):
        select = spiral['train']['y'] == m
        plt.scatter(spiral['train']['x'][select, 0],
                    spiral['train']['x'][select, 1],
                    c=color)
    plt.gca().set_aspect(1)
    plt.axis('off')
    plt.title('spiral data', fontsize=font_size)
    plt.show()


def coarse_regions(h, colors, step=0.01):
    xx, yy = np.meshgrid(np.arange(0, 1, step),
                         np.arange(0, 1, step))
    label = h.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    color_map = ListedColormap(colors)
    plt.contourf(xx, yy, label, cmap=color_map)
