import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Polygon

# from IPython.display import clear_output

# @title Plotting Code
PLT_NOOP = np.array([[-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]])
PLT_UP = np.array([[0, 0], [0.5, 0.5], [-0.5, 0.5]])
PLT_LEFT = np.array([[0, 0], [-0.5, 0.5], [-0.5, -0.5]])
PLT_RIGHT = np.array([[0, 0], [0.5, 0.5], [0.5, -0.5]])
PLT_DOWN = np.array([[0, 0], [0.5, -0.5], [-0.5, -0.5]])

TXT_OFFSET_VAL = 0.3
TXT_CENTERING = np.array([-0.08, -0.05])
TXT_NOOP = np.array([0.0, 0]) + TXT_CENTERING
TXT_UP = np.array([0, TXT_OFFSET_VAL]) + TXT_CENTERING
TXT_LEFT = np.array([-TXT_OFFSET_VAL, 0]) + TXT_CENTERING
TXT_RIGHT = np.array([TXT_OFFSET_VAL, 0]) + TXT_CENTERING
TXT_DOWN = np.array([0, -TXT_OFFSET_VAL]) + TXT_CENTERING

ACT_OFFSETS = [
    [PLT_NOOP, TXT_NOOP],
    [PLT_UP, TXT_UP],
    [PLT_DOWN, TXT_DOWN],
    [PLT_LEFT, TXT_LEFT],
    [PLT_RIGHT, TXT_RIGHT],
]

PLOT_CMAP = cm.RdYlBu


def plot_sa_values(
    env, q_values, text_values=True, invert_y=True, update=False, title=None
):
    w = env.gs.width
    h = env.gs.height

    # if update:
    #     clear_output(wait=True)
    plt.figure(figsize=(2 * w, 2 * h))
    ax = plt.gca()
    normalized_values = q_values
    normalized_values = normalized_values - np.min(normalized_values)
    normalized_values = normalized_values / np.max(normalized_values)
    for x, y in itertools.product(range(w), range(h)):
        state_idx = env.gs.xy_to_idx((x, y))
        if invert_y:
            y = h - y - 1
        xy = np.array([x, y])
        xy3 = np.expand_dims(xy, axis=0)

        for a in range(4, -1, -1):
            val = normalized_values[state_idx, a]
            og_val = q_values[state_idx, a]
            patch_offset, txt_offset = ACT_OFFSETS[a]
            if text_values:
                xy_text = xy + txt_offset
                ax.text(xy_text[0], xy_text[1], "%.2f" % og_val, size="small")
            color = PLOT_CMAP(val)
            ax.add_patch(Polygon(xy3 + patch_offset, True, color=color))
    ax.set_xticks(np.arange(-1, w + 1, 1))
    ax.set_yticks(np.arange(-1, h + 1, 1))
    plt.grid()
    if title:
        plt.title(title)
    plt.show()


def plot_s_values(
    env, v_values, text_values=True, invert_y=True, update=False, title=None
):
    w = env.gs.width
    h = env.gs.height
    # if update:
    #     clear_output(wait=True)
    plt.figure(figsize=(2 * w, 2 * h))
    ax = plt.gca()
    normalized_values = v_values
    normalized_values = normalized_values - np.min(normalized_values)
    normalized_values = normalized_values / np.max(normalized_values)
    for x, y in itertools.product(range(w), range(h)):
        state_idx = env.gs.xy_to_idx((x, y))
        if invert_y:
            y = h - y - 1
        xy = np.array([x, y])

        val = normalized_values[state_idx]
        og_val = v_values[state_idx]
        if text_values:
            xy_text = xy
            ax.text(xy_text[0], xy_text[1], "%.2f" % og_val, size="small")
        color = PLOT_CMAP(val)
        ax.add_patch(Rectangle(xy - 0.5, 1, 1, color=color))
    ax.set_xticks(np.arange(-1, w + 1, 1))
    ax.set_yticks(np.arange(-1, h + 1, 1))
    plt.grid()
    if title:
        plt.title(title)
    plt.show()
