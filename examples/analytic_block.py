import numpy as np

def analytic_solution(block1, block2, x, y):
    x_res = np.zeros_like(x, dtype=float)
    y_res = np.zeros_like(y, dtype=float)
    ones_x = np.ones_like(x, dtype=float)
    ones_y = np.ones_like(y, dtype=float)

    dx = block2[0] - block1[0]
    dy = block2[1] - block1[1]

    # Check that the blocks are the same size
    assert block1[2] == block2[2]
    width = block1[2]

    cos = dx/(np.sqrt(dx**2 + dy**2))
    sin = dy/(np.sqrt(dx**2 + dy**2))
    if cos == 0:
        cos = 1e-16
    if sin == 0:
        sin = 1e-16

    # Mask to denote the
    mask1 = (block1[0] - width <= x) & (x <= block1[0] + width) & (block1[1] - width <= y) & (y <= block1[1] + width)
    mask2 = (block2[0] - width <= x) & (x <= block2[0] + width) & (block2[1] - width <= y) & (y <= block2[1] + width)

    #mask3 = np.logical_not(mask1) & np.logical_not(mask2) & (b - width <= y - a * x) & (y - a * x <= b + width) & (x >= min(block1[0] - width, block1[1] - width)) & (x <= max(block2[0] + width, block2[1] + width)) & (y >= min(block1[1] - width, block1[1] - width)) & (y <= max(block2[1] + width, block2[1] + width)) & (y >= min(block1[1] - width, block2[1] - width)) & (y <= max(block1[1] + width, block2[1] + width))

    scaling1 = np.minimum(np.abs(x[mask1] - (block1[0] - np.sign(dx)*width)) / abs(cos), np.abs(y[mask1] - (block1[1] - np.sign(dy)*width)) / abs(sin))
    x_res[mask1] = cos * scaling1
    y_res[mask1] = sin * scaling1
    scaling2 = np.minimum(np.abs(x[mask2] - (block2[0] + np.sign(dx)*width))/abs(cos), np.abs(y[mask2] - (block2[1] + np.sign(dy)*width))/abs(sin))
    x_res[mask2], y_res[mask2] = cos * scaling2, sin * scaling2

    d = width * np.abs(1 - np.abs(sin/cos))
    e = 2 * width * min(abs(cos), abs(sin))


    if dx == 0:
        dist = np.abs(x - block1[0]) # The blocks are horizontally aligned
    else:
        a = dy / dx
        b = block1[1] - a * block1[0]
        dist = np.abs(a*x - y + b)*abs(cos)
    mask3 = np.logical_not(mask1) & np.logical_not(mask2) & (dist <= d + e) & (x >= min(block1[0] - width, block2[0] - width)) & (x <= max(block1[0] + width, block2[0] + width)) & (y >= min(block1[1] - width, block2[1] - width)) & (y <= max(block1[1] + width, block2[1] + width))
    mask3a = mask3 & (dist < d)
    mask3b = mask3 & (dist >= d)


    scaling3a = 2 / max(abs(cos), abs(sin))
    x_res[mask3a], y_res[mask3a] = cos * scaling3a * ones_x[mask3a], sin * scaling3a * ones_y[mask3a]
    scaling3b = scaling3a * 1/e
    x_res[mask3b], y_res[mask3b] = cos * scaling3b * (d+e - dist[mask3b]), sin * scaling3b * (d+e - dist[mask3b])

    return np.stack([x_res, y_res], axis=2)

