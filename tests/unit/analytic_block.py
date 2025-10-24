import numpy as np

def analytic_solution(block1, block2, x, y):
    x_res = np.zeros_like(x, dtype=float)
    y_res = np.zeros_like(y, dtype=float)
    ones_x = np.ones_like(x, dtype=float)
    ones_y = np.ones_like(y, dtype=float)

    dx = block2[0] - block1[0]
    dy = block2[1] - block1[1]

    a = dy/dx
    b = block1[1] - a*block1[0]

    # Check that the blocks are the same size
    assert block1[2] == block2[2]
    width = block1[2]

    cos = dx/(np.sqrt(dx**2 + dy**2))
    sin = -dy/(np.sqrt(dx**2 + dy**2))
    theta_vec = np.array([cos, sin])

    # Mask to denote the
    mask1 = (block1[0] - width <= x) & (x <= block1[0] + width) & (block1[1] - width <= y) & (y <= block1[1] + width)
    mask2 = (block2[0] - width <= x) & (x <= block2[0] + width) & (block2[1] - width <= y) & (y <= block2[1] + width)

    mask3 = np.logical_not(mask1) & np.logical_not(mask2) & (b - width <= y - a * x) & (y - a * x <= b + width)

    #scaling1 = np.min(np.abs(x[mask1] - block1[0] + width) / cos, np.abs(y[mask1] - block1[1] + width) / sin)
    x_res[mask1] = cos * np.min(np.abs(x[mask1] - block1[0] + width) / cos, np.abs(y[mask1] - block1[1] + width) / sin) * ones_x[mask1]
    y_res[mask1] = sin * np.min(np.abs(x[mask1] - block1[0] + width) / cos, np.abs(y[mask1] - block1[1] + width) / sin)
    scaling2 = np.min(np.abs(x[mask2] - block2[0] - width)/cos, -np.abs(y[mask2] - block2[1] - width)/sin)
    x_res[mask2], y_res[mask2] = cos * scaling2 * ones_x[mask2], sin * scaling2 * ones_y[mask2]

    d = width * np.max(1 + sin/cos, 1 - sin/cos)
    dist = np.abs(a*x - y + b)*cos
    mask3a = mask3 & (dist < d)
    mask3b = mask3 & (dist >= d)

    e = 2*width*min(abs(cos), abs(sin))
    scaling3a = 1 / min(abs(cos), abs(sin))
    x_res[mask3a], y_res[mask3a] = cos * scaling3a * ones_x[mask3a], sin * scaling3a * ones_y[mask3a]
    scaling3b = scaling3a * 1/(d - e)
    x_res[mask3b], y_res[mask3b] = cos * scaling3b * np.abs(dist[mask3b] - e), sin * scaling3b * np.abs(dist[mask3b] - e)

    return np.stack([x_res, y_res], axis=2)

