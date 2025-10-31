import numpy as np
import matplotlib.pyplot as plt

import darsia

from analytic_block import analytic_solution


def L_dist(x, n=1):
    return (np.sum(np.abs(x)**n))**(1/n)/(x.shape[0]*x.shape[1])


def block_test(factor, block1, block2, options, weights=None, method="newton", plotting=False):
    dim = 2
    shape = (factor * 8, factor * 8)
    voxel_size = [1 / factor, 1 / factor]

    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)

    mass1_array[factor * (8-block1[1]-block1[2]) : factor * (8-block1[1]+block1[2]), factor * (block1[0]-block1[2]) : factor * (block1[0]+block1[2])] = 1
    mass2_array[factor * (8-block2[1]-block2[2]) : factor * (8-block2[1]+block2[2]), factor * (block2[0]-block2[2]) : factor * (block2[0]+block2[2])] = 1

    width = shape[1] * voxel_size[1]
    height = shape[0] * voxel_size[0]
    mass1 = darsia.Image(
        mass1_array,
        width=width,
        height=height,
        scalar=True,
        dim=dim,
        series=False,
    )

    mass2 = darsia.Image(
        mass2_array,
        width=width,
        height=height,
        scalar=True,
        dim=dim,
        series=False,
    )

    if weights is not None:
        weights = darsia.Image(
            weights,
            width=width,
            height=height,
            scalar=True,
            dim=dim,
            series=False,
        )

    distance, info = darsia.wasserstein_distance(
        mass1,
        mass2,
        weight=weights,
        method=method,
        options=options,
        plot_solution=True,
        return_solution=True,
    )

    flux = info["flux"]

    if plotting:
        plot_options = {
            "resolution": 1,
            "save": False,
            "name": "squares",
            "dpi": 800,
        }
        darsia.plotting.plot_2d_wasserstein_distance(info, **plot_options)

    flux_restructured = np.copy(flux)
    flux_restructured[:, :, 0] = -flux[:, :, 1]
    flux_restructured[:, :, 1] = flux[:, :, 0]

    return distance, flux
    '''
    flux[:, :, 1] = -flux[:, :, 1]  # reverse x-component to match coordinate system
    # make 2D grid

    ys, xs = np.meshgrid(
        voxel_size[0] * (0.5 + np.arange(shape[0] - 1, -1, -1)),
        voxel_size[1] * (0.5 + np.arange(shape[1])),
        indexing="ij",
    )
    #real_flux = analytic_solutions[case](xs, ys)
    from analytic_block import analytic_solution
    real_flux = analytic_solution(block1, block2, xs, ys)

    real_dist = np.sqrt((block1[0]-block2[0])**2 + (block1[1]-block2[1])**2)*(block1[2]*2)**2
    flux_diff = np.copy(real_flux)
    flux_diff[:, :, 0] -= flux[:, :, 1]
    flux_diff[:, :, 1] -= flux[:, :, 0]
    flux_error = L_dist(flux_diff)
    dist_error = abs(real_dist - distance)
    print(f"Distance error: {dist_error}, Flux error: {flux_error}")

    return dist_error, flux_error
    '''

def plot_flux(xs, ys, flux, real_flux):
    fig, axs = plt.subplots(1, 3, dpi=500)
    # plot a vector field
    flux_diff = np.copy(real_flux)
    flux_diff = flux_diff - flux
    for ax in axs:
        # reverse the y-axis
        #ax.invert_yaxis()
        #ax.invert_xaxis()
        pass
    axs[0].quiver(
        xs,
        ys,
        flux[:, :, 1],
        flux[:, :, 0],
        scale=50,
        color="blue",
        label="Computed Flux",
    )
    axs[0].set_title("Computed Flux")
    axs[1].quiver(
        xs,
        ys,
        real_flux[:, :, 0],
        real_flux[:, :, 1],
        scale=50,
        color="red",
        label="Analytic Flux",
    )
    axs[1].set_title("Analytic Flux")
    axs[2].quiver(
        xs,
        ys,
        flux_diff[:, :, 0],
        flux_diff[:, :, 1],
        scale=50,
        color="green",
        label="Flux Error",
    )
    axs[2].set_title("Flux Error")

    plt.show()




def circular_testing(factor, n_datapoints, options, method="newton"):
    angles = np.linspace(0, 2*np.pi, n_datapoints)

    cos = (np.round(2*np.cos(angles)*factor)/factor).astype(int)
    sin = (np.round(2*np.sin(angles)*factor)/factor).astype(int)


    dist_error_array = np.zeros(n_datapoints)
    flux_error_array = np.zeros(n_datapoints)

    for i, angle in enumerate(angles):
        block1 = [4 - cos[i], 4 - sin[i], 1]
        block2 = [4 + cos[i], 4 + sin[i], 1]
        dist = np.sqrt((block1[0]-block2[0])**2 + (block1[1]-block2[1])**2)
        num_dist, num_flux = block_test(factor, block1, block2, options, method=method)
        dist, flux = analytic_solution(block1, block2, factor)
        dist_error_array[i] = abs(dist - num_dist)/dist
        flux_error_array[i] = L_dist(flux - num_flux)/dist
    for i, angle in enumerate(angles):
        if flux_error_array[i] > 0.1:
            print(f"angle : {angle}, flux : {flux_error_array[i]}")
    fig, axs = plt.subplots(2, 1, dpi=500, constrained_layout=True)
    axs[0].plot(angles, dist_error_array)
    axs[0].set_title("Distance Error (relative)")
    axs[0].set_xlabel("Angle")
    axs[1].plot(angles, flux_error_array)
    axs[1].set_title("Flux Error (relative)")
    axs[1].set_xlabel("Angle")
    fig.suptitle(f"Method : {method}")
    plt.show()


def make_wall(factor, L=6, K=10):
    shape = (factor * 8, factor * 8)
    voxel_size = [1 / factor, 1 / factor]
    y, x = np.meshgrid(
        voxel_size[0] * (0.5 + np.arange(shape[0] - 1, -1, -1)),
        voxel_size[1] * (0.5 + np.arange(shape[1])),
        indexing="ij",
    )

    wall = np.ones_like(x)

    # Assume the wall is vertical at x=4
    dx = x[0, 1] - x[0, 0] # Assuming uniform spacing
    wall[(np.abs(x - 4) <= dx) & (np.abs(y - 4) <= L/2)] = 0.5*K/dx
    return wall


def wall_test(factor, L, K, options, plotting=False):
    dim = 2
    shape = (factor * 8, factor * 8)
    voxel_size = [1 / factor, 1 / factor]

    wall_array = make_wall(factor, L=L, K=K)

    block1 = [2, 4, 1]
    block2 = [6, 4, 1]

    dist, flux = block_test(factor, block1, block2, options, weights=wall_array, plotting=plotting)
    
    return dist, flux

if __name__ == "__main__":
    method = "newton"
    L_Bregman = 1e2
    L_Newton = 1e-2
    if method == "newton":
        L = L_Newton
    elif method == "bregman":
        L = L_Bregman
    else:
        raise ValueError("method must be either newton or bregman")

    scaling = 3e1
    regularization = 1e-16
    num_iter = int(3e2)
    tol = 1e-16
    
    options = {
        "L": L,
        "num_iter": num_iter,
        "tol": tol,
        "tol_distance": 1e-5,
        "tol_increment": 1e-5,
        "tol_residual": 1e5,
        "regularization": regularization,
        "scaling": scaling,
        "depth": 0,
        "verbose": True,
        "return_info": True
    }

    wall_test(10, 6, 2, options, plotting=True)
    
    
    factor = 5
    angle = 5.385587406153931
    cos = int(np.round(2 * np.cos(angle) * factor) / factor)
    sin = int(np.round(2 * np.sin(angle) * factor) / factor)
    block1 = [4 - cos, 4 - sin, 1]
    block2 = [4 + cos, 4 + sin, 1]
    #block_test(block1, block2, factor=factor, plotting=True)
    #wasserstein_test(4, "newton", case=1, plotting=True)
    #convergence_test([1, 2, 4, 8, 16], "newton", case=2)
    #circular_testing(10, 30, method="newton")


