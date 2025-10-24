import numpy as np
import matplotlib.pyplot as plt

import darsia

def wasserstein_test(factor, method="newton", case=1, plotting=False):
    dim = 2
    shape = (factor * 8, factor * 8)
    voxel_size = [1/factor, 1/factor]

    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)
    if case == 0:
        mass1_array[factor * 3 : factor * 5, factor * 1 : factor * 3] = 1
        mass2_array[factor * 3 : factor * 5, factor * 5 : factor * 7] = 1
    elif case == 1:
        block1 = [5, 2, 1]
        block2 = [3, 6, 1]
        mass1_array[factor * 4 : factor * 6, factor * 1 : factor * 3] = 1
        mass2_array[factor * 2 : factor * 4, factor * 5 : factor * 7] = 1
    elif case == 2:
        mass1_array[factor * 3 : factor * 5, factor * 3 : factor * 5] = 1
        mass2_array[factor * 1 : factor * 2, factor * 1 : factor * 2] = 1
        mass2_array[factor * 1 : factor * 2, factor * 6 : factor * 7] = 1
        mass2_array[factor * 6 : factor * 7, factor * 1 : factor * 2] = 1
        mass2_array[factor * 6 : factor * 7, factor * 6 : factor * 7] = 1

    else:
        raise ValueError("case must be either 1 or 2 or 3")

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
    num_iter = int(1e4)
    tol = 1e-16


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


    options = {
        "L": L,
        "num_iter": num_iter,
        "tol": tol,
        "tol_distance": 1e-5,
        "regularization": regularization,
        "scaling": scaling,
        "depth": 0,
        "verbose": False,
        "return_info": True
    }


    distance, info = darsia.wasserstein_distance(
        mass1,
        mass2,
        method=method,
        options=options,
        plot_solution=True,
        return_solution=True,
    )
    print(
        f"The DarSIA EMD distance ({method}) between the two mass distributions is: ",
        distance,
    )
    flux = -info["flux"]

    if plotting:
        #darsia.plotting.plot_2d_wasserstein_distance(info, resolution=2, save=False, name='squares', dpi=100)
        pass

    # make 2D grid
    xs, ys = np.mgrid[4/(shape[1]):8:8/shape[1], 4/(shape[0]):8:8/shape[0]]
    #real_flux = analytic_solutions[case](xs, ys)
    from analytic_block import analytic_solution
    analytic_solution(block1, block2, xs, ys)

    real_dist = distances[case]
    flux_error = L_dist(real_flux-flux)
    dist_error = abs(real_dist - distance)
    print(f"Distance error: {dist_error}, Flux error: {flux_error}")

    if plotting:
        fig, axs = plt.subplots(1, 3)
        # plot a vector field
        axs[0].quiver(
            xs,
            ys,
            flux[:, :, 0],
            flux[:, :, 1],
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
            flux[:, :, 0] - real_flux[:, :, 0],
            flux[:, :, 1] - real_flux[:, :, 1],
            scale=50,
            color="green",
            label="Flux Error",
        )
        axs[2].set_title("Flux Error")
        plt.show()

    return dist_error, flux_error


def L_dist(x, n=1):
    return (np.sum(np.abs(x)**n))**(1/n)/(x.shape[0]*x.shape[1])


def convergence_test(factor_list, method, case=0):
    distances = np.zeros(len(factor_list))
    flux_errors = np.zeros(len(factor_list))
    for i, factor in enumerate(factor_list):
        dist_error, flux_error = wasserstein_test(factor, method, case=case)
        distances[i] = dist_error
        flux_errors[i] = flux_error

    # make loglog plot
    fig, axs = plt.subplots(2)
    axs[0].loglog(factor_list, distances)
    axs[1].loglog(factor_list, flux_errors)
    axs[0].set_title("Distance Error")
    axs[1].set_title("Flux Error")
    axs[0].set_xlabel("Refinement Factor")
    plt.show()


def analytic_solution_1(x, y):
    x_res = np.zeros_like(x, dtype=float)
    y_res = np.zeros_like(y, dtype=float)

    mask_y = (x >= 3) & (x <= 5)

    mask1 = mask_y & (y > 1) & (y <= 3)
    mask2 = mask_y & (y > 3) & (y <= 5)
    mask3 = mask_y & (y > 5) & (y <= 7)

    y_res[mask1] = y[mask1] - 1
    y_res[mask2] = 2
    y_res[mask3] = 7 - y[mask3]

    return np.stack([x_res, -y_res], axis=2)

def analytic_solution_2(x, y):
    x_res = np.zeros_like(x, dtype=float)
    y_res = np.zeros_like(y, dtype=float)

    # Define the flux for the regions outside the squares
    mask = (y + x/2 > 4.5) & (y + x/2 < 7) & (x > 2) & (x < 6)
    x_res[mask], y_res[mask] = 2*(1.5-np.abs(y[mask] + x[mask]/2 - 6)), -(1.5-np.abs(y[mask] + x[mask]/2 - 6))

    # Define the flux for the left square
    mask = (x >= 1) & (x <= 3) & (y >= 4) & (y <= 6)
    x_res[mask], y_res[mask] = 2*(np.minimum(x[mask] - 1, 2*(6 - y[mask]))), -(np.minimum(x[mask] - 1, 2*(6 - y[mask])))

    # Define the flux for the right square
    mask = (x >= 5) & (x <= 7) & (y >= 2) & (y <= 4)
    x_res[mask], y_res[mask] = 2*(np.minimum(7 - x[mask], (y[mask] - 2)*2)), -(np.minimum(7 - x[mask], (y[mask] - 2)*2))

    return np.stack([x_res, y_res], axis=2)

def analytic_solution_3(x, y):
    x_res = np.zeros_like(x, dtype=float)
    y_res = np.zeros_like(y, dtype=float)

    # Define the flux for the regions outside the squares
    mask_11 = (x < 4) & (x > 1) & (y > 1) & (y < 4)
    x_res[mask_11], y_res[mask_11] = -np.maximum(1 - np.abs(x[mask_11] - y[mask_11])/np.sqrt(2), 0), -np.maximum(1 - np.abs(x[mask_11] - y[mask_11])/np.sqrt(2), 0)
    mask_12 = (x < 4) & (x > 1) & (y > 4) & (y < 7)
    x_res[mask_12], y_res[mask_12] = -np.maximum(1 - np.abs(x[mask_12] - (8 - y[mask_12]))/np.sqrt(2), 0), np.maximum(1 - np.abs(x[mask_12] - (8 - y[mask_12]))/np.sqrt(2), 0)
    mask_21 = (x > 4) & (x < 7) & (y > 1) & (y < 4)
    x_res[mask_21], y_res[mask_21] = np.maximum(1 - np.abs((8 - x[mask_21]) - y[mask_21])/np.sqrt(2), 0), -np.maximum(1 - np.abs((8 - x[mask_21]) - y[mask_21])/np.sqrt(2), 0)
    mask_22 = (x > 4) & (x < 7) & (y > 4) & (y < 7)
    x_res[mask_22], y_res[mask_22] = np.maximum(1 - np.abs((8 - x[mask_22]) - (8 - y[mask_22]))/np.sqrt(2), 0), np.maximum(1 - np.abs((8 - x[mask_22]) - (8 - y[mask_22]))/np.sqrt(2), 0)

    # Define the fluxes for the four quadrants of the initial square
    mask_11 = (x >=3) & (x <=4) & (y >=3) & (y <=4)
    x_res[mask_11], y_res[mask_11] = -np.minimum(4 - x[mask_11], 4 - y[mask_11]), -np.minimum(4 - x[mask_11], 4 - y[mask_11])
    mask_12 = (x >= 3) & (x <=4) & (y >=4) & (y <=5)
    x_res[mask_12], y_res[mask_12] = -np.minimum(4 - x[mask_12], y[mask_12] - 4), np.minimum(4 - x[mask_12], y[mask_12]-4)
    mask_21 = (x >=4) & (x <=5) & (y >=3) & (y <=4)
    x_res[mask_21], y_res[mask_21] = np.minimum(x[mask_21] - 4, 4 - y[mask_21]), -np.minimum(x[mask_21]-4, 4 - y[mask_21])
    mask_22 = (x >= 4) & (x <= 5) & (y >= 4) & (y <= 5)
    x_res[mask_22], y_res[mask_22] = np.minimum(x[mask_22] - 4, y[mask_22] - 4), np.minimum(x[mask_22] - 4, y[mask_22] - 4)

    # Define the fluxes for the four corners of the target squares
    mask_11 = (x >=1) & (x <=2) & (y >=1) & (y <=2)
    x_res[mask_11], y_res[mask_11] = -np.minimum(x[mask_11] - 1, y[mask_11] - 1), -np.minimum(x[mask_11] - 1, y[mask_11] - 1)
    mask_12 = (x >= 1) & (x <=2) & (y >=6) & (y <=7)
    x_res[mask_12], y_res[mask_12] = -np.minimum(x[mask_12] - 1, 7 - y[mask_12]), np.minimum(x[mask_12] - 1, 7 - y[mask_12])
    mask_21 = (x >=6) & (x <=7) & (y >=1) & (y <=2)
    x_res[mask_21], y_res[mask_21] = np.minimum(7 - x[mask_21], y[mask_21] - 1), -np.minimum(7 - x[mask_21], y[mask_21] - 1)
    mask_22 = (x >= 6) & (x <= 7) & (y >= 6) & (y <= 7)
    x_res[mask_22], y_res[mask_22] = np.minimum(7 - x[mask_22], 7 - y[mask_22]), np.minimum(7 - x[mask_22], 7 - y[mask_22])

    return np.stack([x_res, y_res], axis=2)


analytic_solutions = [analytic_solution_1, analytic_solution_2, analytic_solution_3]
distances = [16, 8*np.sqrt(5), 8*np.sqrt(2)]


if __name__ == "__main__":
    wasserstein_test(1, "newton", case=1, plotting=True)
    #convergence_test([1, 2, 4, 8, 16], "newton", case=2)

