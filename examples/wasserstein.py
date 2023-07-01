import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

import darsia

# Main script

# Define problem parameters
dim = 2
shape = (10, 8)
voxel_size = [1, 1]  # 0.5, 3]

# Create two mass distributions with identical mass, equal to 1
mass1_array = np.zeros(shape, dtype=float)
mass2_array = np.zeros(shape, dtype=float)

mass1_array[2, 3] = 1
mass2_array[4, 5] = 1

# Difference between the masses
mass_diff = mass1_array - mass2_array

# Define dimensions of the problem
dim_cells = shape
num_cells = np.prod(dim_cells)
numbering_cells = np.arange(num_cells, dtype=int).reshape(dim_cells)

# Consider only inner edges
vertical_edges_shape = (shape[0], shape[1] - 1)
horizontal_edges_shape = (shape[0] - 1, shape[1])
num_edges_axis = [
    np.prod(vertical_edges_shape),
    np.prod(horizontal_edges_shape),
]
num_edges = np.sum(num_edges_axis)

# Define connectivity
connectivity = np.zeros((num_edges, 2), dtype=int)
connectivity[: num_edges_axis[0], 0] = np.ravel(numbering_cells[:, :-1])  # left cells
connectivity[: num_edges_axis[0], 1] = np.ravel(numbering_cells[:, 1:])  # right cells
connectivity[num_edges_axis[0] :, 0] = np.ravel(numbering_cells[:-1, :])  # top cells
connectivity[num_edges_axis[0] :, 1] = np.ravel(numbering_cells[1:, :])  # bottom cells

# Define sparse divergence operator, integrated over elements: flat_fluxes -> flat_mass
div_data = np.concatenate(
    (np.ones(num_edges, dtype=float), -np.ones(num_edges, dtype=float))
)
div_row = np.concatenate((connectivity[:, 0], connectivity[:, 1]))
div_cols = np.tile(np.arange(num_edges, dtype=int), 2)
div = sps.csr_matrix((div_data, (div_row, div_cols)), shape=(num_cells, num_edges))

# Define sparse mass matrix on edges: flat_fluxes -> flat_fluxes
lumped_mass_matrix_edges = sps.diags(
    np.prod(voxel_size) * np.ones(num_edges, dtype=float)
)

# Fix mean of the pressure to be zero
integral_cells = np.prod(voxel_size) * np.ones((1, num_cells), dtype=float)

# Combine the operators to a mixed operator: (flat_fluxes, flat_mass) -> (flat_fluxes, flat_mass)
L = 1e0
mixed_darcy = sps.bmat(
    [
        [L * lumped_mass_matrix_edges, -div.T, None],
        [div, None, -integral_cells.T],
        [None, integral_cells, None],
    ]
)

broken_darcy = sps.bmat(
    [
        [None, -div.T, None],
        [div, None, -integral_cells.T],
        [None, integral_cells, None],
    ]
)

# Define sparse embedding operator: flat_fluxes -> (flat_fluxes, flat_pressure, flat_lagrange_multiplier)
flux_embedding = sps.csr_matrix(
    (np.ones(num_edges, dtype=float), (np.arange(num_edges), np.arange(num_edges))),
    shape=(num_edges + num_cells + 1, num_edges),
)

# Sparse lu factorization of mixed operator
mixed_darcy_lu = sps.linalg.splu(mixed_darcy)

# Define sparse mass matrix on cells: flat_mass -> flat_mass
mass_matrix_cells = sps.diags(np.prod(voxel_size) * np.ones(num_cells, dtype=float))

# Flatten the problem parameters and variables
flat_mass_diff = np.ravel(mass_diff)

# Define the right hand side (flat_fluxes, flat_mass, lagrange_multiplier)
rhs = np.concatenate(
    [
        np.zeros(num_edges, dtype=float),
        mass_matrix_cells.dot(flat_mass_diff),
        np.zeros(1, dtype=float),
    ]
)


def cell_reconstruction(flat_flux):
    """Reconstruct the fluxes on the cells from the fluxes on the edges.

    Args:
        flat_flux (np.ndarray): flat fluxes

    Returns:
        np.ndarray: cell-based fluxes

    """
    # Reshape fluxes - use duality of faces and normals
    horizontal_fluxes = flat_flux[: num_edges_axis[0]].reshape(vertical_edges_shape)
    vertical_fluxes = flat_flux[num_edges_axis[0] :].reshape(horizontal_edges_shape)

    # Determine a cell-based Raviart-Thomas reconstruction of the fluxes
    cell_flux = np.zeros((*dim_cells, dim), dtype=float)
    # Horizontal fluxes
    cell_flux[:, :-1, 0] += 0.5 * horizontal_fluxes
    cell_flux[:, 1:, 0] += 0.5 * horizontal_fluxes
    # Vertical fluxes
    cell_flux[:-1, :, 1] += 0.5 * vertical_fluxes
    cell_flux[1:, :, 1] += 0.5 * vertical_fluxes

    return cell_flux


def face_restriction(cell_flux):
    """Restrict the fluxes on the cells to the faces.

    Args:
        cell_flux (np.ndarray): cell-based fluxes

    Returns:
        np.ndarray: face-based fluxes

    """
    # Determine the fluxes on the faces
    horizontal_fluxes = 0.5 * (cell_flux[:, :-1, 0] + cell_flux[:, 1:, 0])
    vertical_fluxes = 0.5 * (cell_flux[:-1, :, 1] + cell_flux[1:, :, 1])

    # Reshape the fluxes
    flat_flux = np.concatenate(
        [horizontal_fluxes.ravel(), vertical_fluxes.ravel()], axis=0
    )

    return flat_flux


def darcy_residual(rhs, solution):
    """Compute the residual of the solution.

    Args:
        rhs (np.ndarray): right hand side
        solution (np.ndarray): solution

    Returns:
        np.ndarray: residual

    """
    flat_flux, _, _ = split_solution(solution)
    return (
        rhs
        - broken_darcy.dot(solution)
        - flux_embedding.dot(lumped_mass_matrix_edges.dot(flat_flux))
    )


def normed_flat_fluxes(flat_flux):
    cell_flux = cell_reconstruction(flat_flux)
    regularization = 1e-8  # TODO
    cell_flux_norm = np.linalg.norm(cell_flux, axis=-1) + regularization
    cell_flux_normed = cell_flux / cell_flux_norm[..., None]
    flat_flux_normed = face_restriction(cell_flux_normed)

    return flat_flux_normed


def residual(rhs, solution):
    """Compute the residual of the solution.

    Args:
        rhs (np.ndarray): right hand side
        solution (np.ndarray): solution

    Returns:
        np.ndarray: residual

    """
    flat_flux, _, _ = split_solution(solution)
    flat_flux_normed = normed_flat_fluxes(flat_flux)
    return (
        rhs
        - broken_darcy.dot(solution)
        - flux_embedding.dot(lumped_mass_matrix_edges.dot(flat_flux_normed))
    )


def jacobian_splu(solution):
    if False:
        return mixed_darcy_lu
    else:
        flat_flux, _, _ = split_solution(solution)
        regularization = 1e-4
        flat_flux_normed = normed_flat_fluxes(flat_flux) + regularization
        approx_jacobian = mixed_darcy + flux_embedding.dot(
            lumped_mass_matrix_edges.dot(1 / flat_flux_normed)
        )
        return sps.linalg.splu(approx_jacobian)


def split_solution(solution):
    """Split the solution into fluxes, pressure and lagrange multiplier."""
    # Split the solution
    flat_flux = solution[:num_edges]
    flat_pressure = solution[num_edges : num_edges + num_cells]
    flat_lagrange_multiplier = solution[-1]

    return flat_flux, flat_pressure, flat_lagrange_multiplier


def l1_dissipation(solution):
    """Compute the l1 dissipation potential of the solution.

    Args:
        solution (np.ndarray): solution

    Returns:
        float: l1 dissipation potential

    """
    flat_flux, _, _ = split_solution(solution)
    cell_flux = cell_reconstruction(flat_flux)
    return np.sum(np.prod(voxel_size) * np.abs(cell_flux))


def l2_dissipation(solution):
    """Compute the l2 dissipation potential of the solution.

    Args:
        solution (np.ndarray): solution

    Returns:
        float: l2 dissipation potential

    """
    flat_flux, _, _ = split_solution(solution)
    cell_flux = cell_reconstruction(flat_flux)
    return 0.5 * np.sum(np.prod(voxel_size) * cell_flux**2)


def lumped_l2_dissipation(solution):
    """Compute the lumped l2 dissipation potential of the solution.

    Args:
        solution (np.ndarray): solution

    Returns:
        float: lumped l2 dissipation potential

    """
    flat_flux, _, _ = split_solution(solution)
    return 0.5 * flat_flux.dot(lumped_mass_matrix_edges.dot(flat_flux))


def newton_solve(num_iter, tol, distance):
    solution_i = np.zeros_like(rhs)
    for i in range(num_iter):
        # if i == 0:
        if True:
            residual_i = darcy_residual(rhs, solution_i)
            jacobian_lu = mixed_darcy_lu
        else:
            residual_i = residual(rhs, solution_i)
            jacobian_lu = jacobian_splu(solution_i)
        update_i = jacobian_lu.solve(residual_i)

        old_distance = distance(solution_i)
        solution_i += update_i
        new_distance = distance(solution_i)
        error = [np.linalg.norm(residual_i), np.linalg.norm(update_i)]

        if min(error) < tol:
            break
        else:
            print(
                f"Newton iteration {i}: {error}, {old_distance - new_distance}, {new_distance}"
            )

    # Split the solution
    flat_flux, flat_pressure, flat_lagrange_multiplier = split_solution(solution_i)

    # Performance
    status = [
        i < num_iter,
        i,
        error[0],
        error[1],
        new_distance - old_distance,
        new_distance,
    ]

    return flat_flux, flat_pressure, flat_lagrange_multiplier, status


# Solve the problem
flat_flux, flat_pressure, flat_lagrange_multiplier, status = newton_solve(
    int(1e4), 1e-6, l1_dissipation
)
print(status)

print(flat_flux)
# flat_flux = normed_flat_fluxes(flat_flux)
# Plot solution

# Reshape the pressure solution, and reconstruct cell fluxes
pressure = flat_pressure.reshape(dim_cells)
cell_flux = cell_reconstruction(flat_flux)

Y, X = np.meshgrid(
    voxel_size[0] * (0.5 + np.arange(shape[0] - 1, -1, -1)),
    voxel_size[1] * (0.5 + np.arange(shape[1])),
    indexing="ij",
)
scaling = 1

# Plot the fluxes and pressure
plt.figure("Beckman solution")
# plt.imshow(pressure, cmap="turbo")
plt.pcolormesh(X, Y, pressure, cmap="turbo")
plt.colorbar()
plt.quiver(
    X,
    Y,
    scaling * cell_flux[:, :, 0],
    scaling * cell_flux[:, :, 1],
    angles="xy",
    scale_units="xy",
    scale=1,
    alpha=0.5,
)
plt.show()

# Cell-based l1 dissipation
l1_dissipation_potential = np.sum(np.prod(voxel_size) * np.abs(cell_flux))
print(f"L1 dissipation potential: {l1_dissipation_potential}")

# Cell-based l2 dissipation
l2_dissipation_potential = 0.5 * np.sum(np.prod(voxel_size) * cell_flux**2)
print(f"L2 dissipation potential: {l2_dissipation_potential}")

# Edge-based l2 dissipation
lumped_l2_dissipation_potential = 0.5 * flat_flux.dot(
    lumped_mass_matrix_edges.dot(flat_flux)
)
print(f"Lumped L2 dissipation potential: {lumped_l2_dissipation_potential}")
