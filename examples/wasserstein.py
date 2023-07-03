import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

import darsia

# Main script

# Define problem parameters
dim = 2

if False:
    print("Case 1")
    shape = (10, 8)
    voxel_size = [1, 1]  # 0.5, 3]
    # TODO test whether this is working: varying voxel size in x and y directions as well.

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)

    mass1_array[2:6, 2:3] = 1
    mass2_array[2:6, 5:6] = 1

    L = 1e3
    regularization = 1e-16

elif False:
    print("Case 2- Newton")
    shape = (100, 80)
    voxel_size = [1, 1]  # 0.5, 3]
    # TODO test whether this is working: varying voxel size in x and y directions as well.

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)

    mass1_array[20:60, 20:30] = 1
    mass2_array[40:80, 50:60] = 1

    L = 1e2
    increasing_L = False
    newton_like = True
    scaling = 3e1
    regularization = 1e-8
    num_iter = int(1e3)
    tol = 1e-6

elif True:
    print("Case 2- bregman")
    factor = 1
    shape = (factor * 50, factor * 40)
    voxel_size = [1, 1]  # 0.5, 3]
    # TODO test whether this is working: varying voxel size in x and y directions as well.

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)

    mass1_array[factor * 10 : factor * 30, factor * 10 : factor * 15] = 1
    mass2_array[factor * 20 : factor * 40, factor * 25 : factor * 30] = 1

    L = 1e2
    # L = 1e-2 # Newton
    scaling = 3e1
    regularization = 1e-16
    num_iter = int(1e4)
    tol = 1e-16

elif False:
    print("Case 3 - Newton")
    shape = (50, 40)
    voxel_size = [1, 1]  # 0.5, 3]
    # TODO test whether this is working: varying voxel size in x and y directions as well.

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)

    # mass1_array[1:3, 1:1] = 1
    # mass2_array[2:4, 2:3] = 1
    mass1_array[1:3, 1:3] = 1
    mass2_array[2:4, 2:4] = 1

    L = 1e-2
    increasing_L = False
    newton_like = True
    scaling = 3e1
    regularization = 1e-8
    num_iter = int(1e3)
    tol = 1e-6

elif True:
    print("Case 3 - Bregman")
    shape = (50, 40)
    voxel_size = [1, 1]  # 0.5, 3]
    # TODO test whether this is working: varying voxel size in x and y directions as well.

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)

    mass1_array[10:30, 10:15] = 1
    mass2_array[20:40, 25:30] = 1

    L = 1e4  ##4
    scaling = 3e1
    num_iter = int(1e5)
    tol = 1e-16
    regularization = 1e-18

elif False:
    print("Case 4")
    shape = (50, 40)
    voxel_size = [0.2, 0.2]  # 0.5, 3]
    # TODO test whether this is working: varying voxel size in x and y directions as well.

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)

    mass1_array[10, 10] = 1
    mass2_array[40, 30] = 1

elif False:
    print("Case 4")
    shape = (200, 160)
    voxel_size = [0.05, 0.005]  # 0.5, 3]
    # TODO test whether this is working: varying voxel size in x and y directions as well.

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)

    mass1_array[40:120, 40:60] = 1
    mass2_array[80:160, 100:120] = 1

elif True:
    print("Case 5")
    shape = (50, 40)
    voxel_size = [0.1, 1]
    # TODO test whether this is working: varying voxel size in x and y directions as well.

    # Create two mass distributions with identical mass, equal to 1
    mass1_array = np.zeros(shape, dtype=float)
    mass2_array = np.zeros(shape, dtype=float)

    mass1_array[2, 2] = 1
    mass2_array[30, 2] = 1

    L = 1e2
else:
    assert False


def integrate(array):
    return np.sum(array) * np.prod(voxel_size)


print("Integrals:", integrate(mass1_array), integrate(mass2_array))

# Normalize masses
mass1_array /= integrate(mass1_array)
mass2_array /= integrate(mass2_array)

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

# Define connectivity: face to cell
connectivity = np.zeros((num_edges, 2), dtype=int)
connectivity[: num_edges_axis[0], 0] = np.ravel(numbering_cells[:, :-1])  # left cells
connectivity[: num_edges_axis[0], 1] = np.ravel(numbering_cells[:, 1:])  # right cells
connectivity[num_edges_axis[0] :, 0] = np.ravel(numbering_cells[:-1, :])  # top cells
connectivity[num_edges_axis[0] :, 1] = np.ravel(numbering_cells[1:, :])  # bottom cells

# Define connectivity: cell to face (only for inner cells)
connectivity_cell_to_vertical_face = np.zeros((num_cells, 2), dtype=int)
connectivity_cell_to_vertical_face[np.ravel(numbering_cells[:, :-1]), 0] = np.arange(
    num_edges_axis[0]
)  # left face
connectivity_cell_to_vertical_face[np.ravel(numbering_cells[:, 1:]), 1] = np.arange(
    num_edges_axis[0]
)  # right face
connectivity_cell_to_horizontal_face = np.zeros((num_cells, 2), dtype=int)
connectivity_cell_to_horizontal_face[np.ravel(numbering_cells[:-1, :]), 0] = np.arange(
    num_edges_axis[0], num_edges_axis[0] + num_edges_axis[1]
)  # top face
connectivity_cell_to_horizontal_face[np.ravel(numbering_cells[1:, :]), 1] = np.arange(
    num_edges_axis[0], num_edges_axis[0] + num_edges_axis[1]
)  # bottom face

# Define sparse divergence operator, integrated over elements: flat_fluxes -> flat_mass
# div_data = np.concatenate(
#    (np.ones(num_edges, dtype=float), -np.ones(num_edges, dtype=float))
# )
# div_row = np.concatenate((connectivity[:, 0], connectivity[:, 1]))
# div_col = np.tile(np.arange(num_edges, dtype=int), 2)
# div = sps.csr_matrix((div_data, (div_row, div_col)), shape=(num_cells, num_edges))

div_data = np.concatenate(
    (
        voxel_size[0] * np.ones(num_edges_axis[0], dtype=float),
        voxel_size[1] * np.ones(num_edges_axis[1], dtype=float),
        -voxel_size[0] * np.ones(num_edges_axis[0], dtype=float),
        -voxel_size[1] * np.ones(num_edges_axis[1], dtype=float),
    )
)
div_row = np.concatenate(
    (
        connectivity[: num_edges_axis[0], 0],
        connectivity[num_edges_axis[0] :, 0],
        connectivity[: num_edges_axis[0], 1],
        connectivity[num_edges_axis[0] :, 1],
    )
)
div_col = np.tile(np.arange(num_edges, dtype=int), 2)
div = sps.csr_matrix((div_data, (div_row, div_col)), shape=(num_cells, num_edges))

# Define sparse mass matrix on edges: flat_fluxes -> flat_fluxes
lumped_mass_matrix_edges = sps.diags(
    np.prod(voxel_size) * np.ones(num_edges, dtype=float)
)

# Info about inner cells
inner_cells_with_vertical_faces = np.ravel(numbering_cells[:, 1:-1])
inner_cells_with_horizontal_faces = np.ravel(numbering_cells[1:-1, :])
num_inner_cells_with_vertical_faces = len(inner_cells_with_vertical_faces)
num_inner_cells_with_horizontal_faces = len(inner_cells_with_horizontal_faces)

# Define true RT0 mass matrix on edges: flat_fluxes -> flat_fluxes
mass_matrix_edges_data = np.prod(voxel_size) * np.concatenate(
    (
        2 / 3 * np.ones(num_edges, dtype=float),  # all faces
        1 / 6 * np.ones(num_inner_cells_with_vertical_faces, dtype=float),  # left faces
        1
        / 6
        * np.ones(num_inner_cells_with_vertical_faces, dtype=float),  # right faces
        1
        / 6
        * np.ones(num_inner_cells_with_horizontal_faces, dtype=float),  # top faces
        1
        / 6
        * np.ones(num_inner_cells_with_horizontal_faces, dtype=float),  # bottom faces
    )
)
mass_matrix_edges_row = np.concatenate(
    (
        np.arange(num_edges, dtype=int),
        connectivity_cell_to_vertical_face[inner_cells_with_vertical_faces, 0],
        connectivity_cell_to_vertical_face[inner_cells_with_vertical_faces, 1],
        connectivity_cell_to_horizontal_face[inner_cells_with_horizontal_faces, 0],
        connectivity_cell_to_horizontal_face[inner_cells_with_horizontal_faces, 1],
    )
)
mass_matrix_edges_col = np.concatenate(
    (
        np.arange(num_edges, dtype=int),
        connectivity_cell_to_vertical_face[inner_cells_with_vertical_faces, 1],
        connectivity_cell_to_vertical_face[inner_cells_with_vertical_faces, 0],
        connectivity_cell_to_horizontal_face[inner_cells_with_horizontal_faces, 1],
        connectivity_cell_to_horizontal_face[inner_cells_with_horizontal_faces, 0],
    )
)
mass_matrix_edges = sps.csr_matrix(
    (mass_matrix_edges_data, (mass_matrix_edges_row, mass_matrix_edges_col)),
    shape=(num_edges, num_edges),
)

# Fix mean of the pressure to be zero
integral_cells = np.prod(voxel_size) * np.ones((1, num_cells), dtype=float)

# Replace mean constraint with pressure constraint for single cell
constrained_cell = numbering_cells[10, 12]
# constrained_cell = numbering_cells[1, 1]
integral_cells = sps.csr_matrix(
    (np.ones(1, dtype=float), (np.zeros(1, dtype=int), np.array([constrained_cell]))),
    shape=(1, num_cells),
    dtype=float,
)

# Combine the operators to a mixed operator: (flat_fluxes, flat_mass) -> (flat_fluxes, flat_mass)
mixed_darcy = sps.bmat(
    [
        [lumped_mass_matrix_edges, -div.T, None],
        [div, None, -integral_cells.T],
        [None, integral_cells, None],
    ]
)

l_scheme_mixed_darcy = sps.bmat(
    [
        [L * mass_matrix_edges, -div.T, None],
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
# mixed_darcy_lu = sps.linalg.splu(mixed_darcy)
l_scheme_mixed_darcy_lu = sps.linalg.splu(l_scheme_mixed_darcy)

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

    # Determine a cell-centered Raviart-Thomas-type reconstruction of the fluxes
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


def face_restriction_scalar(cell_qty):
    """Restrict the fluxes on the cells to the faces.

    Args:
        cell_qty (np.ndarray): cell-based quantity

    Returns:
        np.ndarray: face-based quantity

    """
    # Determine the fluxes on the faces

    horizontal_face_qty = 0.5 * (cell_qty[:, :-1] + cell_qty[:, 1:])
    vertical_face_qty = 0.5 * (cell_qty[:-1, :] + cell_qty[1:, :])

    # Reshape the fluxes - hardcoding the connectivity here
    face_qty = np.concatenate([horizontal_face_qty.ravel(), vertical_face_qty.ravel()])

    return face_qty


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
    cell_flux_norm = np.maximum(np.linalg.norm(cell_flux, 2, axis=-1), regularization)
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
        # - flux_embedding.dot(lumped_mass_matrix_edges.dot(flat_flux_normed))
        - flux_embedding.dot(mass_matrix_edges.dot(flat_flux_normed))
    )


def jacobian_splu(solution, i):
    if False:
        return mixed_darcy_lu
    else:
        # TODO update only every Xth iteration
        flat_flux, _, _ = split_solution(solution)
        cell_flux = cell_reconstruction(flat_flux)
        cell_flux_norm = np.maximum(
            np.linalg.norm(cell_flux, 2, axis=-1), regularization
        )
        flat_flux_norm = face_restriction_scalar(cell_flux_norm)
        print(np.min(flat_flux_norm))
        approx_jacobian = sps.bmat(
            [
                [
                    sps.diags(np.maximum(L, 1.0 / flat_flux_norm), dtype=float)
                    * lumped_mass_matrix_edges,
                    -div.T,
                    None,
                ],
                [div, None, -integral_cells.T],
                [None, integral_cells, None],
            ]
        )
        approx_jacobian_lu = sps.linalg.splu(approx_jacobian)
        return approx_jacobian_lu


def split_solution(solution):
    """Split the solution into fluxes, pressure and lagrange multiplier."""
    # Split the solution
    flat_flux = solution[:num_edges]
    flat_pressure = solution[num_edges : num_edges + num_cells]
    flat_lagrange_multiplier = solution[-1]

    return flat_flux, flat_pressure, flat_lagrange_multiplier


# TODO use linear approximation and numerical integration for more accurate computations.
def l1_dissipation(solution):
    """Compute the l1 dissipation potential of the solution.

    Args:
        solution (np.ndarray): solution

    Returns:
        float: l1 dissipation potential

    """
    flat_flux, _, _ = split_solution(solution)
    cell_flux = cell_reconstruction(flat_flux)
    return np.sum(np.prod(voxel_size) * np.linalg.norm(cell_flux, 2, axis=-1))


def l2_dissipation(solution):
    """Compute the l2 dissipation potential of the solution.

    Args:
        solution (np.ndarray): solution

    Returns:
        float: l2 dissipation potential

    """
    flat_flux, _, _ = split_solution(solution)
    cell_flux = cell_reconstruction(flat_flux)
    return 0.5 * np.sum(
        np.prod(voxel_size) * np.linalg.norm(cell_flux, 2, axis=-1) ** 2
    )


def lumped_l2_dissipation(solution):
    """Compute the lumped l2 dissipation potential of the solution.

    Args:
        solution (np.ndarray): solution

    Returns:
        float: lumped l2 dissipation potential

    """
    flat_flux, _, _ = split_solution(solution)
    return 0.5 * flat_flux.dot(lumped_mass_matrix_edges.dot(flat_flux))


# TODO use multigrid approximation for ramping up.


def newton_solve(num_iter, tol, distance):
    # anderson = darsia.AndersonAcceleration(dimension=num_edges + num_cells + 1, depth=0)
    anderson = darsia.AndersonAcceleration(dimension=num_edges, depth=10)
    # Observation: AA can lead to less stagnation, more accurate results, and therefore
    # better solutions to mu and u. Higher depth is better, but more expensive.
    solution_i = np.zeros_like(rhs)
    for i in range(num_iter):
        if i == 0:
            residual_i = darcy_residual(rhs, solution_i)
            # jacobian_lu = mixed_darcy_lu
        else:
            residual_i = residual(rhs, solution_i)
        jacobian_lu = jacobian_splu(solution_i, i)
        update_i = jacobian_lu.solve(residual_i)

        old_distance = distance(solution_i)
        solution_i += update_i
        # solution_i = anderson(solution_i, update_i, i)
        solution_i[:num_edges] = anderson(
            solution_i[:num_edges], update_i[:num_edges], i
        )
        new_distance = distance(solution_i)
        error = [np.linalg.norm(residual_i, 2), np.linalg.norm(update_i, 2)]

        # TODO include criterion build on staganation of the solution
        # TODO include criterion on distance.

        if i > 1 and min(error) < tol:
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


def shrink(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0)


def shrink_vector(x, l):
    norm = np.linalg.norm(x, 2, axis=-1)
    scaling = np.maximum(norm - l, 0) / (norm + 1e-24)
    return x * scaling[..., None]


def determine_mu(flat_flux):
    cell_flux = cell_reconstruction(flat_flux)
    return np.linalg.norm(cell_flux, 2, axis=-1)


def bregman_solve(num_iter, tol, distance, l_scheme_mixed_darcy_lu, L):
    # First indication, solving the standard Darcy problem
    # mixed_darcy_lu = sps.linalg.splu(mixed_darcy)
    # solution = mixed_darcy_lu.solve(rhs)
    # flat_flux, _, _ = split_solution(solution)
    # cell_flux = cell_reconstruction(flat_flux)
    # cell_flux_norm = np.maximum(np.linalg.norm(cell_flux, 2, axis=-1), regularization)
    # flat_flux_norm = face_restriction_scalar(cell_flux_norm)

    anderson = darsia.AndersonAcceleration(dimension=num_edges, depth=10)
    solution_i = np.zeros_like(rhs)
    num_neg_diff = 0
    for i in range(num_iter):
        old_distance = distance(solution_i)
        flat_flux_i, _, _ = split_solution(solution_i)
        rhs_i = rhs.copy()
        # Solve regularized, linear problem with small diffusion parameter
        rhs_i[:num_edges] = L * mass_matrix_edges.dot(flat_flux_i)
        intermediate_solution_i = l_scheme_mixed_darcy_lu.solve(rhs_i)
        intermediate_flat_flux_i, _, _ = split_solution(intermediate_solution_i)

        # Shrink (only for finding the scaling parameter)
        cell_intermediate_flux_i = cell_reconstruction(intermediate_flat_flux_i)
        norm = np.linalg.norm(cell_intermediate_flux_i, 2, axis=-1)
        cell_scaling = np.maximum(norm - 1 / L, 0) / (norm + 1e-24)
        flat_scaling = face_restriction_scalar(cell_scaling)
        new_flat_flux_i = flat_scaling * intermediate_flat_flux_i

        flux_inc = new_flat_flux_i - flat_flux_i
        solution_i = intermediate_solution_i.copy()
        if False:
            solution_i[:num_edges] = new_flat_flux_i
        else:
            aa_flat_flux_i = anderson(new_flat_flux_i, flux_inc, i)
            solution_i[:num_edges] = aa_flat_flux_i
        new_distance = distance(solution_i)
        flux_diff = np.linalg.norm(new_flat_flux_i - flat_flux_i)
        mass_conservation_residual = np.linalg.norm(
            (rhs_i - broken_darcy.dot(solution_i))[num_edges:-1], 2
        )
        print(
            "Bregman iteration",
            i,
            new_distance,
            old_distance - new_distance,
            L,
            flux_diff,
            mass_conservation_residual,
        )
        if i > 1 and flux_diff < tol:
            break

        # Increase L if stagnating.
        if new_distance > old_distance:
            num_neg_diff += 1
        if abs(new_distance - old_distance) < 1e-12 or num_neg_diff > 20:
            L = L * 2
            print(f"New L: {L}")
            l_scheme_mixed_darcy = sps.bmat(
                [
                    [L * mass_matrix_edges, -div.T, None],
                    [div, None, -integral_cells.T],
                    [None, integral_cells, None],
                ]
            )
            l_scheme_mixed_darcy_lu = sps.linalg.splu(l_scheme_mixed_darcy)
            num_neg_diff = 0

        L_max = 1e8
        if L > L_max:
            break

    flat_flux, flat_pressure, flat_lagrange_multiplier = split_solution(solution_i)
    status = [i < num_iter, i, flux_diff, new_distance - old_distance, new_distance]
    return flat_flux, flat_pressure, flat_lagrange_multiplier, status


if False:
    # Linear Darcy for debugging
    mixed_darcy_lu = sps.linalg.splu(mixed_darcy)
    solution = mixed_darcy_lu.solve(rhs)
    flat_flux, flat_pressure, flat_lagrange_multiplier = split_solution(solution)
elif False:
    # Solve the problem using Newton
    flat_flux, flat_pressure, flat_lagrange_multiplier, status = newton_solve(
        num_iter, tol, l1_dissipation
    )
    print(status)
elif True:
    # Solve the problem using Bregman
    flat_flux, flat_pressure, flat_lagrange_multiplier, status = bregman_solve(
        num_iter, tol, l1_dissipation, l_scheme_mixed_darcy_lu, L
    )
    print(status)

# Cell-based fluxes
cell_flux = cell_reconstruction(flat_flux)

# Cell-based l1 dissipation
l1_dissipation_potential = np.sum(
    np.prod(voxel_size) * np.linalg.norm(cell_flux, 2, axis=-1)
)
print(f"L1 dissipation potential: {l1_dissipation_potential}")

# Cell-based l2 dissipation
l2_dissipation_potential = 0.5 * np.sum(np.prod(voxel_size) * cell_flux**2)
print(f"L2 dissipation potential: {l2_dissipation_potential}")

# Edge-based l2 dissipation
lumped_l2_dissipation_potential = 0.5 * flat_flux.dot(
    lumped_mass_matrix_edges.dot(flat_flux)
)
print(f"Lumped L2 dissipation potential: {lumped_l2_dissipation_potential}")

# Plot solution

# Reshape the pressure solution, and reconstruct cell fluxes
pressure = flat_pressure.reshape(dim_cells)

Y, X = np.meshgrid(
    voxel_size[0] * (0.5 + np.arange(shape[0] - 1, -1, -1)),
    voxel_size[1] * (0.5 + np.arange(shape[1])),
    indexing="ij",
)

# CV2 analogon.

# Convert the arrays to actual DarSIA Images
width = shape[1] * voxel_size[1]
height = shape[0] * voxel_size[0]
mass1 = darsia.Image(
    mass1_array, width=width, height=height, scalar=True, dim=2, series=False
)
mass2 = darsia.Image(
    mass2_array, width=width, height=height, scalar=True, dim=2, series=False
)

# Plot the fluxes and pressure
plt.figure("Beckman solution")
plt.pcolormesh(X, Y, pressure, cmap="turbo")
plt.colorbar()
plt.quiver(
    X,
    Y,
    scaling * cell_flux[:, :, 0],
    -scaling * cell_flux[:, :, 1],
    angles="xy",
    scale_units="xy",
    scale=1,
    alpha=0.5,
)

plt.figure("Beckman solution fluxes")
plt.pcolormesh(X, Y, mass_diff, cmap="turbo")
plt.colorbar()
plt.quiver(
    X,
    Y,
    scaling * cell_flux[:, :, 0],
    -scaling * cell_flux[:, :, 1],
    angles="xy",
    scale_units="xy",
    scale=1,
    alpha=0.5,
)

plt.figure("Beckman solution mobility")
plt.pcolormesh(X, Y, determine_mu(flat_flux), cmap="turbo")
plt.colorbar()
plt.show()

# Determine the EMD
emd = darsia.EMD()
distance = emd(mass1, mass2)
print(f"The cv2 EMD distance between the two mass distributions is: {distance} meters.")
