"""
Test suite for the FFTPoissonSolverNeumann class, focusing on convergence.

"""

import pytest
import numpy as np
import darsia
import scipy.sparse as sps
import time

def test_fft_poisson_convergence():
    """
    Test the convergence of the FFTPoissonSolverNeumann class in 2D
    by comparing to an analytical solution.
    """

    def analytical_solution(x, y):
        """Analytical solution for the test problem."""
        return x**2 / 2 - x**3 / 3 - 1 / 12 + y**2 / 2 - y**3 / 3 - 1 / 12

    # def analytical_solution(x, y):
    #     """Analytical solution for the test problem."""
    #     return np.cos(np.pi * x) * np.cos(np.pi * y)

    
    def rhs_function(x, y):
        """Right-hand side function corresponding to the analytical solution."""
        return -(1 - 2 * x) - (1 - 2 * y)

    # def rhs_function(x, y):
    #     """Right-hand side function corresponding to the analytical solution."""
    #     return 2 * np.pi **2 * np.cos(np.pi * x) * np.cos(np.pi * y)

    
    # Define the domain
    domain_start = np.array([0.0, 0.0])
    domain_end = np.array([1.0, 1.0])

    # Define the number of grid refinements
    base_mesh = 5
    num_refinements = 4

    # Define the initial grid size
    initial_shape = (8, 8)

    # Define list for storing errors and voxel sizes
    errors_dct = []
    errors_ksp = []
    voxel_sizes = []

    
    for i in range(base_mesh,base_mesh + num_refinements):
        
        # Refine the grid
        shape = (initial_shape[0] * 2**i, initial_shape[1] * 2**i)
        
        voxel_size = (
            (domain_end[0] - domain_start[0]) / shape[0],
            (domain_end[1] - domain_start[1]) / shape[1],
        )
        print(f" Mesh {i=} {shape=} {voxel_size=}")


        # Create the grid
        grid = darsia.Grid(
            shape=shape, voxel_size=voxel_size
        )
        meta_2d = {"width": 1, "height": 1, "space_dim": 2, "scalar": True}
        

        # Create a Poisson solver with zero Neumann boundary conditions
        poisson_solver = darsia.utils.linear_solvers.dct.DCTSolver(grid)

        # Define the x and y coordinates
        x_coordinates = np.linspace(
            domain_start[0] + voxel_size[0] / 2,
            domain_end[0] - voxel_size[0] / 2,
            shape[0],
        )
        #print(x_coordinates)
        y_coordinates = np.linspace(
            domain_start[1] + voxel_size[1] / 2,
            domain_end[1] - voxel_size[1] / 2,
            shape[1],
        )

        
        x, y = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")

        
        #xx = -1+dx/2:dx:1-dx/2
        #yy = -1+dy/2:dy:1-dy/2
        #[x,y] = np.mgrid(x_coordinates,y_coordinates) # Note: not meshgrid


        print(np.min(x), np.max(x), voxel_size[0] / 2)
        print(np.min(y), np.max(y), voxel_size[1] / 2)
        
        # Compute the right-hand side and ensure zero mean
        rhs = rhs_function(x, y)


        # Compute the analytical solution
        analytical_sol = analytical_solution(x, y)

        exact_img = darsia.Image(img=analytical_sol,**meta_2d)
        exact_img.to_vtk(f"exact_nref{i}","pot")

        
        #print(np.mean(rhs))
        #rhs -= np.mean(rhs)
        
        rhs = np.ravel(rhs, order="F")
        volume = np.prod(voxel_size)
        print(volume)
        rhs *= volume
        
        # Solve the Poisson problem
        start = time.time()
        solution_dct = poisson_solver.solve(rhs)
        end = time.time()
        print(f" DCT solver time: {end - start} seconds")


        pot_dct = solution_dct.reshape(shape, order="F")
        #pot_dct = solution_dct.reshape(shape, order="F")
        pivot_img = darsia.Image(img=pot_dct,**meta_2d)
        pivot_img.to_vtk(f"pot_dct_nref{i}","pot_dct")



        # fv solution

        
        mass_matrix_faces = darsia.FVMass(grid, "faces", True).mat
        """sps.csc_matrix: mass matrix on faces: flat fluxes -> flat fluxes"""

        inverse_mass_matrix_faces = sps.diags(
            1.0 / mass_matrix_faces.diagonal(), format="csc"
        )
        div = darsia.FVDivergence(grid).mat
        grad = div.transpose().tocsc()
        Laplacian_matrix = div * inverse_mass_matrix_faces * grad


        res = Laplacian_matrix * solution_dct - rhs
        rel_res = np.linalg.norm(res)/np.linalg.norm(rhs)
        print(f" DCT residual norm: {rel_res:.2e}")
        assert (rel_res < 1e-8)

        ksp = False
        if ksp:
        
            #print(Laplacian_matrix)
            # save laplacian matrix to file as text
            #
            #
            #
            # from scipy.io import mmwrite, savemat
            # mmwrite(f"laplacian_nref{i}.mtx",Laplacian_matrix)


            kernel = np.ones(grid.num_cells)
            kernel /= np.linalg.norm(kernel)
            ksp_poisson_solver = darsia.linalg.KSP(
                Laplacian_matrix,
                nullspace=[kernel],
                appctx={},
                solver_prefix="poisson",
            )
            ksp_ctrl = {    
                "ksp_type": "cg",
                "ksp_rtol": 1e-10,
                "ksp_max_it": 100,
                "pc_type": "hypre",
                #"ksp_monitor_true_residual" : None
            }
            ksp_poisson_solver.setup(ksp_ctrl)
            
            start = time.time()
            solution_ksp = ksp_poisson_solver.solve(rhs)
            end = time.time()
            print(f" KSP solver time: {end - start} seconds")
            
            res = Laplacian_matrix * solution_ksp - rhs
            print(f" KSP residual norm: {np.linalg.norm(res)/np.linalg.norm(rhs)}")
            
        
            pot_ksp = solution_ksp.reshape(shape, order="F")
            ksp_img = darsia.Image(img=pot_ksp,**meta_2d)
            ksp_img.to_vtk(f"pot_ksp_nref{i}","pot_ksp")
            
            error_ksp = np.linalg.norm(solution_ksp - analytical_sol) / np.linalg.norm(
            analytical_sol
            )
            
            errors_ksp.append(error_ksp)
        

       

        error = analytical_sol - pot_dct
        error_img = darsia.Image(img=error,**meta_2d)
        error_img.to_vtk(f"error_dct_nref{i}","err")
        
        analytical_sol = np.ravel(analytical_sol,"F")

        
        # Compute the error
        error_dct = np.linalg.norm(solution_dct - analytical_sol) / np.linalg.norm(
            analytical_sol
        )
        errors_dct.append(error_dct)
        voxel_sizes.append(voxel_size[0])

    # Check the convergence rate (should be approximately 2)
    print(errors_dct)


    convergence_rate_dct = np.polyfit(np.log(voxel_sizes), np.log(errors_dct), 1)[0]
    print(f"{convergence_rate_dct=}")

    # Assertions
    assert len(errors_dct) == num_refinements
    assert convergence_rate_dct >= 1.5

    
    if ksp:
        print(errors_ksp)
        convergence_rate_ksp = np.polyfit(np.log(voxel_sizes), np.log(errors_ksp), 1)[0]
        print(f"{convergence_rate_ksp=}")


        assert convergence_rate_ksp >= 1.5
    

    
        
if __name__ == "__main__":
    test_fft_poisson_convergence()
