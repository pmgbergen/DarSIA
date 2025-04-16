from ot import dist
from ot.bregman import sinkhorn2
from ot.bregman import empirical_sinkhorn2


class WassersteinDistanceSinkhorn(darsia.EMD):
    """
    Class based on the Sinkhorn algorithm to compute the Wasserstein distance
    """
    def __init__(self, 
                grid: darsia.Grid,
                options: dict = {}):
        
        # Cache geometrical infos
        self.grid = grid
        """darsia.Grid: grid"""

        self.voxel_size = grid.voxel_size
        """np.ndarray: voxel size"""

        # Cache solver options
        self.options = options
        """dict: options for the solver"""

        self.sinkhorn_regularization = self.options.get("sinkhorn_regularization", 1e-1)
        """float: regularization parameter"""

        self.sinkhorn_algorithm = options.get("sinkhorn_algorithm", "sinkhorn")
        """ str: algorithm to use for the Sinkhorn algorithm"""       
        """ Available algorithms are:
        sinkhorn,sinkhorn_log, greenkhorn, sinkhorn_stabilized, sinkhorn_epsilon_scaling,
        """

        self.num_iter = self.options.get("num_iter", 100)
        """ int: max number of iterations"""

        self.only_non_zeros = options.get("only_non_zeros", True)
        """ bool: consider only non-zero pixels"""
        
        self.verbose = options.get("verbose", True)
        """ bool: verbosity"""

        self.store_cost_matrix = options.get("store_cost_matrix", False)
        """ bool: store the cost matrix"""
        if self.store_cost_matrix:
            self.M = dist(self.grid.cell_centers, self.grid.cell_centers, metric='euclidean')


        self.geomloss_scaling = self.options.get("geomloss_scaling", 0.5)
        """float: scaling factor for eps"""

        # TODO: rewrite
        N = self.grid.shape
        x_i = []
        for i, N in enumerate(self.grid.shape):
            x_i.append(np.arange(N)*self.grid.voxel_size[i]+ self.grid.voxel_size[i]/2)

        self.cc_xyz = np.meshgrid(*x_i, indexing="ij")
        self.cc = np.vstack([c.ravel() for c in self.cc_xyz]).T


    def support(self, img: darsia.Image) -> np.ndarray:
        """
        Return the indices of the non-zero pixels in the image.

        Args:
            img (darsia.Image): image

        Returns: 
            np.ndarray: support

        """
        # flatten the image
        img_flat = img.img.ravel()

        # return the indices of the non-zero pixels
        return np.where(img_flat > 0)
    
    def interpolate_kantorovich_potentials(self, 
                            support_1, img1, pot_1,
                            support_2, img2, pot_2) -> (np.ndarray, np.ndarray): 
        """
        When we work only on the "support" of the images,
        we need to extend the kantorovich potentials to the whole domain.
        We can do it using the entropic interpolation 
        (see Eq 3.7 in https://hal.science/hal-02539799/document).
        It should given the same result of the Sinkhorn algorithm
        passing the whole images.
        """

        coord_support_1 = self.cc[support_1]
        coord_support_2 = self.cc[support_2]

        pot_1_whole = np.zeros(self.cc.shape[0])
        for i, coord_center in enumerate(self.cc):
            M_i = dist(coord_center, coord_support_2, metric='euclidean')
            pot_1_whole[i] = -self.sinkhorn_regularization * np.log(
                np.dot(
                    np.exp(pot_2[i]- M_i )/self.sinkhorn_regularization),
                    img2)
            
        pot_2_whole = np.zeros(self.cc.shape[0])
        for i, coord_center in enumerate(self.cc):
            M_i = dist(coord_center, coord_support_1, metric='euclidean')
            pot_2_whole[i] = -self.sinkhorn_regularization * np.log(
                np.dot(
                    np.exp(pot_1 - M_i )/self.sinkhorn_regularization),
                    img1)
            
        return pot_1_whole, pot_2_whole

    def img2pythorch(self, img: darsia.Image) -> torch.Tensor:
        """
        Convert a darsia image to a pythorch tensor suitable for geomloss.sinkhorn_image
        that that takes tensor with (nimages, nchannels, *spatial dimensions)
        """
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
         
        return torch.from_numpy(img.img).type(dtype).view(1,1,*img.img.shape)

        
    def __call__(self, img_1: darsia.Image, img_2: darsia.Image) -> float:
        """
        Earth mover's distance between images with same total sum.

        Args:
            img_1 (darsia.Image): image 1
            img_2 (darsia.Image): image 2

        Returns:
            float or array: distance between img_1 and img_2.

        """
        # FIXME investigation required regarding resize preprocessing...
        # Preprocess images
        preprocessed_img_1 = img_1
        preprocessed_img_2 = img_2

        # Compatibilty check
        self._compatibility_check(preprocessed_img_1, preprocessed_img_2)

        # Only non-zero pixels are considered
        if self.only_non_zeros:
            support_1 = np.where(preprocessed_img_1.img.flatten("F")>0)
            support_2 = np.where(preprocessed_img_2.img.flatten("F")>0)

            non_zero_img1 = preprocessed_img_1.img.flatten("F")[support_1]*np.prod(self.voxel_size)
            non_zero_img2 = preprocessed_img_2.img.flatten("F")[support_2]*np.prod(self.voxel_size)
            
            coord_support_1 = self.cc[support_1]
            coord_support_2 = self.cc[support_2]
        else:
            non_zero_img1 = preprocessed_img_1.img.flatten("F") * np.prod(self.voxel_size)
            non_zero_img2 = preprocessed_img_2.img.flatten("F") * np.prod(self.voxel_size)
            coord_support_1 = self.cc
            coord_support_2 = self.cc


        # Compute the distance
        if "empirical" in self.sinkhorn_algorithm:
            distance, log = empirical_sinkhorn2(
                coord_support_1, # coordinate of non-zero pixels in image 1
                coord_support_2, # coordinate of non-zero pixels in image 2
                reg=self.sinkhorn_regularization,
                a=non_zero_img1,
                b=non_zero_img2, 
                metric='euclidean', # distance metric 
                numIterMax=self.num_iter,
                isLazy=False,  #boolean, 
                # If True, then only calculate the cost matrix by block and return
                # the dual potentials only (to save memory). If False, calculate full
                #cost matrix and return outputs of sinkhorn function.
                verbose=self.verbose, 
                log=True, # return ierr and log
                )
            self.niter = log['niter']
            self.kantorovich_potential_source = log['u']
            self.kantorovich_potential_target = log['v']
        elif "geomloss_sinkhorn_images" == self.sinkhorn_algorithm:
            raise NotImplementedError("geomloss_sinkhorn_images not implemented yet")
            
            # package to be imported
            from geomloss.sinkhorn_images import sinkhorn_divergence
            import torch
            use_cuda = torch.cuda.is_available()
            
            # convert the images to pythorch tensors
            torch_img1 = self.img2pythorch(preprocessed_img_1)
            torch_img2 = self.img2pythorch(preprocessed_img_2)
            
            distance, log = sinkhorn_divergence(
                torch_img1,
                torch_img2,
                p=1,
                blur=None,
                reach=None,
                axes=self.grid.voxel_size*self.grid.shape,
                scaling=0.5,
                cost=None,
                debias=True,
                potentials=True,
                verbose=True,
                )
            # sinkhorn and sinkhorn_log work with the potentials    
            self.niter = log['niter']
            self.kantorovich_potential_source = log['u']
            self.kantorovich_potential_target = log['v']
        
        elif "geomloss_sinkhorn_samples" in self.sinkhorn_algorithm:
            #
            # Explicit sparsity of images 
            # 
            from geomloss.sinkhorn_divergence import epsilon_schedule
            from geomloss.sinkhorn_samples import sinkhorn_tensorized
            import torch
            
            use_cuda = torch.cuda.is_available()
            dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
            
            # first 1 in view() is for passing the batch size
            point_img1 = torch.from_numpy(non_zero_img1).type(dtype).view(1,len(non_zero_img1))
            point_img2 = torch.from_numpy(non_zero_img2).type(dtype).view(1,len(non_zero_img2))
            x = torch.from_numpy(coord_support_1).type(dtype).view(1,*coord_support_1.shape)
            y = torch.from_numpy(coord_support_2).type(dtype).view(1,*coord_support_2.shape)
            
            diameter = np.linalg.norm(self.grid.voxel_size*self.grid.shape)
            blur = self.sinkhorn_regularization
            scaling = self.geomloss_scaling
            p_exponent = 1
            f_torch, g_torch = sinkhorn_tensorized(
                point_img1,#a 
                x,#x
                point_img2, #b
                y,#y
                p=p_exponent,
                blur=blur, #blur
                reach=None,
                diameter=diameter, #diameter
                scaling=0.5, # reduction of the regularization
                cost=None,
                debias=False,
                potentials=True,
            )

            f = f_torch.detach().cpu().numpy().flatten()
            g = g_torch.detach().cpu().numpy().flatten()
            distance = np.dot(f, non_zero_img1) + np.dot(g, non_zero_img2) 
            print(f"distance: {distance=} {f.shape=} {g.shape=}")


            # it is not clear to me how to get the number of iterations
            # what about the residual of the marginals?
            self.niter = len(epsilon_schedule(p_exponent, diameter, blur, scaling)) 
            self.kantorovich_potential_source = f
            self.kantorovich_potential_target = g


        else:
            M = dist(coord_support_1, coord_support_2, metric='euclidean')
            distance, log = sinkhorn2(
                non_zero_img1, non_zero_img2, M, 
                self.sinkhorn_regularization,
                method=self.sinkhorn_algorithm,
                numItermax=self.options["num_iter"],
                stopThr=1e-8,#self.options["tol_residual"],
                verbose=self.verbose,
                log=True)

            
            if self.sinkhorn_algorithm == 'sinkhorn_stabilized':
                self.niter = log['n_iter']
                self.kantorovich_potential_source = np.exp(log['logu'])
                self.kantorovich_potential_target = np.exp(log['logv'])
            else:
                # sinkhorn and sinkhorn_log work with the potentials    
                self.niter = log['niter']
                self.kantorovich_potential_source = log['u']
                self.kantorovich_potential_target = log['v']


        print("distance: ", distance)
        exit()
        import matplotlib.pyplot as pl

        pl.figure(1)
        pl.plot(coord_support_1, "+b", label="Source samples")
        pl.plot(coord_support_2, "xr", label="Target samples")
        pl.legend(loc=0)
        pl.title("Source and target distributions")

        pl.figure(2)
        pl.imshow(M, interpolation="nearest")
        pl.title("Cost matrix M")    
        pl.show()


        if self.only_non_zeros:
            self.kantorovich_potential_source, self.kantorovich_potential_target = self.interpolate_kantorovich_potentials(
                support_1, preprocessed_img_1, self.kantorovich_potential_source,
                support_2, preprocessed_img_2, self.kantorovich_potential_target
            )



        self.kantorovich_potential_source = self.kantorovich_potential_source.reshape(img_1.shape, order="F")
        self.kantorovich_potential_target = self.kantorovich_potential_target.reshape(img_2.shape, order="F")





        info = {
            "converged" : True,
            "niter": self.niter,
            "kantorovich_potential_source": self.kantorovich_potential_source,
            "kantorovich_potential_target": self.kantorovich_potential_target,
        }

        return distance, info

class WassersteinDistanceDMK(darsia.EMD):
    """
    This contains the implementation of the GproxPDHG algorithm
    described in "SOLVING LARGE-SCALE OPTIMIZATION PROBLEMS WITH
    A CONVERGENCE RATE INDEPENDENT OF GRID SIZE"
    """

    def __init__(self, grid, options) -> None:
        # Cache geometrical infos
        self.grid = grid
        """darsia.Grid: grid"""

        self.voxel_size = grid.voxel_size
        """np.ndarray: voxel size"""

        # Cache solver options
        self.options = options
        """dict: options for the solver"""

        self.verbose = self.options.get("verbose", False)
        """bool: verbosity"""

        self.div_contraint_tol = self.options.get("",1e-6)


        # Allocate space for main and auxiliary variables
        self.transport_density = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: flux"""

        self.flux = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: flux"""


        self.pressure = np.zeros(self.grid.num_cells, dtype=float)
        """ np.ndarray: pressure of the poisson problem -div(p) = f = img1- img2"""

        self.flux_old = copy.deepcopy(self.flux)
        self.transport_density_old = copy.copy(self.transport_density)
        self.pressure_old = copy.deepcopy(self.pressure)



        self._setup_discretization()

        self.kappa = self.options.get("kappa", np.ones(self.grid.shape,dtype=float))
        """np.ndarray: kappa"""
        self.kappa_faces = darsia.cell_to_face_average(self.grid, self.kappa, mode="harmonic")
        """np.ndarray: kappa on faces"""

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators.

        Add linear contribution of the optimality conditions of the Newton linearization.

        """

        # ! ---- Discretization operators ----

        if self.grid.voxel_size[0] != self.grid.voxel_size[1]:
            raise ValueError("The grid must be isotropic")
        self.h = self.grid.voxel_size[0]


        self.div = darsia.FVDivergence(self.grid).mat / self.h
        """sps.csc_matrix: divergence operator: flat fluxes -> flat pressures"""
        
        self.grad = darsia.FVDivergence(self.grid).mat.T / self.h**2


        self.mass_matrix_cells = darsia.FVMass(self.grid).mat
        """sps.csc_matrix: mass matrix on cells: flat pressures -> flat pressures"""

                

        

    def setup_elliptic_solver(self, transport_density, rtol=1e-6):
        """
        
        """
        self.weighted_Laplacian_matrix = self.div * sps.diags(transport_density) / self.h * self.grad

        # Define CG solver
        kernel = np.ones(self.grid.num_cells, dtype=float) / np.sqrt(self.grid.num_cells)
        weighted_Poisson_solver = darsia.linalg.KSP(self.weighted_Laplacian_matrix, 
                                                nullspace=[kernel],
                                                appctx={})
        
        weighted_Poisson_ksp_ctrl = {
                        "ksp_type": "cg",
                        "ksp_rtol": rtol,
                        "ksp_maxit": 100,
                        "pc_type": "hypre",
                        #"ksp_monitor": None,
        }
        weighted_Poisson_solver.setup(weighted_Poisson_ksp_ctrl)
        
        return weighted_Poisson_solver
        
    def _solve(self, 
               flat_mass_diff: np.ndarray,
               ) -> tuple[float, np.ndarray, dict]:
        """Solve the Beckman problem using Newton's method.

        Args:
            flat_mass_diff (np.ndarray): difference of mass distributions

        Returns:
            tuple: distance, solution, info

        """
        # Setup time and memory profiling
        tic = time.time()
        tracemalloc.start()

        # Solver parameters. By default tolerances for increment and distance are
        # set, such that they do not affect the convergence.
        num_iter = self.options.get("num_iter", 400)
        tol_residual = self.options.get("tol_residual", np.finfo(float).max)
        tol_increment = self.options.get("tol_increment", np.finfo(float).max)
        tol_distance = self.options.get("tol_distance", np.finfo(float).max)
            
        
        # Initialize Newton iteration with Darcy solution for unitary mobility
        self.transport_density[:] = 1.0 / self.kappa_faces
        Poisson_solver = self.setup_elliptic_solver(self.transport_density)
        pressure = Poisson_solver.solve(flat_mass_diff)
        gradient_pressure = self.grad.dot(pressure)
        distance = self.compute_dual(pressure, flat_mass_diff) 
        flux = self.transport_density * gradient_pressure

        # Initialize container for storing the convergence history
        convergence_history = {
            "distance": [],
            "residual": [],
            "flux_increment": [],
            "distance_increment": [],
            "timing": [],
            "run_time": [],
        }

        # Print  header for later printing performance to screen
        # - distance
        # - distance increment
        # - flux increment
        # - residual
        if self.verbose:
            print(
                "DMK iter. \t| W^1 \t\t| Δ W^1 \t| Δ flux \t| residual",
                "\n",
                """---------------|---------------|---------------|---------------|"""
                """---------------""",
            )


        self.iter = 0
        deltat = 0.01
        # DMK iterations
    
        while self.iter <= num_iter:
            self.flux_old[:] = flux[:]
            self.pressure_old[:] = pressure[:]
            self.transport_density_old[:] = self.transport_density[:]
            distance_old = distance

            # start update 
            start = time.time()

            # udpate transport density
            update = self.transport_density * ( np.abs(gradient_pressure)  - self.kappa_faces)
            deltat = min(deltat * 1.05, 0.5)
            self.transport_density += deltat * update
            min_tdens = 1e-10
            self.transport_density[self.transport_density < min_tdens] = min_tdens
            # udpate potential
            Poisson_solver = self.setup_elliptic_solver(self.transport_density)
            pressure = Poisson_solver.solve(flat_mass_diff)
            gradient_pressure = self.grad.dot(pressure)

            # update flux
            flux = self.transport_density * gradient_pressure

            self.iter += 1
            self.iter_cpu = time.time() - start

            
            # int_ pot f = int_ pot div poisson_pressure = int_ grad pot \cdot \nabla poisson_pressure 
            self.dual_value = self.compute_dual(pressure, flat_mass_diff) 
            self.primal_value = self.compute_primal(flux)
            self.duality_gap = abs(self.dual_value - self.primal_value)
            distance = self.primal_value

            distance_increment = abs( distance - distance_old) / distance
            if distance_increment < tol_distance:
                break


            residual = np.linalg.norm(( gradient_pressure ** 2 - 1.0) * self.transport_density)
            if residual < tol_residual:
                break

            flux_increment = np.linalg.norm(flux - self.flux_old) / np.linalg.norm(self.flux_old)  
            if flux_increment < tol_increment:
                break
            
            
            if self.verbose :
                print(
                    f"it: {self.iter:03d}" +
                    f" gap={self.duality_gap:.1e}" +
                    f" dual={self.dual_value:.3e} primal={self.primal_value:.3e}" +
                    f" cpu={self.iter_cpu:.3f}"+
                    f" max grad={np.max(gradient_pressure):.2e}"
                    f" {np.min(self.transport_density):.2e}<=TDENS<={np.max(self.transport_density):.2e}"
                    ) 

            
            # Compute the error and store as part of the convergence history:
            # 0 - full residual (Newton interpretation)
            # 1 - flux increment (fixed-point interpretation)
            # 2 - distance increment (Minimization interpretation)

            # Update convergence history
            convergence_history["distance"].append(distance)
            convergence_history["residual"].append(residual)
            convergence_history["flux_increment"].append(flux_increment)
            convergence_history["distance_increment"].append(distance_increment)
            #convergence_history["timing"].append(stats_i)

            # Extract current total run time
            #current_run_time = self._analyze_timings(convergence_history["timing"])[
            #    "total"
            #]
            #convergence_history["run_time"].append(current_run_time)

            # Print performance to screen
            # - distance
            # - distance increment
            # - flux increment
            # - residual
            if self.verbose:
                print(
                    f"""Iter. {self.iter} \t| {distance:.6e} \t| """
                    + f"""{distance_increment:.6e} \t| {flux_increment:.6e} \t| """
                    + f"""{residual:.6e}"""
                )

        # Summarize profiling (time in seconds, memory in GB)
        #total_timings = self._analyze_timings(convergence_history["timing"])
        #peak_memory_consumption = tracemalloc.get_traced_memory()[1] / 10**9



        # Define performance metric
        info = {
            "converged": self.iter < num_iter,
            "number_iterations": self.iter,
            "convergence_history": convergence_history,
            #"timings": total_timings,
            #"peak_memory_consumption": peak_memory_consumption,
        }

        solution = np.concatenate([pressure, self.transport_density])

        return distance, solution, info

    def compute_dual(self, pressure, forcing):
        """
        Compute the value of the dual functional
        $\int_{\Domain} pot (f^+ - f^-)$
        """
        return np.dot(pressure, forcing) *  np.prod(self.grid.voxel_size)
    
    def compute_primal(self,flux):
        """
        Compute the value of the primal functional
        $\int_{\Domain} | flux | $
        """
        return np.sum(np.abs(flux)) * np.prod(self.grid.voxel_size) 

    def __call__(
        self,
        img_1: darsia.Image,
        img_2: darsia.Image,
    ) -> float:
        """L1 Wasserstein distance for two images with same mass.

        NOTE: Images need to comply with the setup of the object.

        Args:
            img_1 (darsia.Image): image 1, source distribution
            img_2 (darsia.Image): image 2, destination distribution

        Returns:
            float: distance between img_1 and img_2.
            dict (optional): solution
            dict (optional): info

        """

        # Compatibilty check
        assert img_1.scalar and img_2.scalar
        self._compatibility_check(img_1, img_2)

        # Determine difference of distributions and define corresponding rhs
        mass_diff = img_2.img - img_1.img
        flat_mass_diff = np.ravel(mass_diff, "F")
        
        # Main method
        distance, solution, info = self._solve(flat_mass_diff)

        flat_pressure = solution[:self.grid.num_cells]
        flat_gradient = self.div.T.dot(flat_pressure)
        flat_tdens = solution[-self.grid.num_faces:]
        flat_flux = flat_tdens * flat_gradient
        

        flux = darsia.face_to_cell(self.grid, flat_flux)
        temp = np.hstack([flat_tdens,flat_tdens]) 
        transport_density = darsia.face_to_cell(self.grid, temp)[:,:,0]

        
        pressure = flat_pressure.reshape(self.grid.shape, order="F")
        
        

        
        # Return solution
        return_info = self.options.get("return_info", False)
        if return_info:
            info.update(
                {
                    "grid": self.grid,
                    "mass_diff": mass_diff,
                    "flux": flux,
                    "pressure": pressure,
                    "transport_density": transport_density,
                    "src": img_1,
                    "dst": img_2,
                }
            )
            return distance, info
        else:
            return distance



class WassersteinDistanceGproxPGHD(darsia.EMD):
    """
    This contains the implementation of the GproxPDHG algorithm
    described in "SOLVING LARGE-SCALE OPTIMIZATION PROBLEMS WITH
    A CONVERGENCE RATE INDEPENDENT OF GRID SIZE"
    """

    def __init__(self, grid, options) -> None:
        # Cache geometrical infos
        self.grid = grid
        """darsia.Grid: grid"""

        self.voxel_size = grid.voxel_size
        """np.ndarray: voxel size"""

        # Cache solver options
        self.options = options
        """dict: options for the solver"""

        self.verbose = self.options.get("verbose", False)
        """bool: verbosity"""

        # Allocate space for main and auxiliary variables
        self.flux = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: flux"""

        self.poisson_pressure = np.zeros(self.grid.num_cells, dtype=float)
        """ np.ndarray: pressure of the poisson problem -div(p) = f = img1- img2"""

        self.gradient_poisson = np.zeros(self.grid.num_faces, dtype=float)
        """ varaible storing the gradient of the poisson problem"""

        self.rhs_forcing = np.zeros(self.grid.num_cells, dtype=float)
        """ variable storing the right hand side of the poisson problem"""


        

        self.div_free_flux = np.zeros(self.grid.num_faces, dtype=float)
        """ np.ndarray: divergence free flux"""

        self.u = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: u"""

        self.p = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: $p^{n}$"""


        self.p_bar = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: $\bar{p}$"""

        self.new_p = np.zeros(self.grid.num_faces, dtype=float)
        """np.ndarray: $p^{n+1}$"""

        self._setup_discretization()

    def _setup_discretization(self) -> None:
        """Setup of fixed discretization operators.

        Add linear contribution of the optimality conditions of the Newton linearization.

        """

        # ! ---- Discretization operators ----
        if self.grid.voxel_size[0] != self.grid.voxel_size[1]:
            raise ValueError("The grid must be isotropic")
        self.h = self.grid.voxel_size[0]
        

        self.div = darsia.FVDivergence(self.grid).mat #/ self.h**2
        """sps.csc_matrix: divergence operator: flat fluxes -> flat pressures"""
    
        self.grad = darsia.FVDivergence(self.grid).mat.T #* self.h**4
        """sps.csc_matrix: grad operator:  flat pressures -> falt fluxes"""


        self.mass_matrix_cells = darsia.FVMass(self.grid).mat
        """sps.csc_matrix: mass matrix on cells: flat pressures -> flat pressures"""

        self.Laplacian_matrix = self.div * self.grad  #/ self.h**2

        # Define CG solver
        kernel = np.ones(self.grid.num_cells, dtype=float) / np.sqrt(self.grid.num_cells)
        self.Poisson_solver = darsia.linalg.KSP(self.Laplacian_matrix, 
                                                nullspace=[kernel],
                                                appctx={})
        
        self.Poisson_ksp_ctrl = {
                        "ksp_type": "cg",
                        "ksp_rtol": 1e-6,
                        "ksp_maxit": 100,
                        "pc_type": "hypre",
                        #"ksp_monitor": None,
        }
        self.Poisson_solver.setup(self.Poisson_ksp_ctrl)


    def residual(self, rhs: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """Compute the residual of the solution.

        Args:
            rhs (np.ndarray): right hand side
            solution (np.ndarray): solution

        Returns:
            np.ndarray: residual

        """
        return self.optimality_conditions(rhs, solution)
    

    def leray_projection(self, p: np.ndarray) -> np.ndarray:
        """Leray projection of a vector fiels

        Args:
            p (np.ndarray): pressure

        Returns:
            np.ndarray: divergence free flux

        """
        rhs = self.div.dot(p)
        poisson_solution = self.Poisson_solver.solve(rhs) #/ self.h**2
        return p - self.grad.dot(poisson_solution) 
    
    def compute_pressure(self, flux: np.ndarray, forcing: np.array) -> np.ndarray:
        """Compute the pressure from the flux.

        Args:
            flux (np.ndarray): flux

        Returns:
            np.ndarray: pressure

        """
        self.Laplacian_matrix = self.div * sps.diags(np.abs(flux), dtype=float) * self.grad / self.h **4

        # Define CG solver
        kernel = np.ones(self.grid.num_cells, dtype=float) / np.sqrt(self.grid.num_cells)
        self.weighted_Poisson_solver = darsia.linalg.KSP(self.Laplacian_matrix, 
                                                nullspace=[kernel],
                                                appctx={})
        
        self.weighted_Poisson_ksp_ctrl = {
                        "ksp_type": "cg",
                        "ksp_rtol": 1e-6,
                        "ksp_maxit": 100,
                        "pc_type": "hypre",
                        #"ksp_monitor": None,
        }
        self.weighted_Poisson_solver.setup(self.weighted_Poisson_ksp_ctrl)
        pressure = self.weighted_Poisson_solver.solve(forcing)
        self.weighted_Poisson_solver.kill()
        return pressure
    
    def _solve(self, flat_mass_diff: np.ndarray) -> tuple[float, np.ndarray, dict]:
        """Solve the Beckman problem using Newton's method.

        Args:
            flat_mass_diff (np.ndarray): difference of mass distributions

        Returns:
            tuple: distance, solution, info

        """
        # Setup time and memory profiling
        tic = time.time()
        tracemalloc.start()

        # Solver parameters. By default tolerances for increment and distance are
        # set, such that they do not affect the convergence.
        num_iter = self.options.get("num_iter", 100)
        tol_residual = self.options.get("tol_residual", np.finfo(float).max)
        tol_increment = self.options.get("tol_increment", np.finfo(float).max)
        tol_distance = self.options.get("tol_distance", np.finfo(float).max)

        
        # Initialize Newton iteration with Darcy solution for unitary mobility
        poisson_pressure = self.Poisson_solver.solve(flat_mass_diff)
        gradient_poisson_pressure = self.grad.dot(poisson_pressure)

        # Initialize distance in case below iteration fails
        new_distance = 0

        # Initialize container for storing the convergence history
        convergence_history = {
            "distance": [],
            "residual": [],
            "flux_increment": [],
            "distance_increment": [],
            "timing": [],
            "run_time": [],
        }

        # Print  header for later printing performance to screen
        # - distance
        # - distance increment
        # - flux increment
        # - residual
        if self.verbose:
            print(
                "PD iter. \t| W^1 \t\t| Δ W^1 \t| Δ flux \t| residual",
                "\n",
                """---------------|---------------|---------------|---------------|"""
                """---------------""",
            )


        p_bar = self.p_bar
        u = self.u
        p = self.p
        new_p = self.new_p
        self.iter = 0
        # PDHG iterations
        while self.iter <= num_iter:
            #
            start = time.time()
            tau = self.options.get("tau", 1.0)
            sigma =  self.options.get("sigma", 1.0) 
            if self.iter > 0:
                p[:] = new_p[:]

            # eq 3.14
            div_free = self.leray_projection(p_bar)
            u -= tau * div_free

            # new flux
            self.flux = u + gradient_poisson_pressure

            flat_pressure = self.compute_pressure(self.flux, flat_mass_diff)
            grad_pressure = self.grad.dot(flat_pressure)
            print(f"MAX GRAD {np.max(abs(grad_pressure))}")
            
            # eq 3.15
            sigma_vel = p + sigma * self.flux
            #print(sigma_vel)
            new_p[:] = sigma_vel[:]
            abs_sigma_vel = np.abs(sigma_vel)
            greater_than_1 = np.where(abs_sigma_vel > 1)
            new_p[greater_than_1] /= abs_sigma_vel[greater_than_1] # normalize too +1 or -1
            #print(new_p)


            # eq 3.16
            p_bar[:] = 2 * new_p[:] - p[:]

            # int_ pot f = int_ pot div poisson_pressure = int_ grad pot \cdot \nabla poisson_pressure 
            self.dual_value = self.compute_dual(p, gradient_poisson_pressure) 
            self.primal_value = self.compute_primal(self.flux)
            self.duality_gap = abs(self.dual_value - self.primal_value)
            distance = self.primal_value


            self.iter += 1

            self.iter_cpu = time.time() - start
            #if self.verbose :
            print(
                    f"it: {self.iter:03d}" +
                    f" gap={self.duality_gap:.1e}" +
                    f" dual={self.dual_value:.3e} primal={self.primal_value:.3e}" +
                    f" cpu={self.iter_cpu:.3f}") 

            #convergence_history["distance"].append(new_distance)
            #convergence_history["residual"].append(np.linalg.norm(residual_i, 2))
            
            # Compute the error and store as part of the convergence history:
            # 0 - full residual (Newton interpretation)
            # 1 - flux increment (fixed-point interpretation)
            # 2 - distance increment (Minimization interpretation)

            # Update convergence history
            #convergence_history["distance"].append(new_distance)
            #convergence_history["residual"].append(np.linalg.norm(residual_i, 2))
            #convergence_history["flux_increment"].append(
            #    np.linalg.norm(increment[self.flux_slice], 2)
            #)
            #convergence_history["distance_increment"].append(
            #    abs(new_distance - old_distance)
            #)
            #convergence_history["timing"].append(stats_i)

            # Extract current total run time
            #current_run_time = self._analyze_timings(convergence_history["timing"])[
            #    "total"
            #]
            #convergence_history["run_time"].append(current_run_time)

            # Print performance to screen
            # - distance
            # - distance increment
            # - flux increment
            # - residual
            if False:#self.verbose:
                distance_increment = convergence_history["distance_increment"][-1]
                flux_increment = (
                    convergence_history["flux_increment"][-1]
                    / convergence_history["flux_increment"][0]
                )
                residual = (
                    convergence_history["residual"][-1]
                    / convergence_history["residual"][0]
                )
                print(
                    f"""Iter. {iter} \t| {new_distance:.6e} \t| """
                    f"""{distance_increment:.6e} \t| {flux_increment:.6e} \t| """
                    f"""{residual:.6e}"""
                )

        # Summarize profiling (time in seconds, memory in GB)
        #total_timings = self._analyze_timings(convergence_history["timing"])
        #peak_memory_consumption = tracemalloc.get_traced_memory()[1] / 10**9



        # Define performance metric
        info = {
            "converged": self.iter < num_iter,
            "number_iterations": self.iter,
            "convergence_history": convergence_history,
            #"timings": total_timings,
            #"peak_memory_consumption": peak_memory_consumption,
        }

        return distance, self.flux, info 
    
    def compute_dual(self,p,gradient_poisson):
        """
        Compute the value of the dual functional
        $\int_{\Domain} pot (f^+ - f^-)$
        $=\int_{\Domain} pot -div(poisson)$
        $\int_{\Domain} \nabla pot \cdot \nabla poisson$
        $\int_{\Domain} p \cdot \nabla poisson$
        """
        return np.dot(p, gradient_poisson) *  np.prod(self.grid.voxel_size)
    
    def compute_primal(self,flux):
        """
        Compute the value of the primal functional
        $\int_{\Domain} | flux | $
        """
        return np.sum(np.abs(flux)) * np.prod(self.grid.voxel_size)

    def __call__(
        self,
        img_1: darsia.Image,
        img_2: darsia.Image,
    ) -> float:
        """L1 Wasserstein distance for two images with same mass.

        NOTE: Images need to comply with the setup of the object.

        Args:
            img_1 (darsia.Image): image 1, source distribution
            img_2 (darsia.Image): image 2, destination distribution

        Returns:
            float: distance between img_1 and img_2.
            dict (optional): solution
            dict (optional): info

        """

        # Compatibilty check
        assert img_1.scalar and img_2.scalar
        self._compatibility_check(img_1, img_2)

        # Determine difference of distributions and define corresponding rhs
        mass_diff = img_2.img - img_1.img
        flat_mass_diff = np.ravel(mass_diff, "F")


        # Main method
        distance, solution, info = self._solve(flat_mass_diff)

        flux = darsia.face_to_cell(self.grid, solution)
        flat_pressure = self.compute_pressure(solution, flat_mass_diff)
        temp = np.hstack([np.abs(solution),np.abs(solution)])
        transport_density = darsia.face_to_cell(self.grid, temp)[:,:,0]
        pressure = flat_pressure.reshape(self.grid.shape, order="F")
        

        
        # Return solution
        return_info = self.options.get("return_info", False)
        if return_info:
            info.update(
                {
                    "grid": self.grid,
                    "mass_diff": mass_diff,
                    "flux": flux,
                    "pressure": pressure,
                    "transport_density": transport_density,
                    "src": img_1,
                    "dst": img_2,
                }
            )
            return distance, info
        else:
            return distance

