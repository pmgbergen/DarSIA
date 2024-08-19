"""Example for Wasserstein computations moving a square to another location."""

import numpy as np
import pytest
from scipy.ndimage import zoom
from scipy.optimize import minimize_scalar
import darsia


import sys
method = sys.argv[1]
try:
    nref = int(sys.argv[2])
except:
    nref = 0

try:
    permeability_layer = float(sys.argv[3])
except:
    permeability_layer = 1.0


def test_case(rows=8, cols=8, 
            x_src=1, y_src=1,
            x_dst=6, y_dst=6,
            permeability_value = 1.0, nref=0):
    
    h=1.0/8.0

    # Coarse src image
    src_square_2d = np.zeros((rows, cols), dtype=float)
    src_square_2d[x_src:x_src+1, y_src:y_src+1] = 1
    src_square_2d = zoom(src_square_2d, 2**nref, order=0)
    meta_2d = {"width": 1, "height": 1, "space_dim": 2, "scalar": True}
    src_image_2d = darsia.Image(src_square_2d, **meta_2d)

    # Coarse dst image
    dst_squares_2d = np.zeros((rows, cols), dtype=float)
    dst_squares_2d[x_dst:x_dst+1, y_dst:y_dst+1] = 1
    dst_squares_2d = zoom(dst_squares_2d, 2**nref, order=0)
    dst_image_2d = darsia.Image(dst_squares_2d, **meta_2d)

    # Rescale
    shape_meta_2d = src_image_2d.shape_metadata()
    geometry_2d = darsia.Geometry(**shape_meta_2d)
    src_image_2d.img /= geometry_2d.integrate(src_image_2d)
    dst_image_2d.img /= geometry_2d.integrate(dst_image_2d)


    bar_src = [(x_src+0.5)/rows, (y_src+0.5)/cols]
    bar_dst = [(x_dst+0.5)/rows, (y_dst+0.5)/cols]
    intermidate_points = (bar_src + bar_dst)/2


    a=0.5-bar_src[0] # x distance from the source to the layer
    b=1/8   # half of the width of the layer
    l=bar_src[1]-intermidate_points[1] # y_distance from the source to intermidate point

    
    # Compute the incidence point of the ray starting at the source and
    # going to the intermidate point.
    # Varaible name from snell'law Wikipedia page
    from scipy.optimize import toms748

    def h(x):
        """
        Rearranged derivative of travel time from the source to the intermidate point
        """
        return x**2 * ( (l-x)**2+b**2) - permeability_layer **2 * (l-x)**2 * (x**2+a**2)
               
    def L(x):
        " length of the shortest path from the source center to the intermidate point"
        L = np.sqrt(x**2+a**2) + np.sqrt((l-x)**2+b**2)
        return 2 * L 


    x_opt = toms748(h,0,l)
    print(f"{x_opt=} {x_opt+1.5/8=} {h(x_opt)=} L={L(x_opt)}")
    
    theta_1 = np.arctan(x_opt/a)
    theta_2 = np.arctan((l-x_opt)/b)

    # check result 
    if not np.close(np.sin(theta_1)/np.sin(theta_2),permeability_value):
        raise ValueError("Angles do not satisfiy Snell Law")


    opt_direction_1 = [np.cos(theta_1),-np.sin(theta_1)]
    opt_direction_2 = [np.cos(theta_2),-np.sin(theta_2)]

    if np.close(theta_1<np.pi/4):
        L_max = 1/8 / np.cos(theta_1)
        tdens_max = L_max * src_value

        def tdens_1(x,y):
            if 0<(x-x_src/8)<1/8 and 0<(y-y_src)<1/8:
                # compute the distance to the upper/left boundary along
                # the direction 
                dist = (x_src * h - x) / np.sin(theta_1)
            return dist
    






        Lx = abs(x_src-x_dst)/8
        Ly = abs(y_src-y_dst)/8
        L=np.sqrt((x_src-x_dst)**2+(y_src-y_dst)**2)/8



def test_case(ndivx=3, ndivy=3, 
            x_src=1, y_src=1,
            x_dst=6, y_dst=6,
            permeability_layer = 1.0, nref=0):
    """"
    Transport a square of size 1/8 from one location to another in a 2D image.
    The is a layer of a given permeability in the middle of the domain (3/8<x<5/8)
    The source source is contained in the layer (0<x<3/8).
    The destination is outside the layer (5/8<x<1).
    The geodesic is a piecewise straight liner that goes from the source to the destination.
    Its path is determined by the Snell's law.

    Parameters
    ----------
    rows : int
        Number of rows in the image.
    cols : int
        Number of columns in the image.
    x_src : int
        x grid-coordinate of the source (0<=x_src<rows*3/8).
    y_src : int
        y grid-coordinate of the source. (0<=y_src<cols).
    x_dst : int
        x grid-coordinate of the destination. (rows*5/8<=x_dst<rows).
    y_dst : int
        y grid-coordinate of the destination. (0<=y_dst<cols).
    permeability_value : float
        Permeability value of the layer.
    nref : int
        Number of refinements of the image.
    """
    if ndivx < 3 or ndivy < 3:
        raise ValueError("ndivx and ndivy must be at least 3.")
    cols = 2**ndivx
    rows = 2**ndivy


    #  Coarse src image
    src_square_2d = np.zeros((rows, cols), dtype=float)
    src_square_2d[x_src:x_src+1, y_src:y_src+1] = 1
    src_square_2d.transpose() # transpose to match the image orientation
    meta_2d = {"width": 1, "height": 1, "space_dim": 2, "scalar": True}
    src_image_2d = darsia.Image(src_square_2d, **meta_2d)
    src_image_2d = darsia.uniform_refinement(src_image_2d, nref)



    # Coarse dst image
    dst_square_2d = np.zeros((rows, cols), dtype=float)
    dst_square_2d[x_dst:x_dst+1, y_dst:y_dst+1] = 1
    dst_square_2d.transpose() # transpose to match the image orientation
    dst_image_2d = darsia.Image(dst_square_2d, **meta_2d)
    dst_image_2d = darsia.uniform_refinement(dst_image_2d, nref)

    # Normalize
    shape_meta_2d = src_image_2d.shape_metadata()
    geometry_2d = darsia.Geometry(**shape_meta_2d)
    src_image_2d.img /= geometry_2d.integrate(src_image_2d)
    dst_image_2d.img /= geometry_2d.integrate(dst_image_2d)


    # kappa coefficient
    permeability_2d = np.ones((rows, cols), dtype=float)
    permeability_2d[:, 3:5] = permeability_layer
    permeability_2d.transpose() # transpose to match the image orientation
    kappa_2d = 1.0 / permeability_2d
    kappa_image_2d = darsia.Image(kappa_2d, **meta_2d)
    kappa_image_2d = darsia.uniform_refinement(kappa_image_2d, nref)


    
    # Reference value for comparison
    n = src_image_2d.shape[0]
    # optimal potential/pressure is just -x
    x_approx = np.linspace(1/(2*n), 1-1/(2*n), n)
    x = -np.outer(x_approx,np.ones(src_image_2d.shape[1]))
    y = -np.outer(np.ones(src_image_2d.shape[1]),x_approx)
    opt_pot = x + y

    # By the symmetry of the problem, the optimal ray fromthe center 
    # of the source to the center of the destination will pass
    # through their common center.
    bar_src = [(x_src+0.5)/rows, (y_src+0.5)/cols]
    bar_dst = [(x_dst+0.5)/rows, (y_dst+0.5)/cols]
    intermidate_point = [(x_src+x_dst+1)/(2*rows), (y_src+y_dst+1)/(2*cols)]

    # Hence we can restrict to find the optimal ray from the source to the intermidate point
    """
    Src  |      Layer    | Destination
    a    | b   | 
    --------------
    .    |     |
      .  |x    |
    t1  .|     |  t1=theta_1
    -----------|l t2=thet1_2
         |. t2 |
         | .   |
         |  .  |  
         |   . |
         |    .|
    ---------- o=Intermidate point ---- 
    """

    print(f"{intermidate_point=}")
    a = 3/8-bar_src[0] # x distance from the source to the layer
    b = 1/8   # half of the width of the layer
    l = intermidate_point[1]-bar_src[1] # y_distance from the source to intermidate point
    


    # solve using scipy newton
    from scipy.optimize import toms748

    def h(x):
        return x**2 * ( (l-x)**2+b**2) - 1/permeability_layer **2 * (l-x)**2 * (x**2+a**2)

    def derivative_h(x):
        return 2*x * ( (l-x)**2+b**2) - 2*x**2 * (l-x) - permeability_layer**2 * 2 * (l-x) *(-1) * (x**2+a**2) - value_layer**2 * (l-x)**2 * (2 * x) 


    def L(x):
        """"
        We double the length of the path to the intermidate point to get the total length.
        """
        L=  np.sqrt(x**2+a**2) + np.sqrt((l-x)**2+b**2) / permeability_layer # high permeability=small length
        return 2 * L


    # use scipy newton to solve the equation
    
    L_min = minimize_scalar(L,bounds=(0,l),method="bounded")
    theta_1 = np.arctan(L_min.x/a)
    theta_2 = np.arctan((l-L_min.x)/b)

    print(f"{L_min.fun=:.3e} {L_min.x=:.3e} | {theta_1/(2*np.pi/360)=:.3e} {theta_2/(2*np.pi/360)=:.3e} {np.sin(theta_2)/np.sin(theta_1)=:.3e}")



    # Wassestein distnace is the integral of mass transported times the travel time
    src_value = np.max(src_image_2d.img)
    true_distance_2d = src_value * (1/8)**2 * L_min.fun


    plot_L = False
    if plot_L:
        t = np.linspace(0,l,100)
        import matplotlib.pyplot as plt
        plt.plot(t,L(t))
        plt.plot([L_min.x], [L_min.fun], 'ro')
        plt.show()


    return src_image_2d, dst_image_2d, true_distance_2d, kappa_image_2d



# ! ---- Solver options ----

# Linearization
newton_options = {
    # Scheme
    "L": 1e9,
}
bregman_std_options = {
    # Scheme
    "L": 1,
}
bregman_adaptive_options = {
    # Scheme
    "L": 1,
    "bregman_update": lambda iter: iter % 20 == 0,
}
linearizations = {
    "newton": [newton_options],
    "bregman": [
        bregman_std_options,
        bregman_adaptive_options,
    ],
}

# Acceleration
off_aa = {
    # Nonlinear solver
    "aa_depth": 0,
    "aa_restart": None,
}
on_aa = {
    # Nonlinear solver
    "aa_depth": 5,
    "aa_restart": 5,
}
accelerations = [off_aa, on_aa]

# Linear solver
lu_options = {
    # Linear solver
    "linear_solver": "direct",
}
amg_options = {
    "linear_solver": "amg",
    "linear_solver_options": {
        "tol": 1e-8,
    },
}

ksp_options_amg = {
    "linear_solver": "ksp",
    "linear_solver_options": {
        "tol": 1e-8,
    "prec_schur": "hypre"
    },
}

ksp_options_direct = {
    "linear_solver": "ksp",
    "linear_solver_options": {
        "tol": 1e-8,
    },
    "prec_schur": "lu"
}

formulations = ["full", "pressure"]

solvers = [ksp_options_amg, ksp_options_direct]



# General options
options = {
    # Method definition
    "l1_mode": "constant_cell_projection",
    "mobility_mode": "face_based",
    # Performance control
    "num_iter": 400,
    "tol_residual": 1e-10,
    "tol_increment": 1e-6,
    "tol_distance": 1e-10,
    "return_info": True,
    "verbose": False,
}

# ! ---- Tests ----
if __name__ == "__main__":
    """Test all combinations for Newton."""
    dim = 2

    ndivx = 3
    ndivy = 3
    x_src = 1
    y_src = 1
    x_dst = 6
    y_dst = 6

    h = []
    error_distance = []
    for nref in range(0,4):
        print()
        print(f"**************nref={nref}*****************")
        h.append((1.0/(2**ndivx)) / (2 ** nref))
        src_image_2d, dst_image_2d, true_distance_2d, kappa_image_2d = test_case(
            ndivx=ndivx, ndivy=ndivy,
            x_src=x_src, y_src=y_src, 
            x_dst=x_dst, y_dst=y_dst, 
            permeability_layer=permeability_layer, nref=nref
        )


        if method == "bregman":
            options.update(bregman_std_options)
            options.update(accelerations[0])
            options.update(solvers[0])
    
    
        # options.update({"formulation": "full"})
        options.update({"verbose": False})
        options.update({"num_iter": 200})
        options.update({"linear_solver": "ksp"})
        options.update({"kappa": kappa_image_2d.img})
        distance, info = darsia.wasserstein_distance(
            src_image_2d,
            dst_image_2d,
            options=options,
            method=method,
        )
        relative_err=abs(distance-true_distance_2d)/true_distance_2d
        print(f"{distance=:.2e} {true_distance_2d=:.2e} {relative_err=:.1e}")        
        error_distance.append(relative_err)

    
    # linear regression of the log-log plot of the error w.r.t to the mesh size h
    import numpy as np
    import matplotlib.pyplot as plt

    h = np.asarray(h)
    error_distance = np.asarray(error_distance)
    plt.plot(h,error_distance,'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    # fit a line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(np.log(h), np.log(error_distance))
    print(f"{slope=:.3e} {intercept=:.3e}")

    if slope < 0.9:
        raise ValueError("Wassersetin distance is not converging")
        
    hs = np.linspace(h[0],h[-1],100)
    plt.plot(h,error_distance,'o')
    plt.plot(hs,np.exp(intercept)*hs**slope)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    print(h)
    print(error_distance)

    darsia.plotting.plot_2d_wasserstein_distance(info)