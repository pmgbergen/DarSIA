"""Example for Wasserstein computations moving a square to another location."""

import numpy as np
import pytest
from scipy.ndimage import zoom
from scipy.optimize import minimize_scalar
import darsia
import matplotlib.pyplot as plt


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

def plan_3points(A,B,C):
    """
    Return the coefficients of the plane passing through the points A,B,C
    in the form z   = a x + b y + c
    """
    AB = np.array(B)-np.array(A)
    AC = np.array(C)-np.array(A)
    n = np.cross(AB,AC)
    a,b,c = n
    d = -np.dot(n,A)
    return np.array([a/c,b/c,d/c])

def line_2points(A,B):
    """
    Return the coefficients of the line passing through the points A,
    in the form y= m x + k
    """
    m = (B[1]-A[1])/(B[0]-A[0])
    k = A[1]-m*A[0]
    return np.array([m,k])

def line_point_direction(point,direction):
    """
    Return the coefficients of the line passing through the points A,
    in the form y= m x + k
    """
    m = direction[1]/direction[0]
    k = point[1]-m*point[0]
    return np.array([m,k])

def asfunction(coefficients):
    """
    Return the function f(x) = a^T x + b
    """
    def f(x):
        return np.dot(coefficients[0:-1],np.array(x))+coefficients[-1]
    return f
    

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

    opt_direction_1 = [np.cos(theta_1),-np.sin(theta_1)]
    opt_direction_2 = [np.cos(theta_2),-np.sin(theta_2)]

    def opt_pot(x,y):
        if 0<x<3/8:
            # compute the distance to the upper/left boundary along
            # the direction 
            return x * np.cos(theta_1) - y * np.sin(theta_1)
        if 3/8<=x<5/8:
            # compute the distance to the upper/left boundary along
            # the direction 
            return (x * np.cos(theta_2) - y * np.sin(theta_2)) / permeability_layer
        if 5/8<=x<1:
            # compute the distance to the upper/left boundary along
            # the direction 
            return  x * np.cos(theta_1) - y * np.sin(theta_1)
    
    def opt_grad_pot(x,y):
        if 0<x<3/8:
            # compute the distance to the upper/left boundary along
            # the direction 
            grad = [np.cos(theta_1), - np.sin(theta_1)]
        if 3/8<=x<5/8:
            # compute the distance to the upper/left boundary along
            # the direction 
            grad = [np.cos(theta_2)/ permeability_layer, -np.sin(theta_2) / permeability_layer ]
        if 5/8<=x<1:
            # compute the distance to the upper/left boundary along
            # the direction 
            grad = [np.cos(theta_1), -np.sin(theta_1)]
        return grad
    


    # coordinate of the vertices of the square
    # Q3------Q2
    # |       |
    # |       |
    # Q0------Q1
    Q0 = [bar_src[0]-0.5/8, bar_src[1]-0.5/8]
    Q1 = [bar_src[0]+0.5/8, bar_src[1]-0.5/8]
    Q2 = [bar_src[0]+0.5/8, bar_src[1]+0.5/8]
    Q3 = [bar_src[0]-0.5/8, bar_src[1]+0.5/8]

    # point intersection of the ray from Q3 to the direction opt_direction_1
    # with the segment Q2Q3
    line = asfunction(line_point_direction(Q3,opt_direction_1))
    P3 = [Q2[0],line([Q2[0]])]
    print(f"{P3=}")
    # point intersection of the ray from Q1 to the oppposite direction opt_direction_1
    # with the segment Q0Q3
    line = asfunction(line_point_direction(Q1,[-np.cos(theta_1),np.sin(theta_1)]))
    P1 = [Q0[0],line([Q0[0]])]
    print(f"{P1=}")

    line = asfunction(line_point_direction(Q0,opt_direction_1))
    P0 = [Q1[0],line([Q1[0]])]
    print(f"{P0=}")

    # point intersection of the ray from Q2 to the direction opt_direction_1
    line = asfunction(line_point_direction(Q0,opt_direction_1))
    L0 = [3/8,line([3/8])]
    line = asfunction(line_point_direction(Q1,opt_direction_1))
    L1 = [3/8,line([3/8])]
    line = asfunction(line_point_direction(Q2,opt_direction_1))
    L2 = [3/8,line([3/8])]
    line = asfunction(line_point_direction(Q3,opt_direction_1))
    L3 = [3/8,line([3/8])]

    # point intersection of the rays with x=0.5
    line = asfunction(line_point_direction(L0,opt_direction_2))
    M0 = [0.5,line([0.5])]
    line = asfunction(line_point_direction(L1,opt_direction_2))
    M1 = [0.5,line([0.5])]
    line = asfunction(line_point_direction(L2,opt_direction_2))
    M2 = [0.5,line([0.5])]
    line = asfunction(line_point_direction(L3,opt_direction_2))
    M3 = [0.5,line([0.5])]


    line0 = asfunction(line_point_direction(Q0,opt_direction_1))
    line1 = asfunction(line_point_direction(Q1,opt_direction_1))
    line2 = asfunction(line_point_direction(Q2,opt_direction_1))
    line3 = asfunction(line_point_direction(Q3,opt_direction_1))


    L_max = 1/8 / np.cos(theta_1)
    tdens_max = L_max * src_value
    print([*Q2,0],[*Q3,0],[*P3,tdens_max])
    Q0P0Q1 = asfunction(plan_3points([*Q0,0],[*P0,0],[*Q1,tdens_max]))
    Q0Q1P1 = asfunction(plan_3points([*Q0,0],[*Q1,tdens_max],[*P1,0]))
    Q1P3Q3 = asfunction(plan_3points([*Q1,tdens_max],[*P3,tdens_max],[*Q3,0]))
    Q2Q3P3 = asfunction(plan_3points([*Q2,0],[*Q3,0],[*P3,tdens_max]))
    
    P0L0L1 = asfunction(plan_3points([*P0,0],[*L0,0],[*L1,tdens_max]))
    Q1L1L3 = asfunction(plan_3points([*Q1,tdens_max],[*L1,tdens_max],[*L3,tdens_max]))
    P3L3L2 = asfunction(plan_3points([*P3,0],[*L3,0],[*L2,tdens_max]))
    
    L0M0M1 = asfunction(plan_3points([*L0,0],[*M0,0],[*M1,tdens_max]))
    L1M1M3 = asfunction(plan_3points([*L1,tdens_max],[*M1,tdens_max],[*M3,tdens_max]))
    L3M3M2 = asfunction(plan_3points([*L3,tdens_max],[*M3,tdens_max],[*M2,0]))



    if theta_1 <= np.pi/4:
        def opt_tdens(x,y):
            if 0<x<Q0[0]:
                return 0
            elif Q0[0]<=x<Q1[0]:
                y0 = line0(x)
                y1 = line1(x)
                y2 = line2(x)
                y3 = line3(x)

                if y0<y<y1:
                    if y<Q0[1]:
                        print("Q0P0Q1",x,y)
                        return 1#Q0P0Q1([x,y])
                        
                    else:
                        print("Q0Q1P1",x,y)
                        return 2#Q0Q1P1([x,y])
                if y1<y<y2:
                    print("Q1P3Q3",x,y)
                    return 3#Q1P3Q3([x,y])
                if y2<y<y3 and y<=Q2[1]:
                    print("Q2Q3P3",x,y)
                    return 4#Q2Q3P3([x,y])                    
                else:
                    return 0
            elif Q1[0]<=x<3/8:
                print("extra")
                y0 = line0(x)
                y1 = line1(x)
                y2 = line2(x)
                y3 = line3(x)

                if y0<y<y1:
                    return 5#tdens_max
                elif y1<y<y2:
                    return 6#tdens_max
                elif y2<y<y3:
                    return 7#tdens_max
                else:
                    return 0
            elif 3/8<=x<0.5:
                print("layer")
                y0 = line0(x)
                y1 = line1(x)
                y2 = line2(x)
                y3 = line3(x)
                if y0<y<y1:
                    return 8#L0M0M1([x,y])
                elif y1<y<y2:
                    return 9#L1M1M3([x,y])
                elif y2<y<y3:
                    return 10#L3M3M2([x,y])
                else:
                    return 0
            elif 0.5<=x<1:
                # mirrow the previous case
                return opt_tdens(1-x,1-y)
                

            #     if x+y-(bar_src[0]+bar_src[1])<0:
            #         # compute the distance to the upper/left boundary along
            #         # the direction 
            #         dist = (x_src * h - x) / np.sin(theta_1)
            #         return dist
            #     else:


            # if 0<(x-bar_src[0])<1/16 and 0<(y-bar_src[1])<1/16:
            #     # compute the distance to the upper/left boundary along
            #     # the direction 
            #     dist = (x_src * h - x) / np.sin(theta_1)
            #     return dist
            
            # if bar_src[0]+1/16<x<3/8:
            #     if 
            #     A=[Q0[0],Q0[1],0]
            #     B=[3/8,
            #     a,b,c = plan_coefficients(A,B,C)

    plot_L = False
    if plot_L:
        t = np.linspace(0,l,100)
        plt.plot(t,L(t))
        plt.plot([L_min.x], [L_min.fun], 'ro')
        plt.show()

    return src_image_2d, dst_image_2d, true_distance_2d, kappa_image_2d, opt_pot, opt_grad_pot, opt_tdens



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
    for nref in range(5,6):
        print()
        print(f"**************nref={nref}*****************")
        h.append((1.0/(2**ndivx)) / (2 ** nref))
        src_image_2d, dst_image_2d, true_distance_2d, kappa_image_2d, opt_pot, opt_grad_pot, opt_tdens = test_case(
            ndivx=ndivx, ndivy=ndivy,
            x_src=x_src, y_src=y_src, 
            x_dst=x_dst, y_dst=y_dst, 
            permeability_layer=permeability_layer, nref=nref
        )

        # Meshgrid
        grid: darsia.Grid = darsia.generate_grid(src_image_2d)
        Y, X = np.meshgrid(
            grid.voxel_size[0] * (0.5 + np.arange(grid.shape[0] - 1, -1, -1)),
            grid.voxel_size[1] * (0.5 + np.arange(grid.shape[1])),
            indexing="ij",
        )
        
        XY = np.vstack((X.flatten(),Y.flatten()))
        opt_tdens = np.vectorize(opt_tdens)
        opt_tdens_flat = opt_tdens(X,Y).flatten()
        opt_tdens_grid = opt_tdens_flat.reshape(grid.shape, order="F")
        

        # Plot the pressure
        plt.figure("Optimal transport density")
        plt.pcolormesh(X, Y, opt_tdens_grid, cmap="turbo")
        plt.colorbar(label="pressure")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.show()


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
        info["xy"]=[3/8,1.0-((y_src+0.5)/8.0+0.269)]#L_min.x]

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