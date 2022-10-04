import daria as da

#%matplotlib qt5

 

curv_correction = da.CurvatureCorrection(image_source = "../laser_grid_on_geometry.jpg", width=2.8, height=1.5)

# curv_correction.show_image()

#Apply setup pre-bulge correction

# curv_correction.pre_bulge_correction(horizontal_bulge = 5e-10)
# curv_correction.show_image()
 

# curv_correction.crop([[17, 26], [76, 4410], [7878, 4390], [7912, 17]])
# curv_correction.show_image()

# curv_correction.crop([

#         [28, 18],

#        [39, 4379],

#         [7917, 4369],

#         [7916, 8],

#     ])

 

curv_correction.bulge_corection(left = 0, right = 0, top = 124, bottom = 132)
curv_correction.show_image()

# curv_correction.show_image()