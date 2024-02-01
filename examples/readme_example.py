import numpy as np

import darsia

# Create a darsia Image: An image that also contains information of physical entities
image = darsia.imread("images/baseline.jpg", width=2.8, height=1.5)

# Use the show method to take a look at the imported image.
image.show()

# Copy the image and adds a grid on top of it.
grid_image = image.add_grid(dx=0.1, dy=0.1)
grid_image.show()

# Extract region of interest (ROI) from image (box defined by two corners)
roi = darsia.make_coordinate([[1.5, 0], [2.8, 0.7]])
ROI_image = image.subregion(roi)
ROI_image.show()

# Metadata
metadata = image.metadata()
print(f"The stored metadata of the full image: {metadata}.")

# The image array can be accessed
array = image.img
