import cv2
import numpy as np

class PatchwiseIlluminationCorrection:
    """Class for performing patchwise illumination correction on images using baseline images."""

    def __init__(
        self,
        image_path: str,
        baseline_path1: str,
        baseline_path2: str,
        nw: int = 1000,
        limit: int = 1450,
        show_images: bool = True,
        saving_path: str = "./correction_coefficients.npz", 
        eps: float = 1e-6
    ):

        self.nw = nw
        self.limit = limit
        self.saving_path = saving_path
        self.eps = eps

        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Image not found : {image_path}")

        self.baseline1 = cv2.imread(baseline_path1)
        if self.baseline1 is None:
            raise ValueError(f"Image not found : {baseline_path1}")

        self.baseline2 = cv2.imread(baseline_path2)
        if self.baseline2 is None:
            raise ValueError(f"Image not found : {baseline_path2}")

        
        self.height, self.width = self.img.shape[:2]
        self.nh = int((self.height - self.limit) * self.nw / self.width)
        self.dh = int((self.height - self.limit) / self.nh)
        self.dw = int(self.width / self.nw)

        r1, g1, b1 = self.extract_color_values_patches(self.baseline1)
        r2, g2, b2 = self.extract_color_values_patches(self.baseline2)

        self.r_diff, self.g_diff, self.b_diff = self.correction_coefficient(r1, g1, b1, r2, g2, b2)
      
        image_calibrated = self.correct_array(self.img)

        if show_images == True:
            cv2.imshow("calibrated image", image_calibrated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("calibrated image.png", image_calibrated)        


    def extract_color_values_patches(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract RGB values from image patches.
 
        Returns:
            Tuple containing R, G, B matrices.
        """
        r = np.zeros((self.nh, self.nw), dtype=np.float32)
        g = np.zeros((self.nh, self.nw), dtype=np.float32)
        b = np.zeros((self.nh, self.nw), dtype=np.float32)

        for i in range(self.nh):

            for j in range(self.nw):

                y0 = max(self.limit + int((i - 0.5) * self.dh), 0)
                y1 = min(self.limit + int((i + 0.5) * self.dh), self.height)

                x0 = max(int((j - 0.5) * self.dw), 0)
                x1 = min(int((j + 0.5) * self.dw), self.width)

                roi = image[y0:y1, x0:x1]

                mean_color = cv2.mean(roi)

                r[i, j] = int(mean_color[2])
                g[i, j] = int(mean_color[1])
                b[i, j] = int(mean_color[0])

        return np.array(r), np.array(g), np.array(b)

     
    def correction_coefficient(self, r1: np.ndarray, g1: np.ndarray, b1: np.ndarray, r2: np.ndarray, g2: np.ndarray, b2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate correction coefficients based on baseline images."""
        eps = self.eps
        r_m1 = np.mean(r1)
        g_m1 = np.mean(g1)
        b_m1 = np.mean(b1)
        r_m2 = np.mean(r2)
        g_m2 = np.mean(g2)
        b_m2 = np.mean(b2)
        r_diff2 = 1 / ((r1**2)/(r1**2+r2**2+eps) * r_m1/(r1+eps) + (r2**2)/(r1**2+r2**2+eps) * r_m2/(r2+eps)+eps)
        g_diff2 = 1 / ((g1**2)/(g1**2+g2**2+eps) * g_m1/(g1+eps) + (g2**2)/(g1**2+g2**2+eps) * g_m2/(g2+eps)+eps)
        b_diff2 = 1 / ((b1**2)/(b1**2+b2**2+eps) * b_m1/(b1+eps) + (b2**2)/(b1**2+b2**2+eps) * b_m2/(b2+eps)+eps)
        return r_diff2, g_diff2, b_diff2

    def correct_array(self, img: np.ndarray) -> np.ndarray:
        """Apply patchwise illumination correction to the input image."""

        r, g, b = self.extract_color_values_patches(img)
        
        r_new = r/self.r_diff
        g_new = g/self.g_diff
        b_new = b/self.b_diff

        image_calib = cv2.merge((b_new.astype(np.uint8), g_new.astype(np.uint8), r_new.astype(np.uint8)))
        img_rebuilt = cv2.resize(image_calib, (self.width, self.height-self.limit), interpolation=cv2.INTER_LINEAR)

        return img_rebuilt

    def save(self):
        """Save correction coefficients to a file."""
        np.savez(self.saving_path, r_diff=self.r_diff, g_diff=self.g_diff, b_diff=self.b_diff)
        print(f"Correction coefficients saved to {self.saving_path}")

    def load(self):
        """Load correction coefficients from a file."""
        data = np.load(self.saving_path)
        self.r_diff = data['r_diff']
        self.g_diff = data['g_diff']
        self.b_diff = data['b_diff']
        print(f"Correction coefficients loaded from {self.saving_path}")
