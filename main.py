from PIL import Image

from bilinear import bilinear_interpolation
from fsr import fsr

if __name__=="__main__":
    small_img = Image.open("mid.png")
    small_size = small_img.size
    scale_factor = 2.0

    bilinear_interpolation(small_img, scale_factor)
    fsr(small_img, scale_factor)

    

