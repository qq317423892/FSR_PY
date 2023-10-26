from PIL import Image

def bilinear_interpolation(src_img, scale_factor):
    img_size = src_img.size
    img_width = int(img_size[0] * scale_factor)
    img_height = int(img_size[1] * scale_factor)
    dst_img = Image.new('RGB', (img_width, img_height), (255, 255, 255))

    real_scale = img_size[0] / img_width

    for dst_y in range(0, img_height):
        for dst_x in range(0, img_width):
            dstx_srcx = dst_x * real_scale
            dsty_srcy = dst_y * real_scale
            u = round(dstx_srcx % 1, 2)
            v = round(dsty_srcy % 1, 2)

            int_dstx = int(dstx_srcx)
            int_dsty = int(dsty_srcy)

            pos_x = []
            pos_y = []

            if int_dstx == img_size[0] - 1:
                pos_x = [int_dstx - 1, int_dstx]
            else:
                pos_x = [int_dstx, int_dstx + 1]

            if int_dsty == img_size[1] - 1:
                pos_y = [int_dsty - 1, int_dsty]
            else:
                pos_y = [int_dsty, int_dsty + 1]

            q1 = src_img.getpixel((pos_x[0], pos_y[0]))
            q2 = src_img.getpixel((pos_x[0], pos_y[1]))
            q3 = src_img.getpixel((pos_x[1], pos_y[0]))
            q4 = src_img.getpixel((pos_x[1], pos_y[1]))

            color_r = int(q1[0] * (1 - u) * (1 - v) + \
                      q2[0] * (1 - u) * (v) + \
                      q3[0] * (u) * (1 - v) + \
                      q4[0] * (u) * (v))

            color_g = int(q1[1] * (1 - u) * (1 - v) + \
                      q2[1] * (1 - u) * (v) + \
                      q3[1] * (u) * (1 - v) + \
                      q4[1] * (u) * (v))

            color_b = int(q1[2] * (1 - u) * (1 - v) + \
                      q2[2] * (1 - u) * (v) + \
                      q3[2] * (u) * (1 - v) + \
                      q4[2] * (u) * (v))

            dst_img.putpixel((dst_x, dst_y), (color_r, color_g, color_b))

    dst_img.save("out/scale_b.png")