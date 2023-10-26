import math
from PIL import Image

EPS = 1e-5
FSR_RCAS_LIMIT = 0.25 - 1.0 / 16.0

def stat(value):
    return min(1.0, max(0, value))

class FSR(object):
    def __init__(self, src_img, scale_factor):
        self.src_img = src_img
        self.scale_factor = scale_factor
        self.dst_img = None

    ##  获取 t 插值使用到的四个像素颜色
    ##  p1 p2
    ##  t  p3
    ##  返回 t, p3, p2, p1
    def _get_near_four_color(self, p):
        img_width, img_height = self.src_img.size
        x = max(0, min(p[0], img_width - 1))
        y = max(0, min(p[1], img_height - 1))

        p1x, p1y = x, y - 1
        p2x, p2y = x + 1, y - 1
        p3x, p3y = x + 1, y

        p1_pos = (p1x, max(0, p1y))
        p2_pos = (min(p2x, img_width - 1), max(0, p2y))
        t_pos = (x, y)
        p3_pos = (min(p3x, img_width - 1), p3y)

        p1 = self.src_img.getpixel(p1_pos)
        p2 = self.src_img.getpixel(p2_pos)
        t = self.src_img.getpixel(t_pos)
        p3 = self.src_img.getpixel(p3_pos)

        p1_n = [item / 255.0 for item in p1]
        p2_n = [item / 255.0 for item in p2]
        p3_n = [item / 255.0 for item in p3]
        t_n  = [item / 255.0 for item in t]
        return (t_n, p3_n, p2_n, p1_n)

    # 计算灰度值，范围 0-2
    def _calc_color_lumen(self, color):
        r, g, b = color
        return 0.5 * r + 0.5 * b + g
        # return 0.3 * r + 0.1 * b + 0.5 * g

    ## 
    ## s t
    ## u v
    ## type = 0, 1, 2, 3
    ##        s  t  u  v
    def _fsr_easu_set_f(self, dir, length, p, stuv,
        al, bl, cl, dl, el):

        u, v = p
        w = 0
        if stuv == 0:
            w = (1 - u) * (1 - v)
        elif stuv == 1:
            w = u * (1 - v)
        elif stuv == 2:
            w = (1 - u) * v
        else:
            w = u * v
        
        ##     a
        ##  b  c  d
        ##     e
        dc = abs(dl - cl)
        cb = abs(cl - bl)
        dirx = dl - bl
        dir[0] += dirx * w

        max_dc_cb = max(EPS, max(dc, cb))
        ## 截断到 0-1
        lenx = stat(abs(dirx) / max_dc_cb)

        lenx *= lenx
        length += lenx * w

        ## for y axis
        ec = abs(el - cl)
        ca = abs(cl - al)
        diry = el - al
        dir[1] += diry * w

        max_ec_ca = max(EPS, max(ec, ca))
        leny = stat(abs(diry) / max_ec_ca)
        
        leny *= leny
        length += leny * w
        return length

    def _fsr_easu_tapf(self, ac, aw, off, dir, length, lob, clp, c):
        # vx = x * cos + y * sin
        # vy = -x * sin + y * cos
        vx = off[0] * dir[0] + off[1] * dir[1]
        vy = off[0] * (-dir[1]) + off[1] * dir[0]

        vx *= length[0]
        vy *= length[1]

        d2 = vx * vx + vy * vy
        d2 = min(d2, clp)

        ##  (25/16 * (2/5 * x^2 - 1)^2 - (25/16 - 1)) * (1/4 * x^2 - 1)^2
        ##  |        |____________|                 |   |               |
        ##  |              wb                       |   |      wa       |
        ##  |_______________________________________|   |_______________|
        ##                   base                             window

        wb = 2.0 / 5 * d2 -1
        wa = lob * d2 - 1
        wb *= wb
        wa *= wa
        base = 25.0 / 16 * wb - (25.0 / 16 - 1)
        w = base * wa
        ac[0] += c[0] * w
        ac[1] += c[1] * w
        ac[2] += c[2] * w
        aw += w
        return aw

    ## 12-tap kernel. f 是我们需要采样的原图中的点  zzzz，不需要的点
    ##    z z         z  z
    ##    b c         p0 c
    ##  e f g h   e   f  g  h
    ##  i j k l   p1  j  p2  l
    ##    n o         n  o
    ##    z z         p3 z
    ## dstx dsty 放大后的像素位置
    ## srcx srcy 原图中的像素位置
    def _fsr_easu_pixel(self, dstx, dsty):
        srcx_f = dstx / self.scale_factor
        srcy_f = dsty / self.scale_factor
        srcx = math.floor(srcx_f)
        srcy = math.floor(srcy_f)

        ppx = srcx_f - srcx
        ppy = srcy_f - srcy
        pp = (ppx, ppy)

        # p0 -> b
        # p1 -> i
        # p2 -> k
        # p3 -> z
        p0 = (srcx, srcy - 1)
        p1 = (srcx - 1, srcy + 1)
        p2 = (srcx + 1, srcy + 1)
        p3 = (srcx, srcy + 3)

        bczz = self._get_near_four_color(p0)
        ijfe = self._get_near_four_color(p1)
        klhg = self._get_near_four_color(p2)
        zzon = self._get_near_four_color(p3)

        b, c, _, _ = bczz
        i, j, f, e = ijfe
        k, l, h, g = klhg
        _, _, o, n = zzon

        bl = self._calc_color_lumen(b)
        cl = self._calc_color_lumen(c)
        il = self._calc_color_lumen(i)
        jl = self._calc_color_lumen(j)
        fl = self._calc_color_lumen(f)
        el = self._calc_color_lumen(e)
        kl = self._calc_color_lumen(k)
        ll = self._calc_color_lumen(l)
        hl = self._calc_color_lumen(h)
        gl = self._calc_color_lumen(g)
        ol = self._calc_color_lumen(o)
        nl = self._calc_color_lumen(n)

        dir = [0, 0]
        length = 0

        ##    b c         b        c
        ##  e f g h     e f g    f g h     f         g
        ##  i j k l       j        k     i j k     j k l
        ##    n o                          n         o
        length = self._fsr_easu_set_f(dir, length, pp, 0, bl, el, fl, gl, jl)
        length = self._fsr_easu_set_f(dir, length, pp, 1, cl, fl, gl, hl, kl)
        length = self._fsr_easu_set_f(dir, length, pp, 2, fl, il, jl, kl, nl)
        length = self._fsr_easu_set_f(dir, length, pp, 3, gl, jl, kl, ll, ol)

        self.max_lumen = max(self.max_lumen, length)
        if length > 2:
            self.debug_dict[(dstx, dsty)] = length

        # dir_dot = dir[0] * dir[0] + dir[1] * dir[1]
        dir2 = (dir[0] * dir[0], dir[1] * dir[1])
        dirr = dir2[0] + dir2[1]
        zero = dirr < 1.0 / 32768.0
        if zero:
            dirr = 1
            dir[0] = 1

        dirr = 1 / math.sqrt(dirr)
        dir[0] *= dirr
        dir[1] *= dirr

        ## 将 F 值范围从 (0, 2) 转换到 (0, 1)
        length = (length * 0.5) ** 2

        stretch = (dir[0] * dir[0] + dir[1] * dir[1]) / (max(abs(dir[0]), abs(dir[1])))
        len2 = (1 + (stretch - 1) * length, 1 - 0.5 * length)

        # w = 1/2 - 1/4 * F   w ∈ [1/4, 1/2]
        lob = 0.5 + ((1.0 / 4 - 0.04) - 0.5) * length
        # clob  1/w    clob ∈ [1, 2]
        clob = 1.0 / lob

        ##    b c
        ##  e f g h
        ##  i j k l
        ##    n o
        minr = min(f[0], g[0], j[0], k[0])
        ming = min(f[1], g[1], j[1], k[1])
        minb = min(f[2], g[2], j[2], k[2])

        maxr = max(f[0], g[0], j[0], k[0])
        maxg = max(f[1], g[1], j[1], k[1])
        maxb = max(f[2], g[2], j[2], k[2])

        ac = [0, 0, 0]
        aw = 0

        boff = ( 0 - pp[0], -1 - pp[1])
        coff = ( 1 - pp[0], -1 - pp[1])
        ioff = (-1 - pp[0],  1 - pp[1])
        joff = ( 0 - pp[0],  1 - pp[1])
        foff = ( 0 - pp[0],  0 - pp[1])
        eoff = (-1 - pp[0],  0 - pp[1])
        koff = ( 1 - pp[0],  1 - pp[1])
        loff = ( 2 - pp[0],  1 - pp[1])
        hoff = ( 2 - pp[0],  0 - pp[1])
        goff = ( 1 - pp[0],  0 - pp[1])
        ooff = ( 1 - pp[0],  2 - pp[1])
        noff = ( 0 - pp[0],  2 - pp[1])

        offlist = [boff, coff, ioff, joff, foff, eoff, koff, loff, hoff, goff, ooff, noff]
        colorlist = [b, c, i, j, f, e, k, l, h, g, o, n]

        for i in range(0, len(offlist)):
            aw = self._fsr_easu_tapf(ac, aw, offlist[i], dir, len2, lob, clob, colorlist[i])

        if aw == 0:
            aw = EPS

        pixr = int(min(maxr, max(minr, ac[0] / aw)) * 255)
        pixg = int(min(maxg, max(ming, ac[1] / aw)) * 255)
        pixb = int(min(maxb, max(minb, ac[2] / aw)) * 255)

        self.dst_img.putpixel((dstx, dsty), (pixr, pixg, pixb))

    def fsr_easu(self):
        img_width, img_height = self.dst_img.size
        for dstx in range(0, img_width):
            for dsty in range(0, img_height):
                self._fsr_easu_pixel(dstx, dsty)

    def get_dst_pixel_normal(self, p):
        img_width, img_height = self.dst_img.size
        x = max(0, min(img_width - 1, p[0]))
        y = max(0, min(img_height - 1, p[1]))
        valid_p = (x, y)
        color = self.dst_img.getpixel(valid_p)
        color_n = [item / 255.0 for item in color]
        return color_n

    ## RCAS
    def _fsr_rcas_input_f(self, r, g, b):
        pass

    ##     b
    ##  d  e  f
    ##     h
    def _fsr_rcas_pixel(self, dstx, dsty):
        img_width, img_height = self.dst_img.size

        bpos = (dstx, max(0, dsty - 1))
        dpos = (max(0, dstx - 1), dsty)
        epos = (dstx, dsty)
        fpos = (min(dstx + 1, img_width), dsty)
        hpos = (dstx, min(img_height, dsty + 1))

        b = self.get_dst_pixel_normal(bpos)
        d = self.get_dst_pixel_normal(dpos)
        e = self.get_dst_pixel_normal(epos)
        f = self.get_dst_pixel_normal(fpos)
        h = self.get_dst_pixel_normal(hpos)

        ## 空函数
        self._fsr_rcas_input_f(b[0], b[1], b[2])
        self._fsr_rcas_input_f(d[0], d[1], d[2])
        self._fsr_rcas_input_f(e[0], e[1], e[2])
        self._fsr_rcas_input_f(f[0], f[1], f[2])
        self._fsr_rcas_input_f(h[0], h[1], h[2])

        bl = self._calc_color_lumen(b)
        dl = self._calc_color_lumen(d)
        el = self._calc_color_lumen(e)
        fl = self._calc_color_lumen(f)
        hl = self._calc_color_lumen(h)

        ##     b
        ##  d  e  f
        ##     h
        lumen_list = [bl, dl, el, fl, hl]
        max_lumen = max(lumen_list)
        min_lumen = min(lumen_list)
        lumen_offset = max(EPS, max_lumen - min_lumen)
        # noise detection
        nz = (bl + dl + hl + fl) / 4.0 - el
        nz = stat(abs(nz) / lumen_offset)
        nz = -0.5 * nz + 1

        mnr = min((b[0], d[0], f[0], h[0]))
        mng = min((b[1], d[1], f[1], h[1]))
        mnb = min((b[2], d[2], f[2], h[2]))
        mxr = max((b[0], d[0], f[0], h[0], EPS))
        mxg = max((b[1], d[1], f[1], h[1], EPS))
        mxb = max((b[2], d[2], f[2], h[2], EPS))

        peakc = [1.0, -4.0]
        hit_min_r = min(mnr, e[0]) / (4.0 * mxr)
        hit_min_g = min(mng, e[1]) / (4.0 * mxg)
        hit_min_b = min(mnb, e[2]) / (4.0 * mxb)

        four_min_sub_mnr = max(EPS, 4.0 * mnr + peakc[1])
        four_min_sub_mng = max(EPS, 4.0 * mng + peakc[1])
        four_min_sub_mnb = max(EPS, 4.0 * mnb + peakc[1])
        hit_max_r = (peakc[0] - max(mxr, e[0])) / (four_min_sub_mnr)
        hit_max_g = (peakc[0] - max(mxg, e[1])) / (four_min_sub_mng)
        hit_max_b = (peakc[0] - max(mxb, e[2])) / (four_min_sub_mnb)

        lobr = max(-hit_min_r, hit_max_r)
        lobg = max(-hit_min_g, hit_max_g)
        lobb = max(-hit_min_b, hit_max_b)

        lobe = max(-FSR_RCAS_LIMIT, min((lobr, lobg, lobb, 0))) * self.scale_factor

        rcp_lobe = 1 / (4.0 * lobe + 1)
        pix_r = (lobe * b[0] + lobe * d[0] + lobe * d[0] + lobe * f[0] + e[0]) * rcp_lobe
        pix_g = (lobe * b[1] + lobe * d[1] + lobe * d[1] + lobe * f[1] + e[1]) * rcp_lobe
        pix_b = (lobe * b[2] + lobe * d[2] + lobe * d[2] + lobe * f[2] + e[2]) * rcp_lobe

        pix_color = (int(pix_r * 255), int(pix_g * 255), int(pix_b * 255))
        self.dst_img.putpixel(epos, pix_color)

    def fsr_rcas(self):
        img_width, img_height = self.dst_img.size
        for dstx in range(0, img_width):
            for dsty in range(0, img_height):
                self._fsr_rcas_pixel(dstx, dsty)

    def process(self, out_path):
        img_size = self.src_img.size
        img_width = int(img_size[0] * self.scale_factor)
        img_height = int(img_size[1] * self.scale_factor)
        self.dst_img = Image.new('RGB', (img_width, img_height), (255, 255, 255))

        self.debug_dict = {}
        self.max_lumen = 0
        self.fsr_easu()
        self.dst_img.save("out/scale_fsr_easu.png")
        self.fsr_rcas()
        self.dst_img.save("out/scale_fsr.png")
        # self.dst_img.save(out_path)


def fsr(src_img, scale_factor):
    fsr_obj = FSR(src_img, scale_factor)
    fsr_obj.process("scale_fsr.png")
