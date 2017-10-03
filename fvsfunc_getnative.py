import vapoursynth as vs
from functools import partial

core = vs.core

'''
This is shamelessly ripped from fvsfunc https://github.com/Irrational-Encoding-Wizardry/fvsfunc by frechdachs
to remove unnecessary dependencies.
'''


# Wrapper with fmtconv syntax that tries to use the internal resizers whenever it is possible
def Resize(src, w, h, sx=None, sy=None, sw=None, sh=None, kernel='spline36', taps=None, a1=None,
           a2=None, a3=None, invks=None, invkstaps=None, fulls=None, fulld=None):

    bits = src.format.bits_per_sample

    if (src.width, src.height, fulls) == (w, h, fulld):
        return src

    if kernel is None:
        kernel = 'spline36'
    kernel = kernel.lower()

    if invks and kernel in ['bilinear', 'bicubic', 'lanczos', 'spline16', 'spline36'] and hasattr(core, 'descale_getnative') and invkstaps is None:
        return descale_getnative(src, w, h, kernel=kernel, b=a1, c=a2, taps=taps)
    if not invks:
        if kernel == 'bilinear':
            return core.resize.Bilinear(src, w, h, range=fulld, range_in=fulls, src_left=sx, src_top=sy,
                                        src_width=sw, src_height=sh)
        if kernel == 'bicubic':
            return core.resize.Bicubic(src, w, h, range=fulld, range_in=fulls, filter_param_a=a1, filter_param_b=a2,
                                       src_left=sx, src_top=sy, src_width=sw, src_height=sh)
        if kernel == 'spline16':
            return core.resize.Spline16(src, w, h, range=fulld, range_in=fulls, src_left=sx, src_top=sy,
                                        src_width=sw, src_height=sh)
        if kernel == 'spline36':
            return core.resize.Spline36(src, w, h, range=fulld, range_in=fulls, src_left=sx, src_top=sy,
                                        src_width=sw, src_height=sh)
        if kernel == 'lanczos':
            return core.resize.Lanczos(src, w, h, range=fulld, range_in=fulls, filter_param_a=taps,
                                       src_left=sx, src_top=sy, src_width=sw, src_height=sh)
    return Depth(core.fmtc.resample(src, w, h, sx=sx, sy=sy, sw=sw, sh=sh, kernel=kernel, taps=taps,
                              a1=a1, a2=a2, a3=a3, invks=invks, invkstaps=invkstaps, fulls=fulls, fulld=fulld), bits)


def descale_getnative(src, width, height, kernel='bilinear', b=1/3, c=1/3, taps=3, yuv444=False, gray=False, chromaloc=None):
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h

    descale_getnative_filter = get_descale_getnative_filter(b, c, taps, kernel)

    if src_cf == vs.RGB and not gray:
        rgb = descale_getnative_filter(to_rgbs(src), width, height)
        return rgb.resize.Point(format=src_f.id)

    y = descale_getnative_filter(to_grays(src), width, height)
    y_f = core.register_format(vs.GRAY, src_st, src_bits, 0, 0)
    y = y.resize.Point(format=y_f.id)

    if src_cf == vs.GRAY or gray:
        return y

    if not yuv444 and ((width % 2 and src_sw) or (height % 2 and src_sh)):
        raise ValueError('descale_getnative: The output dimension and the subsampling are incompatible.')

    uv_f = core.register_format(src_cf, src_st, src_bits, 0 if yuv444 else src_sw, 0 if yuv444 else src_sh)
    uv = src.resize.Spline36(width, height, format=uv_f.id, chromaloc_s=chromaloc)

    return core.std.ShufflePlanes([y,uv], [0,1,2], vs.YUV)


def to_grays(src):
    return src.resize.Point(format=vs.GRAYS)


def to_rgbs(src):
    return src.resize.Point(format=vs.RGBS)


def get_descale_getnative_filter(b, c, taps, kernel):
    kernel = kernel.lower()
    if kernel == 'bilinear':
        return core.descale_getnative.Debilinear
    elif kernel == 'bicubic':
        return partial(core.descale_getnative.Debicubic, b=b, c=c)
    elif kernel == 'lanczos':
        return partial(core.descale_getnative.Delanczos, taps=taps)
    elif kernel == 'spline16':
        return core.descale_getnative.Despline16
    elif kernel == 'spline36':
        return core.descale_getnative.Despline36
    else:
        raise ValueError('descale_getnative: Invalid kernel specified.')


def Depth(src, bits, dither_type='error_diffusion', range=None, range_in=None):
    src_f = src.format
    src_cf = src_f.color_family
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h
    dst_st = vs.INTEGER if bits < 32 else vs.FLOAT

    if isinstance(range, str):
        range = RANGEDICT[range]

    if isinstance(range_in, str):
        range_in = RANGEDICT[range_in]

    if (src_bits, range_in) == (bits, range):
        return src
    out_f = core.register_format(src_cf, dst_st, bits, src_sw, src_sh)
    return core.resize.Point(src, format=out_f.id, dither_type=dither_type, range=range, range_in=range_in)


RANGEDICT = {'limited': 0, 'full': 1}
