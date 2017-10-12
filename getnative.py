import time
import argparse
import asyncio
import vapoursynth
import os
from functools import partial

"""
Reworke by Infi
Original Author: kageru https://gist.github.com/kageru/549e059335d6efbae709e567ed081799
Thanks: BluBb_mADe, FichteFoll, stux!, Frechdachs
"""

core = vapoursynth.core
core.add_cache = False
imwri = getattr(core, "imwri", getattr(core, "imwrif", None))
output_dir = os.path.splitext(os.path.basename(__file__))[0]


class GetNative:
    def __init__(self, src, kernel=None, b=None, c=None, taps=None, ar=None, approx=None, min_h=None, max_h=None,
                 frame=None, img_out=None, plot_scaling=None, plot_format=None, show_plot=None):
        self.plot_format = plot_format
        self.plot_scaling = plot_scaling
        self.src = src
        self.min_h = min_h
        self.max_h = max_h
        self.ar = ar
        self.b = b
        self.c = c
        self.taps = taps
        self.approx = approx
        self.kernel = kernel
        self.frame = frame
        self.img_out = img_out
        self.txt_output = ""
        self.show_plot = show_plot
        self.resolutions = []
        self.filename = self.get_filename()

    async def run(self):
        # change format to GrayS with bitdepth 32 for descale
        src = self.src[self.frame]
        matrix_s = '709' if src.format.color_family == vapoursynth.RGB else None
        src_luma32 = core.resize.Point(src, format=vapoursynth.YUV444PS, matrix_s=matrix_s)
        src_luma32 = core.std.ShufflePlanes(src_luma32, 0, vapoursynth.GRAY)
        src_luma32 = core.std.Cache(src_luma32)

        # descale each individual frame
        resizer = descale_approx if self.approx else descale_accurate
        clip_list = []
        for h in range(self.min_h, self.max_h + 1):
            clip_list.append(resizer(src_luma32, self.getw(h), h, self.kernel, self.b, self.c, self.taps))
        full_clip = core.std.Splice(clip_list, mismatch=True)
        full_clip = upscale(full_clip, self.getw(src.height), src.height, self.kernel, self.b, self.c, self.taps)
        if self.ar != src.width / src.height:
            src_luma32 = upscale(src_luma32, self.getw(src.height), src.height, self.kernel, self.b, self.c, self.taps)
        expr_full = core.std.Expr([src_luma32 * full_clip.num_frames, full_clip], 'x y - abs dup 0.015 > swap 0 ?')
        full_clip = core.std.CropRel(expr_full, 5, 5, 5, 5)
        full_clip = core.std.PlaneStats(full_clip)
        full_clip = core.std.Cache(full_clip)

        tasks_pending = set()
        futures = {}
        vals = []
        for frame_index in range(len(full_clip)):
            fut = asyncio.ensure_future(asyncio.wrap_future(full_clip.get_frame_async(frame_index)))
            tasks_pending.add(fut)
            futures[fut] = frame_index
            while len(tasks_pending) >= core.num_threads * (2 if self.approx else 1) + 2:
                tasks_done, tasks_pending = await asyncio.wait(tasks_pending, return_when=asyncio.FIRST_COMPLETED)
                vals += [(futures.pop(task), task.result().props.PlaneStatsAverage) for task in tasks_done]

        tasks_done, _ = await asyncio.wait(tasks_pending)
        vals += [(futures.pop(task), task.result().props.PlaneStatsAverage) for task in tasks_done]
        vals = [v for _, v in sorted(vals)]
        ratios, vals, best_value = self.analyze_results(vals)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        self.save_plot(vals)
        if self.img_out:
            self.save_images(src_luma32)

        self.txt_output += 'Raw data:\nResolution\t | Relative Error\t | Relative difference from last\n'
        for i, error in enumerate(vals):
            self.txt_output += f'{i + self.min_h:4d}\t\t | {error:.10f}\t\t\t | {ratios[i]:.2f}\n'

        with open(f"{output_dir}/{self.filename}.txt", "w") as file_open:
            file_open.writelines(self.txt_output)

        return best_value

    def getw(self, h, only_even=True):
        w = h * self.ar
        w = int(round(w))
        if only_even:
            w = w // 2 * 2

        return w

    def analyze_results(self, vals):
        ratios = [0.0]
        for i in range(1, len(vals)):
            last = vals[i - 1]
            current = vals[i]
            ratios.append(current and last / current)
        sorted_array = sorted(ratios, reverse=True)  # make a copy of the array because we need the unsorted array later
        max_difference = sorted_array[0]

        differences = [s for s in sorted_array if s - 1 > (max_difference - 1) * 0.33][:5]

        for diff in differences:
            current = ratios.index(diff)
            # don't allow results within 20px of each other
            for res in self.resolutions:
                if res - 20 < current < res + 20:
                    break
            else:
                self.resolutions.append(current)

        bicubic_params = self.kernel == 'bicubic' and f'Scaling parameters:\nb = {self.b:.2f}\nc = {self.c:.2f}\n' or ''
        best_values = f"{'p, '.join([str(r + self.min_h) for r in self.resolutions])}p"
        self.txt_output += f"Resize Kernel: {self.kernel}\n{bicubic_params}Native resolution(s) (best guess): " \
                           f"{best_values}\nPlease check the graph manually for more accurate results\n\n"

        return ratios, vals, f"Native resolution(s) (best guess): {best_values}"

    def save_plot(self, vals):
        import matplotlib as mpl
        if not self.show_plot:
            mpl.use('Agg')
        import matplotlib.pyplot

        matplotlib.pyplot.style.use('dark_background')
        matplotlib.pyplot.plot(range(self.min_h, self.max_h + 1), vals, '.w-')
        matplotlib.pyplot.title(self.filename)
        matplotlib.pyplot.ylabel('Relative error')
        matplotlib.pyplot.xlabel('Resolution')
        matplotlib.pyplot.yscale(self.plot_scaling)
        matplotlib.pyplot.savefig(f'{output_dir}/{self.filename}.{self.plot_format}')
        if self.show_plot:
            matplotlib.pyplot.show()
        matplotlib.pyplot.clf()

    # Original idea by Chibi_goku http://recensubshq.forumfree.it/?t=64839203
    # Vapoursynth port by MonoS @github: https://github.com/MonoS/VS-MaskDetail
    def mask_detail(self, clip, final_width, final_height):
        resizer = descale_approx if self.approx else descale_accurate
        temp = resizer(clip, final_width, final_height, self.kernel, self.b, self.c, self.taps)
        temp = upscale(temp, clip.width, clip.height, self.kernel, self.b, self.c, self.taps)
        mask = core.std.Expr([clip, temp], 'x y - abs dup 0.015 > swap 16 * 0 ?').std.Inflate()
        mask = upscale(mask, final_width, final_height, "spline36", self.b, self.c, taps=self.taps)

        return change_bitdepth(mask, dither_type="none")

    # TODO: use PIL for output
    def save_images(self, src_luma32):
        resizer = descale_approx if self.approx else descale_accurate
        src = src_luma32
        first_out = imwri.Write(change_bitdepth(src), 'png', f'{output_dir}/{self.filename}_source%d.png')
        first_out.get_frame(0)  # trick vapoursynth into rendering the frame
        for r in self.resolutions:
            r += self.min_h
            image = self.mask_detail(src, self.getw(r), r)
            mask_out = imwri.Write(change_bitdepth(image), 'png', f'{output_dir}/{self.filename}_mask_{r:d}p%d.png')
            mask_out.get_frame(0)
            descale_out = resizer(src, self.getw(r), r, self.kernel, self.b, self.c, self.taps)
            descale_out = imwri.Write(change_bitdepth(descale_out), 'png', f'{output_dir}/{self.filename}_{r:d}p%d.png')
            descale_out.get_frame(0)

    def get_filename(self):
        return ''.join([
            f"f_{self.frame}",
            f"_k_{self.kernel}",
            f"_ar_{self.ar:.2f}",
            f"_{self.min_h}-{self.max_h}",
            f"_b_{self.b:.2f}_c_{self.c:.2f}" if self.kernel == "bicubic" else "",
            f"_taps_{self.taps}" if self.kernel == "lanczos" else "",
            f"_[approximation]" if self.approx else "",
        ])


def upscale(src, width, height, kernel, b, c, taps):
    resizer = getattr(src.resize, kernel.title())
    if not resizer:
        return src.fmtc.resample(width, height, kernel=kernel, a1=b, a2=c, taps=taps)
    if kernel == 'bicubic':
        resizer = partial(resizer, filter_param_a=b, filter_param_b=c)
    elif kernel == 'lanczos':
        resizer = partial(resizer, filter_param_a=taps)

    return resizer(width, height)


def descale_accurate(src, width, height, kernel, b, c, taps):
    descale = getattr(src, 'descale_getnative', None)
    if descale is None:
        descale = getattr(src, 'descale')
    descale = getattr(descale, 'De' + kernel)
    if kernel == 'bicubic':
        descale = partial(descale, b=b, c=c)
    elif kernel == 'lanczos':
        descale = partial(descale, taps=taps)

    return descale(width, height)


def descale_approx(src, width, height, kernel, b, c, taps):
    return src.fmtc.resample(width, height, kernel=kernel, taps=taps, a1=b, a2=c, invks=True, invkstaps=taps)


def change_bitdepth(src, bits=8, dither_type='error_diffusion'):
    src_f = src.format
    out_f = core.register_format(src_f.color_family,
                                 vapoursynth.INTEGER,
                                 bits,
                                 src_f.subsampling_w,
                                 src_f.subsampling_h)

    return core.resize.Point(src, format=out_f.id, dither_type=dither_type)

    # r39+
    # return src.resize.Point(format=src.format.replace(bits_per_sample=bits, dither_type=dither_type))


def to_float(str_value):
    if set(str_value) - set("0123456789./"):
        raise argparse.ArgumentTypeError("Invalid characters in float parameter")
    try:
        return eval(str_value) if "/" in str_value else float(str_value)
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError):
        raise argparse.ArgumentTypeError("Exception while parsing float") from None


def get_attr(obj, attr, default=None):
    for ele in attr.split('.'):
        obj = getattr(obj, ele, default)
        if obj == default:
            return default
    return obj


def get_source_filter(args):
    ext = os.path.splitext(args.input_file)[1].lower()
    if imwri and (args.img or ext in {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".webp", ".tga", ".jp2"}):
        return imwri.Read
    source_filter = get_attr(core, 'ffms2.Source')
    if source_filter:
        return source_filter
    source_filter = get_attr(core, 'lsmas.LWLibavSource')
    if source_filter:
        return source_filter
    source_filter = get_attr(core, 'lsmas.LSMASHVideoSource')
    if source_filter:
        return source_filter
    raise ValueError("No source filter found.")


parser = argparse.ArgumentParser(description='Find the native resolution(s) of upscaled material (mostly anime)')
parser.add_argument(dest='input_file', type=str, help='Absolute or relative path to the input file')
parser.add_argument('--frame', '-f', dest='frame', type=int, default=None, help='Specify a frame for the analysis. Random if unspecified')
parser.add_argument('--kernel', '-k', dest='kernel', type=str.lower, default='bilinear', help='Resize kernel to be used')
parser.add_argument('--bicubic-b', '-b', dest='b', type=to_float, default="1/3", help='B parameter of bicubic resize')
parser.add_argument('--bicubic-c', '-c', dest='c', type=to_float, default="1/3", help='C parameter of bicubic resize')
parser.add_argument('--lanczos-taps', '-t', dest='taps', type=int, default=3, help='Taps parameter of lanczos resize')
parser.add_argument('--aspect-ratio', '-ar', dest='ar', type=to_float, default=0, help='Force aspect ratio. Only useful for anamorphic input')
parser.add_argument('--approx', '-ap', dest="approx", action="store_true", help='Use fmtc instead of descale [faster, loss of accuracy]')
parser.add_argument('--min-heigth', '-min', dest="min_h", type=int, default=500, help='Minimum height to consider')
parser.add_argument('--max-heigth', '-max', dest="max_h", type=int, default=1000, help='Maximum height to consider')
parser.add_argument('--use', '-u', help='Use specified source filter e.g. (lsmas.LWLibavSource)')
parser.add_argument('--is-image', '-img', dest='img', action="store_true", help='Force image input')
parser.add_argument('--generate-images', '-img-out', dest='img_out', action="store_true", help='Save detail mask as png')
parser.add_argument('--plot-scaling', '-ps', dest='plot_scaling', type=str.lower, default='log', help='Scaling of the y axis. Can be "linear" or "log"')
parser.add_argument('--plot-format', '-pf', dest='plot_format', type=str.lower, default='svg', help='Format of the output image. Can be svg, png, pdf, rgba, jp(e)g, tif(f), and probably more')
parser.add_argument('--show-plot-gui', '-pg', dest='show_plot', action="store_true", help='Show an interactive plot gui window.')


def getnative():
    starttime = time.time()
    args = parser.parse_args()

    if (args.img or args.img_out) and imwri is None:
        raise ValueError("imwri not found.")

    if args.approx:
        if not hasattr(core, 'fmtc'):
            raise ValueError('fmtc not found')

        try:
            core.fmtc.resample(core.std.BlankClip(), kernel=args.kernel)
        except vapoursynth.Error:
            raise ValueError('fmtc: Invalid kernel specified.')
    else:
        if not hasattr(core, 'descale_getnative'):
            if not hasattr(core, 'descale'):
                raise ValueError('Neither descale_getnative nor descale found.\n'
                                 'One of them is needed for accurate descaling')
            print("Warning: only the really really slow descale is available.\n"
                  "Download the modified descale for improved performance:\n"
                  "https://github.com/Infiziert90/vapoursynth-descale")

        if args.kernel not in ['spline36', 'spline16', 'lanczos', 'bicubic', 'bilinear']:
            raise ValueError(f'descale: {args.kernel} is not a supported kernel. Try -ap for approximation.')

    if args.use:
        source_filter = get_attr(core, args.use)
        if not source_filter:
            raise ValueError(f"{args.use} is not available in the current vapoursynth enviroment.")
    else:
        source_filter = get_source_filter(args)

    src = source_filter(args.input_file)
    if args.frame is None:
        args.frame = src.num_frames // 3

    if args.ar is 0:
        args.ar = src.width / src.height

    if args.min_h >= src.height:
        raise ValueError(f"Picture is to small or equal for min height {args.min_h}.")
    elif args.min_h >= args.max_h:
        raise ValueError(f"Your min height is bigger or equal to max height.")
    elif args.max_h > src.height:
        print(f"Your max height cant be bigger than your image dimensions. New max height is {src.height}")
        args.max_h = src.height

    kwargs = args.__dict__.copy()
    del kwargs["input_file"]
    del kwargs["use"]
    del kwargs["img"]

    get_native = GetNative(src, **kwargs)
    try:
        loop = asyncio.get_event_loop()
        best_value = loop.run_until_complete(get_native.run())
    except ValueError as err:
        return print(f"Error in getnative: {err}")

    content = ''.join([
        f"\nKernel: {args.kernel} ",
        f"AR: {args.ar:.2f} ",
        f"B: {args.b:.2f} C: {args.c:.2f} " if args.kernel == "bicubic" else "",
        f"Taps: {args.taps} " if args.kernel == "lanczos" else "",
        f"\n{best_value}",
        f"\n[approximation]" if args.approx else "",
    ])
    print(content)
    print('done in {:.2f} s'.format(time.time() - starttime))


if __name__ == '__main__':
    print("Start getnative.")
    getnative()
