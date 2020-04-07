import gc
import time
import argparse
import asyncio
import vapoursynth
import os
from functools import partial
from typing import Union, List, Tuple
try:
    import matplotlib as mpl
    import matplotlib.pyplot as pyplot
except BaseException:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as pyplot

"""
Rework by Infi
Original Author: kageru https://gist.github.com/kageru/549e059335d6efbae709e567ed081799
Thanks: BluBb_mADe, FichteFoll, stux!, Frechdachs

Version: 1.3.1
"""

core = vapoursynth.core
core.add_cache = False
imwri = getattr(core, "imwri", getattr(core, "imwrif", None))
output_dir = os.path.splitext(os.path.basename(__file__))[0]
_modes = ["bilinear", "bicubic", "bl-bc", "all"]


class GetnativeException(BaseException):
    pass


class DefineScaler:
    def __init__(self, kernel: str, b: Union[float, int]=0, c: Union[float, int]=0, taps: Union[float, int]=0):
        """
        Get a scaler for getnative from descale

        :param kernel: kernel for descale
        :param b: b value for kernel "bicubic" (default 0)
        :param c: c value for kernel "bicubic" (default 0)
        :param taps: taps value for kernel "lanczos" (default 0)
        """
        self.kernel = kernel
        self.b = b
        self.c = c
        self.taps = taps
        self.check_input()
        self.descaler = self.get_descaler()
        self.upscaler = self.get_upscaler()

    def get_descaler(self):
        descaler = getattr(core, 'descale_getnative', getattr(core, 'descale'))
        descaler = getattr(descaler, 'De' + self.kernel)
        if self.kernel == 'bicubic':
            descaler = partial(descaler, b=self.b, c=self.c)
        elif self.kernel == 'lanczos':
            descaler = partial(descaler, taps=self.taps)

        return descaler

    def get_upscaler(self):
        upscaler = getattr(core.resize, self.kernel.title())
        if self.kernel == 'bicubic':
            upscaler = partial(upscaler, filter_param_a=self.b, filter_param_b=self.c)
        elif self.kernel == 'lanczos':
            upscaler = partial(upscaler, filter_param_a=self.taps)

        return upscaler

    def check_input(self):
        if self.kernel not in ['spline36', 'spline16', 'lanczos', 'bicubic', 'bilinear']:
            raise GetnativeException(f'descale: {self.kernel} is not a supported kernel.')


scaler_dict = {
    "Bilinear": DefineScaler("bilinear"),
    "Bicubic (b=1/3, c=1/3)": DefineScaler("bicubic", b=1/3, c=1/3),
    "Bicubic (b=0.5, c=0)": DefineScaler("bicubic", b=.5, c=0),
    "Bicubic (b=0, c=0.5)": DefineScaler("bicubic", b=0, c=.5),
    "Bicubic (b=1, c=0)": DefineScaler("bicubic", b=1, c=0),
    "Bicubic (b=0, c=1)": DefineScaler("bicubic", b=0, c=1),
    "Bicubic (b=0.2, c=0.5)": DefineScaler("bicubic", b=.2, c=.5),
    "Lanczos (3 Taps)": DefineScaler("lanczos", taps=3),
    "Lanczos (4 Taps)": DefineScaler("lanczos", taps=4),
    "Lanczos (5 Taps)": DefineScaler("lanczos", taps=5),
    "Spline16": DefineScaler("spline16"),
    "Spline36": DefineScaler("spline36"),
    }


class GetNative:
    def __init__(self, src, scaler, ar, min_h, max_h, frame, img_out, plot_scaling, plot_format, show_plot, no_save):
        self.plot_format = plot_format
        self.plot_scaling = plot_scaling
        self.src = src
        self.min_h = min_h
        self.max_h = max_h
        self.ar = ar
        self.scaler = scaler
        self.frame = frame
        self.img_out = img_out
        self.show_plot = show_plot
        self.no_save = no_save
        self.txt_output = ""
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
        resizer = self.scaler.descaler
        upscaler = self.scaler.upscaler
        clip_list = []
        for h in range(self.min_h, self.max_h + 1):
            clip_list.append(resizer(src_luma32, self.getw(h), h))
        full_clip = core.std.Splice(clip_list, mismatch=True)
        full_clip = upscaler(full_clip, self.getw(src.height), src.height)
        if self.ar != src.width / src.height:
            src_luma32 = upscaler(src_luma32, self.getw(src.height), src.height)
        expr_full = core.std.Expr([src_luma32 * full_clip.num_frames, full_clip], 'x y - abs dup 0.015 > swap 0 ?')
        full_clip = core.std.CropRel(expr_full, 5, 5, 5, 5)
        full_clip = core.std.PlaneStats(full_clip)
        full_clip = core.std.Cache(full_clip)

        tasks_pending = set()
        futures = {}
        vals = []
        full_clip_len = len(full_clip)
        for frame_index in range(len(full_clip)):
            print(f"{frame_index+1}/{full_clip_len}", end="\r")
            fut = asyncio.ensure_future(asyncio.wrap_future(full_clip.get_frame_async(frame_index)))
            tasks_pending.add(fut)
            futures[fut] = frame_index
            while len(tasks_pending) >= core.num_threads + 2:
                tasks_done, tasks_pending = await asyncio.wait(tasks_pending, return_when=asyncio.FIRST_COMPLETED)
                vals += [(futures.pop(task), task.result().props.PlaneStatsAverage) for task in tasks_done]

        tasks_done, _ = await asyncio.wait(tasks_pending)
        vals += [(futures.pop(task), task.result().props.PlaneStatsAverage) for task in tasks_done]
        vals = [v for _, v in sorted(vals)]
        ratios, vals, best_value = self.analyze_results(vals)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        plot = self.save_plot(vals)
        if not self.no_save and self.img_out:
            self.save_images(src_luma32)

        self.txt_output += 'Raw data:\nResolution\t | Relative Error\t | Relative difference from last\n'
        for i, error in enumerate(vals):
            self.txt_output += f'{i + self.min_h:4d}\t\t | {error:.10f}\t\t\t | {ratios[i]:.2f}\n'

        if not self.no_save:
            with open(f"{output_dir}/{self.filename}.txt", "w") as stream:
                stream.writelines(self.txt_output)

        return best_value, plot, self.resolutions

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

        scaler = self.scaler
        bicubic_params = scaler.kernel == 'bicubic' and f'Scaling parameters:\nb = {scaler.b:.2f}\nc = {scaler.c:.2f}\n' or ''
        best_values = f"{'p, '.join([str(r + self.min_h) for r in self.resolutions])}p"
        self.txt_output += f"Resize Kernel: {scaler.kernel}\n{bicubic_params}Native resolution(s) (best guess): " \
                           f"{best_values}\nPlease check the graph manually for more accurate results\n\n"

        return ratios, vals, f"Native resolution(s) (best guess): {best_values}"

    def save_plot(self, vals):
        plot = pyplot
        plot.close('all')
        plot.style.use('dark_background')
        plot.plot(range(self.min_h, self.max_h + 1), vals, '.w-')
        plot.title(self.filename)
        plot.ylabel('Relative error')
        plot.xlabel('Resolution')
        plot.yscale(self.plot_scaling)
        if not self.no_save:
            for fmt in self.plot_format.split(','):
                plot.savefig(f'{output_dir}/{self.filename}.{fmt}')
        if self.show_plot:
            plot.show()

        return plot

    # Original idea by Chibi_goku http://recensubshq.forumfree.it/?t=64839203
    # Vapoursynth port by MonoS @github: https://github.com/MonoS/VS-MaskDetail
    def mask_detail(self, clip, final_width, final_height):
        temp = self.scaler.descaler(clip, final_width, final_height)
        temp = self.scaler.upscaler(temp, clip.width, clip.height)
        mask = core.std.Expr([clip, temp], 'x y - abs dup 0.015 > swap 16 * 0 ?').std.Inflate()
        mask = scaler_dict['Spline36'].upscaler(mask, final_width, final_height)

        return mask

    # TODO: use PIL for output
    def save_images(self, src_luma32):
        src = src_luma32
        first_out = imwri.Write(src, 'png', f'{output_dir}/{self.filename}_source%d.png')
        first_out.get_frame(0)  # trick vapoursynth into rendering the frame
        for r in self.resolutions:
            r += self.min_h
            image = self.mask_detail(src, self.getw(r), r)
            mask_out = imwri.Write(image, 'png', f'{output_dir}/{self.filename}_mask_{r:d}p%d.png')
            mask_out.get_frame(0)
            descale_out = self.scaler.descaler(src, self.getw(r), r)
            descale_out = imwri.Write(descale_out, 'png', f'{output_dir}/{self.filename}_{r:d}p%d.png')
            descale_out.get_frame(0)

    def get_filename(self):
        return ''.join([
            f"f_{self.frame}",
            f"_k_{self.scaler.kernel}",
            f"_ar_{self.ar:.2f}",
            f"_{self.min_h}-{self.max_h}",
            f"_b_{self.scaler.b:.2f}_c_{self.scaler.c:.2f}" if self.scaler.kernel == "bicubic" else "",
            f"_taps_{self.scaler.taps}" if self.scaler.kernel == "lanczos" else "",
        ])


def getnative(args: Union[List, argparse.Namespace], src: vapoursynth.VideoNode, scaler: Union[DefineScaler, None]) -> Tuple[List, pyplot.plot]:
    """
    Process your VideoNode with the getnative algorithm and return the result and a plot object

    :param args: List of all arguments for argparse or Namespace object from argparse
    :param src: VideoNode from vapoursynth
    :param scaler: DefineScaler object or None
    :return: best resolutions list and plot matplotlib.pyplot
    """
    if type(args) == list:
        args = parser.parse_args(args)

    if (args.img or args.img_out) and imwri is None:
        raise GetnativeException("imwri not found.")

    if "toggaf.asi.xe" not in core.get_plugins():
        if not hasattr(core, 'descale'):
            raise GetnativeException('No descale found.\nIt is needed for accurate descaling')
        print("Warning: only the really really slow descale is available. (See README for help)\n")

    if scaler:
        scaler = scaler
    else:
        scaler = DefineScaler(args.kernel, b=args.b, c=args.c, taps=args.taps)

    if args.frame is None:
        args.frame = src.num_frames // 3
    elif args.frame < 0:
        args.frame = src.num_frames // -args.frame
    elif args.frame > src.num_frames - 1:
        raise GetnativeException(f"Frame is incorrect: {args.number_frames - 1}")

    if args.ar == 0:
        args.ar = src.width / src.height

    if args.min_h >= src.height:
        raise GetnativeException(f"Input image is smaller than min height")
    elif args.min_h >= args.max_h:
        raise GetnativeException(f"Min height must be smaller than max height")
    elif args.max_h > src.height:
        print(f"Your max height is over the image dimensions. New max height is {src.height}")
        args.max_h = src.height

    getn = GetNative(src, scaler, args.ar, args.min_h, args.max_h, args.frame, args.img_out, args.plot_scaling,
                     args.plot_format, args.show_plot, args.no_save)
    try:
        loop = asyncio.get_event_loop()
        best_value, plot, resolutions = loop.run_until_complete(getn.run())
    except ValueError as err:
        raise GetnativeException(f"Error in getnative: {err}")

    content = ''.join([
        f"\nKernel: {scaler.kernel} ",
        f"AR: {args.ar:.2f} ",
        f"B: {scaler.b:.2f} C: {scaler.c:.2f} " if scaler.kernel == "bicubic" else "",
        f"Taps: {scaler.taps} " if scaler.kernel == "lanczos" else "",
        f"\n{best_value}",
    ])
    gc.collect()
    print(content)

    return resolutions, plot


def _getnative():
    args = parser.parse_args()

    if args.use:
        source_filter = _get_attr(core, args.use)
        if not source_filter:
            raise GetnativeException(f"{args.use} is not available in the current vapoursynth enviroment.")
        print(f"Using {args.use} as source filter")
    else:
        source_filter = _get_source_filter(args)

    src = source_filter(args.input_file)

    if args.mode == "bilinear" or args.mode == "bl-bc":
        getnative(args, src, scaler_dict["Bilinear"])
    if args.mode == "bicubic" or args.mode == "bl-bc":  # IF is needed for bl-bc run
        for name, scaler in scaler_dict.items():
            if "bicubic" in name.lower():
                getnative(args, src, scaler)
    elif args.mode == "all":
        for scaler in scaler_dict.values():
            getnative(args, src, scaler)
    elif args.mode != "bilinear":  # ELIF is needed for bl-bc run
        getnative(args, src, None)


def _vpy_source_filter(path):
    import runpy
    runpy.run_path(path, {}, "__vapoursynth__")
    return vapoursynth.get_output(0)


def _get_source_filter(args):
    ext = os.path.splitext(args.input_file)[1].lower()
    if imwri and (args.img or ext in {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".webp", ".tga", ".jp2"}):
        print("Using imwri as source filter")
        return imwri.Read
    if ext in {".py", ".pyw", ".vpy"}:
        print("Using custom VapourSynth script as a source. This may cause garbage results. Only do this if you know what you are doing.")
        return _vpy_source_filter
    source_filter = _get_attr(core, 'ffms2.Source')
    if source_filter:
        print("Using ffms2 as source filter")
        return lambda input_file: source_filter(input_file, alpha=False)
    source_filter = _get_attr(core, 'lsmas.LWLibavSource')
    if source_filter:
        print("Using lsmas.LWLibavSource as source filter")
        return source_filter
    source_filter = _get_attr(core, 'lsmas.LSMASHVideoSource')
    if source_filter:
        print("Using lsmas.LSMASHVideoSource as source filter")
        return source_filter
    raise GetnativeException("No source filter found.")


def _to_float(str_value):
    if set(str_value) - set("0123456789./"):
        raise argparse.ArgumentTypeError("Invalid characters in float parameter")
    try:
        return eval(str_value) if "/" in str_value else float(str_value)
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError):
        raise argparse.ArgumentTypeError("Exception while parsing float") from None


def _get_attr(obj, attr, default=None):
    for ele in attr.split('.'):
        obj = getattr(obj, ele, default)
        if obj == default:
            return default
    return obj


parser = argparse.ArgumentParser(description='Find the native resolution(s) of upscaled material (mostly anime)')
parser.add_argument('--frame', '-f', dest='frame', type=int, default=None, help='Specify a frame for the analysis. Random if unspecified. Negative frame numbers for a frame like this: src.num_frames // -args.frame')
parser.add_argument('--kernel', '-k', dest='kernel', type=str.lower, default="bicubic", help='Resize kernel to be used')
parser.add_argument('--bicubic-b', '-b', dest='b', type=_to_float, default="1/3", help='B parameter of bicubic resize')
parser.add_argument('--bicubic-c', '-c', dest='c', type=_to_float, default="1/3", help='C parameter of bicubic resize')
parser.add_argument('--lanczos-taps', '-t', dest='taps', type=int, default=3, help='Taps parameter of lanczos resize')
parser.add_argument('--aspect-ratio', '-ar', dest='ar', type=_to_float, default=0, help='Force aspect ratio. Only useful for anamorphic input')
parser.add_argument('--min-height', '-min', dest="min_h", type=int, default=500, help='Minimum height to consider')
parser.add_argument('--max-height', '-max', dest="max_h", type=int, default=1000, help='Maximum height to consider')
parser.add_argument('--generate-images', '-img-out', dest='img_out', action="store_true", default=False, help='Save detail mask as png')
parser.add_argument('--plot-scaling', '-ps', dest='plot_scaling', type=str.lower, default='log', help='Scaling of the y axis. Can be "linear" or "log"')
parser.add_argument('--plot-format', '-pf', dest='plot_format', type=str.lower, default='svg', help='Format of the output image. Specify multiple formats separated by commas. Can be svg, png, pdf, rgba, jp(e)g, tif(f), and probably more')
parser.add_argument('--show-plot-gui', '-pg', dest='show_plot', action="store_true", default=False, help='Show an interactive plot gui window.')
parser.add_argument('--no-save', '-ns', dest='no_save', action="store_true", default=False, help='Do not save files to disk.')
parser.add_argument('--is-image', '-img', dest='img', action="store_true", default=False, help='Force image input')
if __name__ == '__main__':
    parser.add_argument(dest='input_file', type=str, help='Absolute or relative path to the input file')
    parser.add_argument('--use', '-u', default=None, help='Use specified source filter e.g. (lsmas.LWLibavSource)')
    parser.add_argument('--mode', '-m', dest='mode', type=str, choices=_modes, default=None, help='Choose a predefined mode ["bilinear", "bicubic", "bl-bc", "all"]')

    starttime = time.time()
    _getnative()
    print(f'done in {time.time() - starttime:.2f}s')
