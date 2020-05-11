import gc
import os
import time
import asyncio
import argparse
import vapoursynth
from pathlib import Path
from functools import partial
from typing import Union, List, Tuple
from getnative.utils import GetnativeException, plugin_from_identifier, get_attr, get_source_filter, to_float
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
Thanks: BluBb_mADe, FichteFoll, stux!, Frechdachs, LittlePox
"""

core = vapoursynth.core
core.add_cache = False
imwri = getattr(core, "imwri", getattr(core, "imwrif", None))
_modes = ["bilinear", "bicubic", "bl-bc", "all"]


class _DefineScaler:
    def __init__(self, kernel: str, b: Union[float, int] = 0, c: Union[float, int] = 0, taps: int = 0):
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
        _d = plugin_from_identifier(core, "tegaf.asi.xe")
        self.plugin = _d if _d is not None else get_attr(core, 'descale', None)
        if self.plugin is None:
            return

        self.descaler = getattr(self.plugin, f'De{self.kernel}', None)
        self.upscaler = getattr(core.resize, self.kernel.title())

        self.check_input()
        self.check_for_extra_paras()

    def check_for_extra_paras(self):
        if self.kernel == 'bicubic':
            self.descaler = partial(self.descaler, b=self.b, c=self.c)
            self.upscaler = partial(self.upscaler, filter_param_a=self.b, filter_param_b=self.c)
        elif self.kernel == 'lanczos':
            self.descaler = partial(self.descaler, taps=self.taps)
            self.upscaler = partial(self.upscaler, filter_param_a=self.taps)

    def check_input(self):
        if self.descaler is None:
            raise GetnativeException(f'descale: {self.kernel} is not a supported kernel.')

    def __str__(self):
        return (
            f"{self.kernel.capitalize()}"
            f"{'' if self.kernel != 'bicubic' else f' b {self.b:.2f} c {self.c:.2f}'}"
            f"{'' if self.kernel != 'lanczos' else f' taps {self.taps}'}"
        )

    def __repr__(self):
        return (
            f"ScalerObject: "
            f"{self.kernel.capitalize()}"
            f"{'' if self.kernel != 'bicubic' else f' b {self.b:.2f} c {self.c:.2f}'}"
            f"{'' if self.kernel != 'lanczos' else f' taps {self.taps}'}"
        )


common_scaler = {
    "bilinear": [_DefineScaler("bilinear")],
    "bicubic": [
        _DefineScaler("bicubic", b=1 / 3, c=1 / 3),
        _DefineScaler("bicubic", b=.5, c=0),
        _DefineScaler("bicubic", b=0, c=.5),
        _DefineScaler("bicubic", b=1, c=0),
        _DefineScaler("bicubic", b=0, c=1),
        _DefineScaler("bicubic", b=.2, c=.5),
        _DefineScaler("bicubic", b=.5, c=.5),
    ],
    "lanczos": [
        _DefineScaler("lanczos", taps=2),
        _DefineScaler("lanczos", taps=3),
        _DefineScaler("lanczos", taps=4),
        _DefineScaler("lanczos", taps=5),
    ],
    "spline": [
        _DefineScaler("spline16"),
        _DefineScaler("spline36"),
        _DefineScaler("spline64"),
    ]
}


class GetNative:
    def __init__(self, src, scaler, ar, min_h, max_h, frame, mask_out, plot_scaling, plot_format, show_plot, no_save,
                 steps, output_dir):
        self.plot_format = plot_format
        self.plot_scaling = plot_scaling
        self.src = src
        self.min_h = min_h
        self.max_h = max_h
        self.ar = ar
        self.scaler = scaler
        self.frame = frame
        self.mask_out = mask_out
        self.show_plot = show_plot
        self.no_save = no_save
        self.steps = steps
        self.output_dir = output_dir
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
        clip_list = [self.scaler.descaler(src_luma32, self.getw(h), h)
                     for h in range(self.min_h, self.max_h + 1, self.steps)]
        full_clip = core.std.Splice(clip_list, mismatch=True)
        full_clip = self.scaler.upscaler(full_clip, self.getw(src.height), src.height)
        if self.ar != src.width / src.height:
            src_luma32 = self.scaler.upscaler(src_luma32, self.getw(src.height), src.height)
        expr_full = core.std.Expr([src_luma32 * full_clip.num_frames, full_clip], 'x y - abs dup 0.015 > swap 0 ?')
        full_clip = core.std.CropRel(expr_full, 5, 5, 5, 5)
        full_clip = core.std.PlaneStats(full_clip)
        full_clip = core.std.Cache(full_clip)

        tasks_pending = set()
        futures = {}
        vals = []
        full_clip_len = len(full_clip)
        for frame_index in range(len(full_clip)):
            print(f"\r{frame_index}/{full_clip_len-1}", end="")
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
        print("\n")  # move the cursor, so that you not start at the end of the progress bar

        self.txt_output += 'Raw data:\nResolution\t | Relative Error\t | Relative difference from last\n'
        self.txt_output += '\n'.join([
            f'{i * self.steps + self.min_h:4d}\t\t | {error:.10f}\t\t | {ratios[i]:.2f}'
            for i, error in enumerate(vals)
        ])

        plot, fig = self.save_plot(vals)
        if not self.no_save:
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)

            print(f"Output Path: {self.output_dir}")
            for fmt in self.plot_format.replace(" ", "").split(','):
                fig.savefig(f'{self.output_dir}/{self.filename}.{fmt}')

            with open(f"{self.output_dir}/{self.filename}.txt", "w") as stream:
                stream.writelines(self.txt_output)

            if self.mask_out:
                self.save_images(src_luma32)

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

        best_values = (
            f"Native resolution(s) (best guess): "
            f"{'p, '.join([str(r * self.steps + self.min_h) for r in self.resolutions])}p"
        )
        self.txt_output = (
            f"Resize Kernel: {self.scaler}\n"
            f"{best_values}\n"
            f"Please check the graph manually for more accurate results\n\n"
        )

        return ratios, vals, best_values

    # Modified from:
    # https://github.com/WolframRhodium/muvsfunc/blob/d5b2c499d1b71b7689f086cd992d9fb1ccb0219e/muvsfunc.py#L5807
    def save_plot(self, vals):
        plot = pyplot
        plot.close('all')
        plot.style.use('dark_background')
        fig, ax = plot.subplots(figsize=(12, 8))
        ax.plot(range(self.min_h, self.max_h + 1, self.steps), vals, '.w-')
        dh_sequence = tuple(range(500, 1001, self.steps))
        ticks = tuple(dh for i, dh in enumerate(dh_sequence) if i % (50 // self.steps) == 0)
        ax.set(xlabel="Height", xticks=ticks, ylabel="Relative error", title=self.filename, yscale="log")
        if self.show_plot:
            plot.show()

        return plot, fig

    # Original idea by Chibi_goku http://recensubshq.forumfree.it/?t=64839203
    # Vapoursynth port by MonoS @github: https://github.com/MonoS/VS-MaskDetail
    def mask_detail(self, clip, final_width, final_height):
        temp = self.scaler.descaler(clip, final_width, final_height)
        temp = self.scaler.upscaler(temp, clip.width, clip.height)
        mask = core.std.Expr([clip, temp], 'x y - abs dup 0.015 > swap 16 * 0 ?').std.Inflate()
        mask = _DefineScaler(kernel="spline36").upscaler(mask, final_width, final_height)

        return mask

    def save_images(self, src_luma32):
        src = src_luma32
        first_out = imwri.Write(src, 'png', f'{self.output_dir}/{self.filename}_source%d.png')
        first_out.get_frame(0)  # trick vapoursynth into rendering the frame
        for r in self.resolutions:
            r = r * self.steps + self.min_h
            image = self.mask_detail(src, self.getw(r), r)
            mask_out = imwri.Write(image, 'png', f'{self.output_dir}/{self.filename}_mask_{r:d}p%d.png')
            mask_out.get_frame(0)
            descale_out = self.scaler.descaler(src, self.getw(r), r)
            descale_out = imwri.Write(descale_out, 'png', f'{self.output_dir}/{self.filename}_{r:d}p%d.png')
            descale_out.get_frame(0)

    def get_filename(self):
        return (
            f"f_{self.frame}"
            f"_{str(self.scaler).replace(' ', '_')}"
            f"_ar_{self.ar:.2f}"
            f"_steps_{self.steps}"
        )


def getnative(args: Union[List, argparse.Namespace], src: vapoursynth.VideoNode, scaler: Union[_DefineScaler, None],
              first_time: bool = True) -> Tuple[List, pyplot.plot]:
    """
    Process your VideoNode with the getnative algorithm and return the result and a plot object

    :param args: List of all arguments for argparse or Namespace object from argparse
    :param src: VideoNode from vapoursynth
    :param scaler: DefineScaler object or None
    :param first_time: prevents posting warnings multiple times
    :return: best resolutions list and plot matplotlib.pyplot
    """
    if type(args) == list:
        args = parser.parse_args(args)

    output_dir = Path(args.dir).resolve()
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Missing write permissions: {output_dir}")
    output_dir = output_dir.joinpath("results")

    if (args.img or args.mask_out) and imwri is None:
        raise GetnativeException("imwri not found.")

    if scaler is None:
        scaler = _DefineScaler(args.kernel, b=args.b, c=args.c, taps=args.taps)
    else:
        scaler = scaler

    if scaler.plugin is None:
        if "toggaf.asi.xe" in core.get_plugins():
            print("Error: descale_getnative support ended, pls use https://github.com/Irrational-Encoding-Wizardry/vapoursynth-descale")
        raise GetnativeException('No descale found!')

    if args.steps != 1 and first_time:
        print(
            "Warning for -steps/--stepping: "
            "If you are not completely sure what this parameter does, use the default step size.\n"
        )

    if args.frame is None:
        args.frame = src.num_frames // 3
    elif args.frame < 0:
        args.frame = src.num_frames // -args.frame
    elif args.frame > src.num_frames - 1:
        raise GetnativeException(f"Last frame is {src.num_frames - 1}, but you want {args.frame}")

    if args.ar == 0:
        args.ar = src.width / src.height

    if args.min_h >= src.height:
        raise GetnativeException(f"Input image {src.height} is smaller min_h {args.min_h}")
    elif args.min_h >= args.max_h:
        raise GetnativeException(f"min_h {args.min_h} > max_h {args.max_h}? Not processable")
    elif args.max_h > src.height:
        print(f"The image height is {src.height}, going higher is stupid! New max_h {src.height}")
        args.max_h = src.height

    getn = GetNative(src, scaler, args.ar, args.min_h, args.max_h, args.frame, args.mask_out, args.plot_scaling,
                     args.plot_format, args.show_plot, args.no_save, args.steps, output_dir)
    try:
        loop = asyncio.get_event_loop()
        best_value, plot, resolutions = loop.run_until_complete(getn.run())
    except ValueError as err:
        raise GetnativeException(f"Error in getnative: {err}")

    gc.collect()
    print(
        f"\n{scaler} AR: {args.ar:.2f} Steps: {args.steps}\n"
        f"{best_value}\n"
    )

    return resolutions, plot


def _getnative():
    args = parser.parse_args()

    if args.use:
        source_filter = get_attr(core, args.use)
        if not source_filter:
            raise GetnativeException(f"{args.use} is not available.")
        print(f"Using {args.use} as source filter")
    else:
        source_filter = get_source_filter(core, imwri, args)

    src = source_filter(args.input_file)

    mode = [None]  # default
    if args.mode == "bilinear":
        mode = [common_scaler["bilinear"][0]]
    elif args.mode == "bicubic":
        mode = [scaler for scaler in common_scaler["bicubic"]]
    elif args.mode == "bl-bc":
        mode = [scaler for scaler in common_scaler["bicubic"]]
        mode.append(common_scaler["bilinear"][0])
    elif args.mode == "all":
        mode = [s for scaler in common_scaler.values() for s in scaler]

    for i, scaler in enumerate(mode):
        if scaler is not None and scaler.plugin is None:
            print(f"Warning: No correct descale version found for {scaler}, continuing with next scaler when available.")
            continue
        getnative(args, src, scaler, first_time=True if i == 0 else False)


parser = argparse.ArgumentParser(description='Find the native resolution(s) of upscaled material (mostly anime)')
parser.add_argument('--frame', '-f', dest='frame', type=int, default=None, help='Specify a frame for the analysis. Random if unspecified. Negative frame numbers for a frame like this: src.num_frames // -args.frame')
parser.add_argument('--kernel', '-k', dest='kernel', type=str.lower, default="bicubic", help='Resize kernel to be used')
parser.add_argument('--bicubic-b', '-b', dest='b', type=to_float, default="1/3", help='B parameter of bicubic resize')
parser.add_argument('--bicubic-c', '-c', dest='c', type=to_float, default="1/3", help='C parameter of bicubic resize')
parser.add_argument('--lanczos-taps', '-t', dest='taps', type=int, default=3, help='Taps parameter of lanczos resize')
parser.add_argument('--aspect-ratio', '-ar', dest='ar', type=to_float, default=0, help='Force aspect ratio. Only useful for anamorphic input')
parser.add_argument('--min-height', '-min', dest="min_h", type=int, default=500, help='Minimum height to consider')
parser.add_argument('--max-height', '-max', dest="max_h", type=int, default=1000, help='Maximum height to consider')
parser.add_argument('--output-mask', '-mask', dest='mask_out', action="store_true", default=False, help='Save detail mask as png')
parser.add_argument('--plot-scaling', '-ps', dest='plot_scaling', type=str.lower, default='log', help='Scaling of the y axis. Can be "linear" or "log"')
parser.add_argument('--plot-format', '-pf', dest='plot_format', type=str.lower, default='svg', help='Format of the output image. Specify multiple formats separated by commas. Can be svg, png, pdf, rgba, jp(e)g, tif(f), and probably more')
parser.add_argument('--show-plot-gui', '-pg', dest='show_plot', action="store_true", default=False, help='Show an interactive plot gui window.')
parser.add_argument('--no-save', '-ns', dest='no_save', action="store_true", default=False, help='Do not save files to disk. Disables all output arguments!')
parser.add_argument('--is-image', '-img', dest='img', action="store_true", default=False, help='Force image input')
parser.add_argument('--stepping', '-steps', dest='steps', type=int, default=1, help='This changes the way getnative will handle resolutions. Example steps=3 [500p, 503p, 506p ...]')
parser.add_argument('--output-dir', '-dir', dest='dir', type=str, default="", help='Sets the path of the output dir where you want all results to be saved. (/results will always be added as last folder)')
def main():
    parser.add_argument(dest='input_file', type=str, help='Absolute or relative path to the input file')
    parser.add_argument('--use', '-u', default=None, help='Use specified source filter e.g. (lsmas.LWLibavSource)')
    parser.add_argument('--mode', '-m', dest='mode', type=str, choices=_modes, default=None, help='Choose a predefined mode ["bilinear", "bicubic", "bl-bc", "all"]')
    starttime = time.time()
    _getnative()
    print(f'done in {time.time() - starttime:.2f}s')
