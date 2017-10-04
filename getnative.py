import time
import argparse
import asyncio
import vapoursynth
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot
import fvsfunc_getnative as fvs

"""
Reworke by Infi
Original Author: kageru https://gist.github.com/kageru/549e059335d6efbae709e567ed081799
Thanks: BluBb_mADe, FichteFoll, stux!, Frechdachs
"""

core = vapoursynth.core
core.add_cache = False
imwri = getattr(core, "imwri", getattr(core, "imwrif"))


class GetNative:
    def __init__(self, src, kernel=None, b=None, c=None, taps=None, ar=None, approx=None, min_h=None, max_h=None,
                 frame=None, img_out=None, plot_scaling=None, plot_format=None):
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
        self.resolutions = None
        self.filename = self.get_filename()

    async def run(self):

        if not self.approx and self.kernel not in ['spline36', 'spline16', 'lanczos', 'bicubic', 'bilinear']:
            return True, f'descale: {self.kernel} is not a supported kernel. Try -ap for approximation.'

        try:
            clip = core.std.BlankClip()
            core.fmtc.resample(clip, kernel=self.kernel)
        except vapoursynth.Error:
            return True, "fmtc: Invalid kernel specified."

        src = self.src[self.frame]
        if self.ar is 0:
            self.ar = src.width / src.height

        src_luma32 = core.resize.Point(src, format=vapoursynth.YUV444PS, matrix_s='709' if src.format.color_family ==
                                                                                           vapoursynth.RGB else None)
        src_luma32 = core.std.ShufflePlanes(src_luma32, 0, vapoursynth.GRAY)
        src_luma32 = core.std.Cache(src_luma32)

        # descale each individual frame
        resizer = core.fmtc.resample if self.approx else fvs.Resize
        clip_list = []
        for h in range(self.min_h, self.max_h + 1):
            clip_list.append(resizer(src_luma32, self.getw(h), h, kernel=self.kernel, a1=self.b, a2=self.c, invks=True,
                                     taps=self.taps))
        full_clip = core.std.Splice(clip_list, mismatch=True)
        full_clip = fvs.Resize(full_clip, self.getw(src.height), src.height, kernel=self.kernel, a1=self.b, a2=self.c,
                               taps=self.taps)
        if self.ar != src.width / src.height:
            src_luma32 = resizer(src_luma32, self.getw(src.height), src.height, kernel=self.kernel, a1=self.b, a2=self.c,
                                 taps=self.taps)
        full_clip = core.std.Expr([src_luma32 * full_clip.num_frames, full_clip], 'x y - abs dup 0.015 > swap 0 ?')
        full_clip = core.std.CropRel(full_clip, 5, 5, 5, 5)
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
        self.save_plot(vals)
        if self.img_out:
            self.save_images(src_luma32)
        self.txt_output += 'Raw data:\nResolution\t | Relative Error\t | Relative difference from last\n'
        for i, error in enumerate(vals):
            self.txt_output += f'{i + self.min_h:4d}\t\t | {error:.6f}\t\t\t | {ratios[i]:.2f}\n'

        with open(f"{self.filename}.txt", "w") as file_open:
            file_open.writelines(self.txt_output)

        return False, best_value

    def getw(self, h, only_even=True):
        w = h * self.ar
        w = int(round(w))
        if only_even:
            w = w // 2 * 2

        return w

    def analyze_results(self, vals):
        ratios = [0.0]
        resolutions = []
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
            for res in resolutions:
                if res - 20 < current < res + 20:
                    break
            else:
                resolutions.append(current)

        self.resolutions = resolutions
        bicubic_params = self.kernel == 'bicubic' and f'Scaling parameters:\nb = {self.b:.2f}\nc = {self.c:.2f}\n' or ''
        best_values = f"{'p, '.join([str(r + self.min_h) for r in resolutions])}p"
        self.txt_output += f"Resize Kernel: {self.kernel}\n{bicubic_params}Native resolution(s) (best guess): " \
                           f"{best_values}\nPlease check the graph manually for more accurate results\n\n"

        return ratios, vals, f"Native resolution(s) (best guess): {best_values}"

    def save_plot(self, vals):
        matplotlib.pyplot.style.use('dark_background')
        matplotlib.pyplot.plot(range(self.min_h, self.max_h + 1), vals, '.w-')
        matplotlib.pyplot.title(self.filename)
        matplotlib.pyplot.ylabel('Relative error')
        matplotlib.pyplot.xlabel('Resolution')
        matplotlib.pyplot.yscale(self.plot_scaling)
        matplotlib.pyplot.savefig(f'{self.filename}.' + self.plot_format)
        matplotlib.pyplot.clf()

    # Original idea by Chibi_goku http://recensubshq.forumfree.it/?t=64839203
    # Vapoursynth port by MonoS @github: https://github.com/MonoS/VS-MaskDetail
    def mask_detail(self, clip, final_width, final_height):
        resizer = core.fmtc.resample if self.approx else fvs.Resize
        startclip = core.fmtc.bitdepth(clip, bits=32)
        original = (startclip.width, startclip.height)
        target = (final_width, final_height)
        temp = resizer(startclip, *target[:2], kernel=self.kernel, invks=True, invkstaps=4, taps=self.taps, a1=self.b, a2=self.c)
        temp = resizer(temp, *original, kernel=self.kernel, taps=self.taps, a1=self.b, a2=self.c)
        mask = core.std.Expr([startclip, temp], 'x y - abs dup 0.015 > swap 16 * 0 ?').std.Inflate()
        mask = resizer(mask, *target, taps=self.taps)

        return core.fmtc.bitdepth(mask, bits=8, dmode=1)

    def save_images(self, src_luma32):
        resizer = core.fmtc.resample if self.approx else fvs.Resize
        src = src_luma32
        temp = imwri.Write(src.fmtc.bitdepth(bits=8), 'png', self.filename + '_source%d.png')
        temp.get_frame(0)  # trick vapoursynth into rendering the frame
        for r in self.resolutions:
            r += self.min_h
            image = self.mask_detail(src, self.getw(r), r)
            # TODO: use PIL for output
            t = imwri.Write(image.fmtc.bitdepth(bits=8), 'png', self.filename + f'_mask_{r:d}p%d.png')
            t.get_frame(0)
            t = resizer(src, self.getw(r), r, kernel=self.kernel, a1=self.b, a2=self.c, invks=True)
            t = imwri.Write(t.fmtc.bitdepth(bits=8), 'png', self.filename + f'_{r:d}p%d.png')
            t.get_frame(0)

    def get_filename(self):
        fn = ''.join([
            f"f_{self.frame}",
            f"_k_{self.kernel}",
            f"_b_{self.b:.2f}_c_{self.c:.2f}" if self.kernel == "bicubic" else "",
            f"_ar_{self.ar:.2f}" if self.ar else "",
            f"_taps_{self.taps}" if self.kernel == "lanczos" else "",
            f"_{self.min_h}-{self.max_h}",
            f"" if not self.approx else "_[approximation]",
        ])

        return fn


def to_float(str_value):
    if set(str_value) - set("0123456789./"):
        raise argparse.ArgumentTypeError("Invalid characters in float parameter")
    try:
        return eval(str_value) if "/" in str_value else float(str_value)
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError):
        raise argparse.ArgumentTypeError("Exception while parsing float") from None


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
parser.add_argument('--use-lsmash', '-ls', dest='ls', action="store_true", help='Use lsmash for input')
parser.add_argument('--is-image', '-img', dest='img', action="store_true", help='Force image input')
parser.add_argument('--generate-images', '-img-out', dest='img_out', action="store_true", help='Save detail mask as png')
parser.add_argument('--plot-scaling', '-ps', dest='plot_scaling', type=str.lower, default='log', help='Scaling of the y axis. Can be "linear" or "log"')
parser.add_argument('--plot-format', '-pf', dest='plot_format', type=str.lower, default='svg', help='Format of the output image. Can be svg, png, pdf, rgba, jp(e)g, tif(f), and probably more')


def getnative():
    starttime = time.time()
    args = parser.parse_args()

    if not args.approx and not hasattr(core, 'descale_getnative'):
        return print("Vapoursynth plugin descale_getnative not found.")

    if (args.img or args.img_out) and imwri is None:
        return print("Vapoursynth plugin imwri not found.")

    if args.img:
        src = imwri.Read(args.input_file)
        args.frame = 0
    elif args.ls:
        src = core.lsmas.LWLibavSource(args.input_file)
    else:
        src = core.ffms2.Source(args.input_file)

    if args.frame is None:
        args.frame = src.num_frames // 3

    if args.min_h >= src.height:
        return print(f"Picture is to small or equal for min height {args.min_h}.")
    elif args.min_h >= args.max_h:
        return print(f"Your min height is bigger or equal to max height.")
    elif args.max_h > src.height:
        print(f"Your max height cant be bigger than your image dimensions. New max height is {src.height}")
        args.max_h = src.height

    kwargs = args.__dict__.copy()
    del kwargs["input_file"]
    del kwargs["ls"]
    del kwargs["img"]

    get_native = GetNative(src, **kwargs)
    try:
        loop = asyncio.get_event_loop()
        forbidden_error, best_value = loop.run_until_complete(get_native.run())
    except BaseException as err:
        return print(f"Error in getnative: {err}")

    if not forbidden_error:
        content = ''.join([
            f"\nKernel: {args.kernel} ",
            f"B: {args.b:.2f} C: {args.c:.2f} " if args.kernel == "bicubic" else "",
            f"AR: {args.ar} " if args.ar else "",
            f"Taps: {args.taps} " if args.kernel == "lanczos" else "",
            f"\n{best_value}",
            f"" if not args.approx else "\n[approximation]",
        ])
        print(content)
        print('done in {:.2f} s'.format(time.time() - starttime))
    else:
        print(best_value)


if __name__ == '__main__':
    print("Start getnative.")
    getnative()
