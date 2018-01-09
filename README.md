# Getnative
Find the native resolution(s) of upscaled material (mostly anime)

# Usage
Start with installing the depdencies through `pip`.

Start by executing:

    $ python getnative.py inputFile [--args]

That's it.

# Requirements

To run this script you will need:

* Python 3.6
* [matplotlib](http://matplotlib.org/users/installing.html)
* [Vapoursynth](http://www.vapoursynth.com)
* [descale](https://github.com/BluBb-mADe/vapoursynth-descale) or [fmtconv](https://github.com/EleonoreMizo/fmtconv)
* [ffms2](https://github.com/FFMS/ffms2) or [lsmash](https://github.com/VFR-maniac/L-SMASH-Works) or [imwri](https://forum.doom9.org/showthread.php?t=170981)

# Example Output
Input Command:

    $ python getnative.py "/input/path/00000.m2ts" -ls -k bicubic -pf png

Output Text:
```
Start getnative.
Kernel: bicubic B: 0.33 C: 0.33 
Native resolution(s) (best guess): 720p
done in 19.51 s
```

Output Graph:

![alt text](https://0x0.st/FaK.png)

# Args

| Property | Description | Default value | Type |
| -------- | ----------- | ------------------ | ---- |
| help | Automatically render the usage information when running `-h` or `--help` | true | Boolean |
|  | Absolute or relative path to the input file | Required | String |
| frame | Specify a frame for the analysis. | num_frames//3 | Int |
| kernel | Resize kernel to be used | bilinear | String |
| bicubic-b | B parameter of bicubic resize | 1/3 | Float |
| bicubic-c | C parameter of bicubic resize | 1/3 | Float |
| lanczos-taps | Taps parameter of lanczos resize | 3 | Int |
| aspect-ratio | Force aspect ratio. Only useful for anamorphic input| w/h | Float |
| approx | Use fmtc instead of descale (faster, loss of accuracy) | False | Bool |
| min-heigth | Minimum height to consider | 500 | Int |
| max-heigth | Maximum height to consider | 1000 | Int |
| use | Use specified source filter (e.g. "lsmas.LWLibavSource") | None | String |
| is-image | Force image input | False | Bool |
| generate-images | Save detail mask as png | False | Bool |
| plot-scaling | Scaling of the y axis. Can be "linear" or "log" | log | String |
| plot-format | Format of the output image. Can be svg, png, tif(f), and more | svg | String |
| show-plot-gui | Show an interactive plot gui window. | False | Bool |


# Warning
This script's success rate is far from perfect.
If possible, do multiple tests on different frames from the same source.
Bright scenes generally yield the most accurate results.
Graphs tend to have multiple notches, so the script's assumed resolution may be incorrect.
Also, due to the current implementation of the autoguess, it is not possible for the script 
to automatically recognize 1080p productions.
Use your eyes or anibin if necessary.
  
# Thanks  
BluBb_mADe, kageru, FichteFoll, stux!

# Help?

Join https://discord.gg/V5vaWwr (Ask in #encode-autism for help)
