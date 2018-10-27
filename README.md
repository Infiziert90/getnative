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
* [Vapoursynth](http://www.vapoursynth.com) R39+
* [descale](https://github.com/Irrational-Encoding-Wizardry/vapoursynth-descale) or [descale_getnative](https://github.com/BluBb-mADe/vapoursynth-descale)
* [ffms2](https://github.com/FFMS/ffms2) or [lsmash](https://github.com/VFR-maniac/L-SMASH-Works) or [imwri](https://forum.doom9.org/showthread.php?t=170981)

# Example Output
Input Command:

    $ python getnative.py /home/infi/mpv-shot0001.png -k bicubic -b 1/3 -c 1/3

Output Text:
```
Using imwri as source filter
501/501
Kernel: bicubic AR: 1.78 B: 0.33 C: 0.33
Native resolution(s) (best guess): 720p, 987p
done in 29.07s
```

Output Graph:

![alt text](https://nayu.moe/UavJvs)

# Args

| Property | Description | Default value | Type |
| -------- | ----------- | ------------------ | ---- |
| help | Automatically render the usage information when running `-h` or `--help` | False | Action |
|  | Absolute or relative path to the input file | Required | String |
| frame | Specify a frame for the analysis. | num_frames//3 | Int |
| scaler | Use a predefined scaler. | Bicubic (b=1/3, c=1/3) | String |
| kernel | Resize kernel to be used | bilinear | String |
| bicubic-b | B parameter of bicubic resize | 1/3 | Float |
| bicubic-c | C parameter of bicubic resize | 1/3 | Float |
| lanczos-taps | Taps parameter of lanczos resize | 3 | Int |
| aspect-ratio | Force aspect ratio. Only useful for anamorphic input| w/h | Float |
| min-heigth | Minimum height to consider | 500 | Int |
| max-heigth | Maximum height to consider | 1000 | Int |
| use | Use specified source filter (e.g. "lsmas.LWLibavSource") | None | String |
| is-image | Force image input | False | Action |
| generate-images | Save detail mask as png | False | Action |
| plot-scaling | Scaling of the y axis. Can be "linear" or "log" | log | String |
| plot-format | Format of the output image. Can be svg, png, tif(f), and more | svg | String |
| show-plot-gui | Show an interactive plot gui window. | False | Action |
| bilinear | Run all predefined bilinear scalers. | False | Action |
| bicubic | Run all predefined bicubic scalers. | False | Action |
| bc-bl | Run all predefined bicubic and bilinear scalers. | False | Action |
| run-all | Run all scalers. | False | Action |
| no-save | Do not save files to disk. | False | Action |


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
