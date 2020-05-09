# Getnative
Find the native resolution(s) of upscaled material (mostly anime)

# Usage
Install it via:

    $ pip install getnative
    
Start by executing:

    $ getnative [--args] inputFile

***or***  
  
Install all depdencies through `pip`

Start by executing:

    $ python run_getnative.py [--args] inputFile

That's it.

# Requirements

To run this script you will need:

* Python 3.6
* [matplotlib](http://matplotlib.org/users/installing.html)
* [Vapoursynth](http://www.vapoursynth.com) R45+
* [descale](https://github.com/Irrational-Encoding-Wizardry/vapoursynth-descale) (really slow for descale but needed for spline64 and lanczos5) 
and/or [descale_getnative](https://github.com/OrangeChannel/vapoursynth-descale) (perfect for getnative)
* [ffms2](https://github.com/FFMS/ffms2) or [lsmash](https://github.com/VFR-maniac/L-SMASH-Works) or [imwri](https://forum.doom9.org/showthread.php?t=170981)

# Example Output
Input Command:

    $ getnative -k bicubic -b 0.11 -c 0.51 -dir "../../Downloads" "../../Downloads/unknown.png"

Terminal Output:
```
Using imwri as source filter

500/500

Output Path: /Users/infi/Downloads/results

Bicubic b 0.11 c 0.51 AR: 1.78 Steps: 1
Native resolution(s) (best guess): 720p

done in 13.56s
```

Output Graph:

![alt text](https://nayu.moe/OSnWbP)

Output TXT (summary):
```
 715		 | 0.0000501392		 | 1.07
 716		 | 0.0000523991		 | 0.96
 717		 | 0.0000413640		 | 1.27
 718		 | 0.0000593276		 | 0.70
 719		 | 0.0000617733		 | 0.96
 720		 | 0.0000000342		 | 1805.60
 721		 | 0.0000599182		 | 0.00
 722		 | 0.0000554626		 | 1.08
 723		 | 0.0000413679		 | 1.34
 724		 | 0.0000448137		 | 0.92
 725		 | 0.0000455203		 | 0.98
```

# Args

| Property | Description | Default value | Type |
| -------- | ----------- | ------------------ | ---- |
| frame | Specify a frame for the analysis. | num_frames//3 | Int |
| kernel | Resize kernel to be used | bicubic | String |
| bicubic-b | B parameter of bicubic resize | 1/3 | Float |
| bicubic-c | C parameter of bicubic resize | 1/3 | Float |
| lanczos-taps | Taps parameter of lanczos resize | 3 | Int |
| aspect-ratio | Force aspect ratio. Only useful for anamorphic input| w/h | Float |
| min-height | Minimum height to consider | 500 | Int |
| max-height | Maximum height to consider | 1000 | Int |
| is-image | Force image input | False | Action |
| generate-images | Save detail mask as png | False | Action |
| plot-scaling | Scaling of the y axis. Can be "linear" or "log" | log | String |
| plot-format | Format of the output image. Specify multiple formats separated by commas. Can be svg, png, tif(f), and more | svg | String |
| show-plot-gui | Show an interactive plot gui window. | False | Action |
| no-save | Do not save files to disk. | False | Action |
| stepping | This changes the way getnative will handle resolutions. Example steps=3 [500p, 503p, 506p ...] | 1 | Int |
| output-dir | Sets the path of the output dir where you want all results to be saved. (/results will always be added as last folder) | (CWD)/results | String |

# CLI Args

| Property | Description | Default value | Type |
| -------- | ----------- | ------------------ | ---- |
| help | Automatically render the usage information when running `-h` or `--help` | False | Action |
|  | Absolute or relative path to the input file | Required | String |
| mode | Choose a predefined mode \["bilinear", "bicubic", "bl-bc", "all"\] | None | String |
| use | Use specified source filter (e.g. "lsmas.LWLibavSource") | None | String |

# Warning
This script's success rate is far from perfect.
If possible, do multiple tests on different frames from the same source.
Bright scenes generally yield the most accurate results.
Graphs tend to have multiple notches, so the script's assumed resolution may be incorrect.
Also, due to the current implementation of the autoguess, it is not possible for the script 
to automatically recognize 1080p productions.
Use your eyes or anibin if necessary.
  
# Thanks  
BluBb_mADe, kageru, FichteFoll, stux!, LittlePox

# Help?

Join https://discord.gg/V5vaWwr (Ask in #encode-autism for help)
