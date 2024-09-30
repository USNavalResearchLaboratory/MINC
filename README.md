# MINC
This repository is concerned with deep learning techniques to detect and localize damage on
a thin aluminum plate. Contained within are three neural network architectures,
each of which is designed to map acoustic sensing data to either a vector indicating the
location of damage or a binary quantity indicating the presence (or absence) of damage.
Of the three models, we found that the best performing architectures was
manifestly aware of the geometry of our sensor grid. Awareness of symmetry was achieved
through equivariant group convolutions.

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> MINC

It is authored by James Amarel in collaboration with Chris Rudolf, Athanasios Iliopoulos,
John Michopoulos, and Leslie N Smith. The paper is available on
[arXiv](https://www.arxiv.org/abs/2409.06084).

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "MINC"
```
which auto-activate the project and enable local path handling from DrWatson.

Some of the data needed to reproduce our experiments is versioned in `data/data_pro/`. However,
not all of the data was of small enough filesize to host on GitHub. Please contact us if you
are interested in the raw data, which should be stored in `data/exp_raw/`.

Tests for this package can be run using the following command:
```
pkg> test
```

Tools for running jobs and plotting results are contained in /scripts. To recreate our
experiments, one can run the following script:
```
julia> scripts/Jobs/jobs_all.jl
```
This runs the experiments defined in `/scripts/Jobs/Exps`. On an NVIDIA A100, training the
ordinary neural network required roughly 1 second per epoch for detection and 4 seconds per
epoch for localization. Training the symmetry-aware models required
roughly 4 seconds per epoch for detection and 25 seconds per epoch for localization.
Outcomes of these experiments together with preliminary visualizations of the results are
created and stored in runs/. Appropriately moving these results to `_research/runs_archive/`
(see the versioned file structure) then allows for further visualizations with the functions
contained in scripts/Plotting. To recreate the figures in our paper, run the `gfx_paper()` and
`gfx_supp()` functions defined in `scripts/Plotting/Plotting.jl`.
