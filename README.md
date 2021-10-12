|__Documentation__| __https://lwsspy.github.io/lwsspy.gcmt3d__|
|-|-|
|__Deployment__  | __[![PyPI version](https://badge.fury.io/py/lwsspy.gcmt3d.svg)](https://badge.fury.io/py/lwsspy.gcmt3d)__|
|__Build Status__| __[![Build Status](https://travis-ci.com/lwsspy/lwsspy.gcmt3d.svg?branch=main)](https://travis-ci.com/lwsspy/lwsspy.gcmt3d)__|
|__License__     |__[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)__|



# GCMT3D 

This package is created to handle Global 3D Centroid Moment Tensor Inversion
using the state-of-the-art global adjoint tomography Earth model [GLAD-M25].

Some of the challenges to create this workflow are the computational cost of
global 3D earthquake simulations as well as handling High Performance Computing
(HPC) infrastructure. For a single earthquake this software may be overkill.
The goal however was to recompute a large part of the [GCMT] catalog, which
required an automated workflow


## Simplest Installation

```bash
pip install lwsspy.gcmt3d
```

## Installation for development purposes

If you want to further develop this package, I recommend creating an environment
that also installs all dependencies in editable mode. The dependencies are
tightly knit with the `gcmt3d` package and chances are you will have to edit 
their source code anyways. To make that process seemless, I created a script
that performs the installation from scratch [devinstall.sh].

> :warning: **You must have Anaconda already installed for this to work**!

First navigate to where you want install everything then
```bash
curl -Lks https://raw.github.com/lwsspy/lwsspy.gcmt3d/main/devinstall.sh | /bin/bash -i
```

## Run Tests

```bash
cd <repo>
pytest tests
```

For some reason the `pytest` command fails due to relative import issues when
run without arguments, so please add the directory `tests` as a directory input.

It's sort of puzzling because I use `pytest` without arguments for 
`lwsspy.seismo` and have `0` issues.


[devinstall.sh]: devinstall.sh
[GCMT]: <https://www.globalcmt.org>
[ADIOS]: <https://adios2.readthedocs.io/en/latest/>
[SPECFEM3D_GLOBE]: <https://geodynamics.org/cig/software/specfem3d_globe/> 
[GLAD-M25]: <https://academic.oup.com/gji/article/223/1/1/5841525>
[GLAD-M25-Wiki]: <https://github.com/computational-seismology/GLAD-M25/wiki>
[gcc]: <https://gcc.gnu.org/install/>
[openmpi]: <https://www.open-mpi.org>
[Par_file]: <https://github.com/geodynamics/specfem3d_globe/blob/devel/DATA/Par_file>
[SpecfemMagic]: <https://github.com/lsawade/SpecfemMagic>