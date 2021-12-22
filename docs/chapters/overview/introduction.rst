.. _sec.introduction:

Introduction
------------

`lwsspy.gcmt3d`_ is a python package that automates the
inversion of global centroid moment tensors given a starting solution. The
package is particularly powerful for global centroid moment tensor inversions
using three-dimensional Green's functions because it includes ready-to-go
scripts usable on HPC clusters (mostly tested for LSF,
but technically agnostic). For background and theory of the software, please
refer to the accompanying publication from Sawade et al. (2022). If you end up
using the software, consider citing the package using its
`lwsspy.gcmt3d-DOI`_.

The package is fairly small as it relies on other packages of the
`lwsspy`_ . The strongest dependency is
`lwsspy.seismo`_  because it controls processing and windowing
of the waveforms used in the inversion. `lwsspy.gcmt3d`_
mostly just wraps functions written as part of my (Lucas Sawade) daily Python 
drivers.






.. _lwsspy: `https://lwsspy.github.io/lwsspy/`
.. _lwsspy.seismo: `https://lwsspy.github.io/lwsspy.seismo/`
.. _lwsspy.gcmt3d: `https://lwsspy.github.io/lwsspy.gcmt3d/`
.. _lwsspy.gcmt3d-DOI: `https://doi.org/10.34770/yctp-3c03`
.. _catalog-DOI: `https://doi.org/10.34770/gp8e-sx34` 

