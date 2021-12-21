Installation
------------

If you have all dependencies a simple

.. code:: bash
    
    pip install lwsspy.gcmt3d

Should do the trick.

While installation on a Mac and or Windows machine is usually simple,
installing the software on clusters has been chronically horrible. As part of
the source code, I provide a script called ``devinstall.sh``. This script
should do three things to set you up.

1. Get all the required repos (`lwsspy`_, `lwsspy.seismo`, `lwsspy.gcmt3d`)
2. Creates a `conda` environment
3. Installs all needed packages

If you need help with a specific installation and/or encounter any problems
don't hesitate to contact me at lsawade@princeton.edu .


`devinstall.sh`
+++++++++++++++

.. code:: bash

    #!/bin/bash

    # Exit on error
    set -e

    if [ ! -d lwsspy ]
    then
        mkdir lwsspy
    fi
    cd lwsspy

    # URLs
    BRANCH=dev
    LWSSPY=git@github.com:lwsspy/lwsspy.git
    LWSSPYSEISMO=git@github.com:lwsspy/lwsspy.seismo.git
    LWSSPYGCMT3D=git@github.com:lwsspy/lwsspy.gcmt3d.git

    # Clone all 3 repositories
    git clone $LWSSPY
    git clone $LWSSPYSEISMO
    git clone $LWSSPYGCMT3D

    # Create environment from lwsspy's file
    cd lwsspy
    git checkout $BRANCH
    conda env create -n lwsspy -f environment.yml
    conda activate lwsspy
    pip install -e .
    cd ..

    # Install seismic processing tools
    cd lwsspy.seismo
    git checkout $BRANCH
    pip install -e .
    cd ..

    # Install gcmt3d package
    cd lwsspy.gcmt3d
    git checkout $BRANCH
    pip install -e .



.. _lwsspy: `https://lwsspy.github.io/lwsspy/`
.. _lwsspy.seismo: `https://lwsspy.github.io/lwsspy.seismo/`
.. _lwsspy.gcmt3d: `https://lwsspy.github.io/lwsspy.gcmt3d/`
.. _lwsspy.gcmt3d-DOI: `https://doi.org/10.34770/yctp-3c03`
.. _catalog-DOI: `https://doi.org/10.34770/gp8e-sx34` 

