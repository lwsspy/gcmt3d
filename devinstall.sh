#!/bin/bash

# Exit on error
set -e

if [ ! -d lwsspy-dev ]
then
    mkdir lwsspy-dev
fi
cd lwsspy-dev

# URLs
LWSSPY=https://github.com/lwsspy/lwsspy
LWSSPYSEISMO=https://github.com/lwsspy/lwsspy.seismo
LWSSPYGCMT3D=https://github.com/lwsspy/lwsspy.gcmt3d

# Clone all 3 repositories
git clone $LWSSPY
git clone $LWSSPYSEISMO
git clone $LWSSPYGCMT3D

# Create environment from lwsspy's file
cd lwsspy
conda env create -n lwsspy-dev -f environment.yml
conda activate lwsspy-dev
pip install -e .
cd ..

# Install seismic processing tools
cd lwsspy.seismo
pip install -e .
cd ..

# Install gcmt3d package
cd lwsspy.gcmt3d
pip install -e .
