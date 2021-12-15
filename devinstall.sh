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
<<<<<<< HEAD
git checkout $BRANCH
=======
>>>>>>> af9d99f7c39747e6409025cecc8fda6f30919731
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
