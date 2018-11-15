#!/bin/bash

#Initialise Large File System(LFS)
git lfs install;

#Clone the repository
git lfs clone https://github.com/Jolig/Spraak1.git;

cd Spraak1;

#Run the test file
python test.py;