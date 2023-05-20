# PeakLoc
A framework for Single Molecule Localization Microscopy using an event-based camera.

## Installation
Clone the repository and create a conda environment with the dependencies.

`git clone https://github.com/androbaza/PeakLoc.git`

`conda env create -f environment.yml -n peakloc`

## Usage

PeakLoc.py is the main script. Input the path to the data. The script will create a folder with the same name as the file and save the localizations there. 

128Gb of RAM is recommended. For a full-schip (1280x720) recording of 600 seconds, the script will take about 10 minutes to run on a 24-core machine.
