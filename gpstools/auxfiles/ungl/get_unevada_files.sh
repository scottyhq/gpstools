#!/bin/bash
# these files from Unevada are updated regularly (-N overwrites if newer)
wget -N http://geodesy.unr.edu/NGLStationPages/decyr.txt
wget -N http://geodesy.unr.edu/velocities/midas.IGS08.txt
wget -N http://geodesy.unr.edu/velocities/midas.NA12.txt
wget -N http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt
wget -N http://geodesy.unr.edu/NGLStationPages/steps.txt
