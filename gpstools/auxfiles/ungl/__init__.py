"""
Auxillary files from each processing center
"""
import os
from pathlib import Path

auxdir = os.path.join(Path(__file__).parent.parent, "ungl")

def update():
    ''' run script to download auxillary files from server'''
    pwd = os.getcwd()
    os.chdir(auxdir)
    print(f'downloading aux files in {auxdir}')
    os.system('./get_unevada_files.sh')
    os.chdir(pwd)
