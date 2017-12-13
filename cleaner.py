#! /usr/bin/env python3

import glob
import json
import subprocess
import os.path

files = glob.glob('/Users/torrance/CSIRO/observations/*/*/*/*/real_RMsynth.json')

for file in files:
    with open(file) as f:
        conf = json.load(f)
        snrPIfit = float(conf['snrPIfit'])
        if snrPIfit <= 10:
            continue

        path = os.path.dirname(file) + '/real.tsv'
        print(path)
        subprocess.check_call(['/Users/torrance/CSIRO/RM-tools/RMtools_1D/do_RMclean_1D.py', '-c', '-10', path])



