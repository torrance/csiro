#! /usr/bin/env python3

import glob
import numpy as np
from pathlib import Path
import pandas as pd


# Parse the frequency data into what's required for rm-tools
# Freq (HZ); I (Jy); Q; U; V; I-err; Q-err; U-err; V-err

logs = glob.glob('observations/*/*/*/*/real.log')
for i, log in enumerate(logs):
    # if i > 0: break

    print("Processing {}".format(log))
    with open(log) as f:
        xs = []
        ys = []
        for line in f:
            line in line.strip()
            if line:
                freq, val = line.strip().split(' ', 1)
                xs.append(float(freq))
                ys.append(float(val))

    count = int(len(xs) / 4)
    xs = np.array(xs[:count]) * 1E9
    I = ys[:count]
    Q = np.array(ys[count:2*count])
    U = np.array(ys[2*count:3*count])
    V = np.array(ys[3*count:])

    df = pd.DataFrame(data=np.array([xs, I, Q, U]).T, columns=['freq', 'I', 'Q', 'U'])

    dI, dQ, dU = [], [], []
    for i, row in df.iterrows():
        # Find average and mean of 12Mhz either side
        freq = row['freq']
        low = freq - 12*10**6
        high = freq + 12*10**6
        window = df.query('@low <= freq <= @high')
        if len(window) < 8:
            error = 0.5
            dI.append(error)
            dQ.append(error)
            dU.append(error)
        else:
            I_std = window['I'].std()
            dI.append(I_std)

            Q_std = window['Q'].std()
            dQ.append(Q_std)

            U_std = window['U'].std()
            dU.append(U_std)

    df['dI'] = dI
    df['dQ'] = dQ
    df['dU'] = dU

    p = Path(log)
    p = Path(p.parent).joinpath(p.stem + '.tsv')
    print("Saving to {}".format(str(p)))
    df.to_csv(str(p), index=False, header=False, sep="\t")