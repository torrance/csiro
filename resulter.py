import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pickle
import glob
import os
import os.path
from pathlib import Path
import json
from io import BytesIO
import base64


def main():
    datadict = load_datadict()
    moments2, polarised, unpolarised = categorise_sources(datadict)

    html = open('results.html', 'w')
    print("""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset=utf-8 />
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>
    </head>
    <body>
    """, file=html)

    # Plot histogram of clean components 2nd moment
    plt.figure('moment2', figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)
    plt.hist(moments2, 50)
    plt.xlabel('Second moment of clean component')
    plt.ylabel('Frequency')
    ax.set_yscale('log', basex=10)
    plt.title('Second moment of clean components')
    plt.grid(True)
    iobuffer = BytesIO()
    plt.savefig(iobuffer, format='png')
    plt.close()
    b64data = base64.b64encode(iobuffer.getvalue()).decode('utf8')
    print('<center><img width=600px src="data:image/png;base64,' + b64data + '" /></center>', file=html)

    print("""
    <table class="table table-bordered">
    <tr><th>Polarised</th><th>Unpolarised</th></tr><tr>
    """, file=html)
    print('<td>' + str(polarised) + '</td>', file=html)
    print('<td>' + str(unpolarised) + '</td>', file=html)
    print('</tr></table>')

    print("""
    <table class="table table-bordered">
    <tr>
    <th>Source name</th>
    <th>Epoch</th>
    <th>Plots</th>
    <th>Info</th>
    </tr>
    """, file=html)

    for i, (source_name, dates) in enumerate(datadict.items()):
        # if i > 50: break
        if source_name not in ['1729-37', '1936-155', '0648-165', 'j1548-6401', 'j0116-7905', '1129-58', '1143-245', 'j1237-7235', 'j2358-4555']: continue

        print('<tr>', file=html)
        print('<td rowspan="' + str(dates['count']) + '"><h3>' + source_name + '</h3>', file=html)
        if dates['info']['polarised']:
            print('<a class="btn btn-success" role="button">Polarised</a><br /><br />', file=html)
        if dates['info']['changed']:
            print('<a class="btn btn-warning" role="button">Changed</a><br />', file=html)
        print('</td>', file=html)

        print("Plotting: " + source_name)

        j = -1
        for date, bands in dates.items():
            if date == 'count' or date == 'info':
                continue

            j += 1

            # If not first date, create new date row
            if j != 0:
                print('<tr>', file=html)

            print('<td rowspan=' + str(bands['count']) + '>' + str(date) + '</td>', file=html)

            k = -1
            for band, observations in bands.items():
                if band == 'count': continue

                k += 1

                # If not first band, create new band row
                if k != 0:
                    print('<tr>', file=html)

                for obs in observations:
                    plt.figure(figsize=(24, 6))

                    I_fit = obs['I_fit']
                    I_bandaveraged = obs['I_bandaveraged']

                    # Plot the raw REAL Q, U values
                    lambda2 = (2.998E8 / obs['real.tsv'].freq)**2
                    U_avg = moving_average(obs['real.tsv'].U/I_fit, 31)
                    Q_avg = moving_average(obs['real.tsv'].Q/I_fit, 31)
                    ylim = max(U_avg.abs().max(), Q_avg.abs().max()) * 1.2

                    ax = plt.subplot(1, 4, 1)
                    ax.scatter(lambda2, obs['real.tsv'].Q/I_fit, color=(63/255, 127/255, 191/255, 0.2), s=7, edgecolor='')
                    ax.plot(lambda2, Q_avg, color=(63/255, 127/255, 191/255))
                    ax.scatter(lambda2, obs['real.tsv'].U/I_fit, color=(191/255, 63/255, 63/255, 0.2), s=7, edgecolor='')
                    ax.plot(lambda2, U_avg, color=(191/255, 63/255, 63/255))
                    ax.set_xlabel("$\lambda^2$")
                    ax.set_ylabel("Fractional polarisation")
                    ax.set_ylim(-ylim, ylim)
                    ax.set_title("Real")
                    ax.grid(True)

                    # Plot the raw IMAGINARY Q, U values
                    U_avg = moving_average(obs['imag.tsv'].U/I_fit, 31)
                    Q_avg = moving_average(obs['imag.tsv'].Q/I_fit, 31)

                    ax = plt.subplot(1, 4, 2)
                    ax.scatter(lambda2, obs['imag.tsv'].Q/I_fit, color=(63/255, 127/255, 191/255, 0.2), s=7, edgecolor='')
                    ax.plot(lambda2, Q_avg, color=(63/255, 127/255, 191/255))
                    ax.scatter(lambda2, obs['imag.tsv'].U/I_fit, color=(191/255, 63/255, 63/255, 0.2), s=7, edgecolor='')
                    ax.plot(lambda2, U_avg, color=(191/255, 63/255, 63/255))
                    ax.set_xlabel("$\lambda^2$")
                    ax.set_ylabel("Fractional polarisation")
                    ax.set_ylim(-ylim, ylim)
                    ax.set_title("Imaginary")
                    ax.grid(True)

                    # Plot the dirty FDF
                    freq = obs['FDFdirty.dat'][0]
                    reals = obs['FDFdirty.dat'][1]/I_bandaveraged * 100
                    imags = obs['FDFdirty.dat'][2]/I_bandaveraged * 100
                    amp = np.sqrt(reals**2 + imags**2)

                    ax = plt.subplot(1, 4, 3)
                    ax.plot(freq, reals, color=(63/255, 127/255, 191/255))
                    ax.plot(freq, imags, color=(191/255, 63/255, 63/255))
                    ax.plot(freq, amp, color=(0.2, 0.2, 0.2))
                    ax.set_xlim(-1300, 1300)
                    ax.set_xlabel("Rotation measure")
                    ax.set_ylabel("Fractional polarisation %")
                    ax.grid(True)
                    ax.set_title("Dirty")

                    ax = plt.subplot(1, 4, 4)
                    if 'FDFclean.dat' in obs:
                        freq = obs['FDFclean.dat'][0]
                        reals = obs['FDFclean.dat'][1]/I_bandaveraged * 100
                        imags = obs['FDFclean.dat'][2]/I_bandaveraged * 100
                        amp = np.sqrt(reals**2 + imags**2)
                        cleanCutoff = obs['RMclean.json']['cleanCutoff'] / I_bandaveraged

                        ax.plot(freq, reals, color=(63/255, 127/255, 191/255))
                        ax.plot(freq, imags, color=(191/255, 63/255, 63/255))
                        ax.plot(freq, amp, color=(0.2, 0.2, 0.2))
                        ax.plot([-2000, 2000], [cleanCutoff, cleanCutoff], color='green', linestyle='--')
                        ax.set_xlabel("Rotation measure")
                        ax.set_ylabel("Fractional polarisation %")
                        ax.set_xlim(-1300, 1300)
                        ax.grid(True)
                        ax.set_title("Clean")

                    plt.suptitle(source_name + ' - ' + str(date), fontsize=16)
                    plt.tight_layout()
                    plt.subplots_adjust(hspace=0.5)

                    # Save plots to html
                    iobuffer = BytesIO()
                    plt.savefig(iobuffer, format='png')
                    b64data = base64.b64encode(iobuffer.getvalue()).decode('utf8')
                    print('<td><img width=850px src="data:image/png;base64,' + b64data + '" /></td>', file=html)

                    plt.close()

                    print('<td>', file=html)
                    if 'RMclean.json' in obs:
                        print(obs['RMclean.json'])
                        # print(obs['RMsynth.json'])
                        print('polAngle0Fit_deg: ', obs['RMclean.json']['polAngle0Fit_deg'], end='<br />', file=html)
                        print('polAngleFit_deg: ', obs['RMclean.json']['polAngleFit_deg'], end='<br />', file=html)
                        print('ampPeakPIfit_Jybm: ', obs['RMclean.json']['ampPeakPIfit_Jybm'], end='<br />', file=html)
                        print('phiPeakPIfit_rm2: ', obs['RMclean.json']['phiPeakPIfit_rm2'], end='<br />', file=html)
                        print('fracPol%: ', obs['RMclean.json']['ampPeakPIfit_Jybm'] / I_bandaveraged * 100, end='<br />', file=html)
                        print('fracPol% (theirs): ', obs['RMsynth.json']['fracPol'] * 100, end='<br />', file=html)
                        print('mom2CCFDF: ', obs['RMclean.json']['mom2CCFDF'], end='<br />', file=html)

                    print('</td>', file=html)
                    print('</tr>', file=html)

    # Close the html
    print("""
    </body>
    </html>
    """, file=html)
    html.close()


def moving_average(data, width):
    offset = int((width - 1) / 2)
    size = len(data)

    avgs = data.copy()

    for i in range(size):
        xmin = max(0, i - offset)
        xmax = min(size, i + offset)
        avgs[i] = np.mean(data[xmin:xmax])

    return avgs


def load_datadict():
    # Either load or recompute datadict object
    if os.path.isfile('/Users/torrance/CSIRO/datadict.pickled'):
        with open('/Users/torrance/CSIRO/datadict.pickled', 'rb') as f:
            datadict = pickle.load(f)
    else:
        files = glob.glob('/Users/torrance/CSIRO/observations/*/*/*/*/real.tsv')

        datadict = {}

        for file in files:
            p = Path(file)
            source_name = p.parts[5]
            band = p.parts[7]

            year, month, day = p.parts[6].split('-')
            year, month, day = int(year), int(month), int(day)

            # Skip bad dates
            if (
                (year, month, day) == (2014, 2, 2) or
                (year, month, day) == (2011, 2, 21) or
                (year, month, day) == (2016, 3, 18)
            ):
                continue
            date = datetime.date(year, month, day)

            parent = p.parent

            if source_name not in datadict:
                datadict[source_name] = {'count': 1}  # Dates
            else:
                datadict[source_name]['count'] += 1

            if date not in datadict[source_name]:
                datadict[source_name][date] = {'count': 1}  # Bands
            else:
                datadict[source_name][date]['count'] += 1

            if band not in datadict[source_name][date]:
                datadict[source_name][date][band] = []  # List of IDs

            leaf = {}
            pp = Path.joinpath(parent, 'real.tsv')
            if pp.exists():
                leaf['real.tsv'] = pd.read_csv(str(pp), sep='\t', names=['freq', 'I', 'Q', 'U', 'dI', 'dQ', 'dU'])
            pp = Path.joinpath(parent, 'imag.tsv')
            if pp.exists():
                leaf['imag.tsv'] = pd.read_csv(str(pp), sep='\t', names=['freq', 'I', 'Q', 'U', 'dI', 'dQ', 'dU'])
            pp = Path.joinpath(parent, 'real_RMclean.json')
            if pp.exists():
                leaf['RMclean.json'] = json.load(pp.open())
            pp = Path.joinpath(parent, 'real_RMsynth.json')
            if pp.exists():
                leaf['RMsynth.json'] = json.load(pp.open())
            pp = Path.joinpath(parent, 'real_FDFdirty.dat')
            if pp.exists():
                leaf['FDFdirty.dat'] = np.loadtxt(str(pp), unpack=True)
            pp = Path.joinpath(parent, 'real_FDFclean.dat')
            if pp.exists():
                leaf['FDFclean.dat'] = np.loadtxt(str(pp), unpack=True)

            # Calculate total intensity (merging real and imaginary)
            leaf['I_total'] = np.sqrt(leaf['real.tsv'].I**2 + leaf['imag.tsv'].I**2)

            # Fit I_total with polynomial
            fit = np.polyfit(leaf['real.tsv'].freq, leaf['I_total'], 10)
            leaf['I_fit'] = np.array([
                    fit[0] * x**10 + fit[1] * x**9 +
                    fit[2] * x**8 + fit[3] * x**7 +
                    fit[4] * x**6 + fit[5] * x**5 +
                    fit[6] * x**4 + fit[7] * x**3 +
                    fit[8] * x**2 + fit[9] * x**1 +
                    fit[10] for x in leaf['real.tsv'].freq])

            # Calculate bandaveraged I
            lam0 = np.sqrt(leaf['RMsynth.json']['lam0Sq_m2'])
            x = 2.998E8 / lam0
            leaf['I_bandaveraged'] = (
                fit[0] * x**10 + fit[1] * x**9 +
                fit[2] * x**8 + fit[3] * x**7 +
                fit[4] * x**6 + fit[5] * x**5 +
                fit[6] * x**4 + fit[7] * x**3 +
                fit[8] * x**2 + fit[9] * x**1 +
                fit[10]
            )

            datadict[source_name][date][band].append(leaf)

        with open('/Users/torrance/CSIRO/datadict.pickled', 'wb') as f:
            pickle.dump(datadict, f)

    return datadict


def categorise_sources(datadict):
    # Let's go through and categorise our sources
    moments2 = []  # collect all clean 2nd moments
    unpolarised = 0
    polarised = 0
    for source_name, source in datadict.items():
        info = {'polarised': False, 'changed': False}
        fracpols = []
        phis = []

        for date, bands in source.items():
            if date == 'count' or date == 'info': continue

            for band, observations in bands.items():
                if band == 'count': continue

                for obs in observations:
                    # Add the 2nd moment
                    if 'RMclean.json' in obs:
                        if obs['RMsynth.json']['fracPol'] > 0.005:
                            moments2.append(obs['RMclean.json']['mom2CCFDF'])
                            info['polarised'] = True
                            fracpols.append(obs['RMsynth.json']['fracPol'])
                            phis.append(obs['RMclean.json']['phiPeakPIfit_rm2'])

        if info['polarised']:
            polarised += 1
        else:
            unpolarised += 1

        # Detect change
        for x in fracpols:
            for y in fracpols:
                if abs(x - y) / x > 0.2:
                    # print(x, y)
                    info['changed'] = True
        for x in phis:
            for y in phis:
                try:
                    if abs(x - y) > 15:
                        # print(x, y)
                        info['changed'] = True
                except TypeError:
                    pass

        datadict[source_name]['info'] = info

    return moments2, polarised, unpolarised


if __name__ in '__main__':
    main()