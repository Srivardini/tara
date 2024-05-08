import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator,AutoLocator, AutoMinorLocator)
# To improve Aesthetics of plots
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10,10)
matplotlib.rcParams['font.size'] = 15

import matplotlib.animation as animation

def gen_plot(data, key='SNR', t=None, out_dir='.',hdul = None, fmt='o',
             camera_name = None,id = None, fig=None, ax =None, figsize=(15,5),
             bin_fact = 1, diff_phot=False, norm=False, lw=2, slice_end=None,
            mask_min=200):

    if fig is None or ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
    if t is None:
        t = np.arange(0,len(data[key][0]), 1)[:slice_end]

    if len(data[key][0][0:slice_end])%bin_fact ==0:
      t = t.reshape(-1, bin_fact).mean(axis=1)
      flag = True
      ref = 0
      for row in data:
          x = np.array(row[key])[:slice_end]
          x = x.reshape(-1, bin_fact).mean(axis=1)

          #Adding mask to plot:
          mask = abs(x[1:]-x[:-1])<mask_min
          mask = np.array([True] + list(mask))
          mask = np.where(~mask,np.nan, 1)
          x = x*mask

          if diff_phot:
            if flag:
              source = x
              flag = False
            else:
              ref += x
          else:
            if norm:
              x = x/x.mean()
            ax.plot(t,x, fmt, alpha=1,lw=lw)
    else:
       raise Exception("Binning Failed. Try changing bin_fact")
    if diff_phot:
      # source = source/source.mean()
      # ref =  ref/ref.mean()
      ax.plot(t,source/ref, fmt, alpha=1, lw=lw)

      # Assuming standard deviation as error for demonstration
      y_err = np.std(source/ref)
      ax.errorbar(t, source/ref, yerr=y_err, fmt=fmt, alpha=1)

    #ax.set_xlabel('Frames')
    ax.set_ylabel(f'{key}')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which='both', width=2,direction="in", top = True,right = True,
                bottom = True, left = True)
    ax.tick_params(which='major', length=7,direction="in")
    ax.tick_params(which='minor', length=4, color='black',direction="in")
    #ax.set_title(f'{key} vs Frames - {id}')
    if hdul is not None:
        text = f"""Obervation Date : {hdul[0].header['DATE-OBS']} \nCamera name     : {camera_name}
        Exposure Time    : {hdul[0].header['EXPTIME']}"""      #\nGain                    : {hdul[0].header['GAIN']}

        ax.text(-3,1.02*np.max(x), text, fontsize=15,
                bbox=dict(facecolor='red', alpha=0.2))
    fig.savefig(f'{out_dir}/exposure_images/{id}.png')
    return fig, ax

def clean_outdir(out_dir='.'):
    if os.path.exists(f'{out_dir}/images'):
        os.system(f"rm  -r {out_dir}/images")

    if os.path.exists(f'{out_dir}/exposure_images'):
        os.system(f"rm -r {out_dir}/exposure_images/")

    if os.path.exists(f'{out_dir}/animations'):
        os.system(f"rm -r {out_dir}/animations")

    if not os.path.exists(f'{out_dir}/SNR_table'):
        os.system(f"rm -r {out_dir}/SNR_table")

    if not os.path.exists(f'{out_dir}/cube'):
        os.system(f"rm -r {out_dir}/cube")

    if not os.path.exists(f'{out_dir}/coordinates'):
        os.system(f"rm -r {out_dir}/coordinates")
