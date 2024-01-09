import pandas as pd
import os

import numpy as np
import tqdm

import glob

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground, MeanBackground

from astropy.wcs import WCS
import astropy.stats as stats
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.table import Table, vstack
from astropy.io import fits

from astropy.stats import SigmaClip, sigma_clipped_stats

import aafitrans as aaf
import astroalign as aa

from astropy.visualization import simple_norm
import matplotlib.pyplot as plt

class Base():

  def detect_stars(self,data, th = 10):

    fwhm = self.fwhm
    sigma_clip = stats.SigmaClip(sigma=self.sigma)
    bkg_estimator = MedianBackground()

    bkg = Background2D( data,
                        self.box_size,
                        filter_size=self.filter_size,
                        sigma_clip=sigma_clip,
                        bkg_estimator=bkg_estimator)  #Background2D to detect the background of the image

    bkg_sub = data - bkg.background
    threshold = th*bkg.background_rms_median

    sigma = fwhm*stats.gaussian_fwhm_to_sigma  # FWHM = 20
    kernel = Gaussian2DKernel(x_stddev=sigma)

    convolved_data = convolve_fft(bkg_sub, kernel)

    daofind = DAOStarFinder(threshold, fwhm)

    sources = daofind(convolved_data)

    return sources, bkg, convolved_data

  def perform_photometry(self,sources, data, gain =1, RN=3, DC=2,
                         r = 3, r_in = 10, r_out = 15):

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=r)
    bags = CircularAnnulus(positions, r_in, r_out)

    ap_pix = apertures.area                                                                                             # calculating the aperture area
    bag_pix = bags.area                                                                                                 # calculating the annulus area

    phot_table = aperture_photometry(data, [apertures,bags])                                                            # calling the package Photutils_aperture_photometry

    phot_table['sky_flux'] = phot_table['aperture_sum_1'].value*(ap_pix/bag_pix)                                              # calculating sky flux by drawing an annulus

    phot_table['flux'] = phot_table['aperture_sum_0'].value - \
                        phot_table['sky_flux'].value                              # calculating source flux

    Noise_2 = gain*(phot_table['flux'].value  + phot_table['sky_flux'].value)+\
                     (DC + RN**2 + (gain/2)**2)*ap_pix

    phot_table['flux_err'] = np.sqrt(Noise_2)          # calculating error on the source flux

    phot_table['SNR'] = phot_table['flux']/phot_table['flux_err']                                                       # calculating signal to noise ratio

    return phot_table, (apertures,bags)
  
def update_dirs(out_dir):
    if not os.path.exists(f'{out_dir}/images'):
      os.mkdir(f'{out_dir}/images')

    else:
      os.system(f"rm {out_dir}/images/*")

    if not os.path.exists(f'{out_dir}/exposure_images'):
      os.mkdir(f'{out_dir}/exposure_images')

    else:
      os.system(f"rm {out_dir}exposure_images/*")

    if not os.path.exists(f'{out_dir}/animations'):
      os.mkdir(f'{out_dir}/animations')

    else:
      os.system(f"rm {out_dir}animations/*")

    if not os.path.exists(f'{out_dir}/SNR_table'):
      os.mkdir(f'{out_dir}/SNR_table')
    else:
      os.system(f"rm {out_dir}/SNR_table/*")

    if not os.path.exists(f'{out_dir}/cube'):
      os.mkdir(f'{out_dir}/cube')
    else:
      os.system(f"rm {out_dir}/cube/*")

    if not os.path.exists(f'{out_dir}/coordinates'):
      os.mkdir(f'{out_dir}/coordinates')
    else:
      os.system(f"rm {out_dir}/coordinates/*")

class tara(Base):

  def __init__(self, input_files=[], out_dir='.', box_size=64,
               sigma=3,fwhm=10, gain=68, filter_size=(3,3), crop_image=False,
               x_cen=None, y_cen = None,
               size=0, bin_image=False, bin_fact=1):

    self.box_size = box_size
    self.fwhm = fwhm
    self.gain = gain
    self.sigma = sigma
    self.filter_size = filter_size
    self.crop_image = crop_image
    self.bin_image = bin_image
    self.bin_fact = bin_fact
    self.time = []

    self.align_phot = False
    self.align_sources = []

    exps = [i for i in input_files if 'fits' in i.split('.')[-1]]

    if len(exps)<1:
      raise Exception("No '.fits' files in input list")

    self.exps = exps
    hdul = fits.open(exps[id])
    img = hdul[0].data

    self.shape = img.shape
    print("-------------------------------------------------")
    print(f"Input directory contains {len(exps)} '.fits' 'files")

    if crop_image:
        if x_cen is not None and y_cen is not None and size>0:
          img = img[x_cen - size: x_cen + size,
                    y_cen - size: y_cen + size].copy()
          print('Input image is Cropped')
          self.x_cen = x_cen
          self.y_cen = y_cen
          self.size = size
          self.shape = img.shape
        else:
          self.crop_image = False

    if bin_image:
      if img.shape[0] % bin_fact == 0 and img.shape[1] % bin_fact ==0 :
        x_bin = img.shape[0]//bin_fact
        y_bin = img.shape[0]//bin_fact

        img = img.reshape(x_bin, bin_fact,y_bin, bin_fact).sum(axis=(1,3))
        print("Input image is binned")

        self.x_bin = x_bin
        self.y_bin = y_bin
        self.bin_fact = bin_fact

      else:
        self.bin_image = False

    print(f"Image shape: {self.shape}")
    print("-------------------------------------------------")

    self.out_dir=out_dir

  def show_image(self, cmap='jet', norm='sqrt', fig=None, id=0,
                 check_photometry=True, th=1, r=10, r_in=None, r_out=None):

    hdul = fits.open(self.exps[id])
    img = hdul[0].data

    if fig is None:
      fig = plt.figure(figsize=(7,7))
      ax = fig.add_subplot()

    norm = simple_norm(img, norm, percent=99.)
    ax.imshow(img, cmap=cmap, norm=norm)

    if check_photometry:
      if r_in is None or r_out is None:
        r_in = r*1.2                             #Changed r_in/out from r*1.2/.5
        r_out = r*1.5

      sources, _, _ = self.detect_stars(img ,th=th)

      phot_table, apers = self.perform_photometry(sources, img,
                                                  gain=self.gain,
                                                  r=r, r_in=r_in, 
                                                  r_out=r_out)

      apers[0].plot(ax, color='blue')
      apers[1].plot(ax, color='black')

    return fig, ax, phot_table

  def gen_cube(self, th=10, start=0, end=-1, r=1, r_in = None, r_out = None,
               mar_pix=5, plot=False, ref_pos=None, mask_hot_pixel=False):


    if r_in is None:
      r_in = r*1.2
    if r_out is None:
      r_out = r*1.5

    poss = []
    imgs = []

    path = f'{self.out_dir}/images'

    if len(os.listdir(path))>0:
      os.system(f"rm {self.out_dir}/images/*")

    aa_flag = True
    for i,exp in enumerate(tqdm.tqdm(self.exps[int(start):int(end)], colour = 'GREEN')):

      hdul = fits.open(exp)
      img = hdul[0].data

      time = hdul[0].header['JD_UTC']
      hdul.close()

      if self.crop_image:
          img = img[self.x_cen - self.size:self.x_cen + self.size,
                    self.y_cen - self.size:self.y_cen + self.size].copy()

      if img.shape != self.shape:
        print(f"\nImage shape is {img.shape} not {self.shape}. Skipping")
        continue

      if mask_hot_pixel:
        mask = np.where(img>0.5*(2**16),0,1)
      else:
        mask = 1

      masked_img = mask*img

      if self.bin_image:
        img = img.reshape(self.x_bin, self.bin_fact,
                          self.y_bin, self.bin_fact).sum(axis = (1,3))


      try:
        sources, bkg, conv_data = self.detect_stars(img,th=th)

      except:
        print(f'Background detection failed for {exp}')
        continue

      if sources is None:
        print(f"No sources detected using DAOfind for {exp}")
        continue

      y_len, x_len = img.shape

      # Removing sources on top and bottom as those regions have high noise

      sources = sources[(sources['ycentroid']<y_len-mar_pix) &
                        (sources['ycentroid']>mar_pix)]

      sources = sources[(sources['xcentroid']<x_len-mar_pix) &
                        (sources['xcentroid']>mar_pix)]

      #Performing aperture photometry
      phot_table, apers = self.perform_photometry(sources, img, self.gain,
                                                  r=r, r_in=r_in, r_out=r_out)

      phot_table.sort('SNR', reverse=True)

      # reducing the sources to just 20 sources to speed the process

      phot_table = phot_table[:20]

      pos = np.transpose([phot_table['xcenter'],phot_table['ycenter']])

      poss.append(pos)

      if ref_pos is not None and aa_flag:

        if len(ref_pos)>5:
          print('\nUsing user reference catalog\n')
          aa_flag= False
          target_ref = ref_pos
          ref_img = img

      if aa_flag:

        source_sel = phot_table[phot_table['SNR']>1]

        if len(source_sel)>2:
          aa_flag = False
          target_ref = np.transpose([source_sel['xcenter'],source_sel['ycenter']])
          ref_img = img.copy()
          imgs.append(img)

        else:
          print("Astroalign requires atleast 3 sources in field")

      elif len(phot_table)>2:

        try:
          transf, (source_list, target_list) = aaf.find_transform(pos,target_ref,
                                                                  min_matches=3)

          registered_image = aa.apply_transform(transf, img, ref_img)
          self.time.append(time)

          if self.align_phot:
            sources_, _, _ = self.detect_stars(registered_image[0],
                                                        th=th)
            pos_ = np.transpose([sources_['xcentroid']
                                  ,sources_['ycentroid']])

            self.align_sources.append(pos_)

          img= registered_image[0]

          imgs.append(img)

        except:

          print(f"\nAA failed for {exp}")
      else:
        print("Target image has <3 sources")

      if plot:
        norm = simple_norm(img,'sqrt', percent=99.)
        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(img, norm=norm, cmap='jet')
        ax.axis(False)
        apers[0].plot(ax, color='blue')
        apers[1].plot(ax, color='black')

        ax.set_title('Image exposure')
        ax.annotate(f'{i}', (10,20), color = 'black',fontsize=10)
        fig.savefig(f'{self.out_dir}/images/{i}.png')
        plt.close(fig)


    poss = np.array(poss, dtype=object)  #making it an array to save as fits file

    return np.array(imgs), poss

  def __call__(self, th=1, rnge=[0,-1], step=1,
                r=1, r_in = None, r_out = None,
                th_cube=None, ref_pos=None, ref_img=None,
              mar_pix=5, mask_hot_pixel=False):
    
    update_dirs(self.out_dir)

    if r_in is None or r_out is None:
      r_in = r*1.2
      r_out = r*1.5

    for i in range(rnge[0], rnge[1]):
      print("------------------------------------")
      print("Iteration:", i)
      start = i*step
      end = (i+1)*step

      if i==1 and ref_pos is None:
        ref_pos = poss[0]

      cube, poss = self.gen_cube( th=th,
                                  start=start, end=end,
                                  r=r, r_in=r_in, r_out=r_out,
                                  mar_pix=mar_pix,
                                  ref_pos=ref_pos,
                                  mask_hot_pixel=mask_hot_pixel)

      fits.writeto(f'{self.out_dir}/cube/{i*step}_{(i+1)*step}cube.fits',
                    cube, overwrite=True)

      np.save(f'{self.out_dir}/coordinates/{i*step}_{(i+1)*step}x_y.npy',
                    poss)

      if i==0 and ref_img is None:
        ref_img = np.median(cube, axis=0)

      if th_cube is None:
        th_cube = 5*th

      sources, _, _ = self.detect_stars(ref_img ,th=th_cube)

      phot_table, _ = self.perform_photometry(sources, ref_img,
                                                gain=self.gain,
                                                r=r, r_in=r_in, r_out=r_out)

      flux = []
      sky_flux = []
      flux_err = []
      SNR = []

      for img in cube:

        phot_table, _ = self.perform_photometry(sources, img, gain=self.gain,
                                                  r=r, r_in=r_in, r_out=r_out)
        flux.append(phot_table['flux'].value)
        flux_err.append(phot_table['flux_err'].value)
        sky_flux.append(phot_table['sky_flux'].value)
        SNR.append(phot_table['SNR'].value)

      phot_table['flux'] =  np.array(flux).T
      phot_table['flux_err'] = np.array(flux_err).T
      phot_table['sky_flux'] = np.array(sky_flux).T
      phot_table['SNR'] = np.array(SNR).T


      out_file = f'{self.out_dir}/SNR_table/{i*step}_{(i+1)*step}Phot_table.fits'
      phot_table.write(out_file, overwrite=True)

    self.cube = cube
    self.poss = poss
    self.phot_table = phot_table
    return ref_pos, ref_img

  def merge_phot_table(self, out_dir='.'):

    tab_name = glob.glob(f'{self.out_dir}/SNR_table/*')
    tab_name = sorted(tab_name, key=lambda x: int(x.split('/')[-1].split('_')[0]))

    # Concatenate photometry of all cubes

    for i, f in enumerate(tqdm.tqdm(tab_name, colour = 'GREEN')):
      phot_table = Table.read(f)
      print(phot_table)
      phot_table['SNR_mean'] = phot_table['SNR'].mean(axis=1)
      phot_table.sort('SNR_mean', reverse=True)
      if i == 0 :
        flux = phot_table['flux'].value
        sky_flux = phot_table['sky_flux'].value
        flux_err = phot_table['flux_err'].value
        SNR = phot_table['SNR'].value

      else:
        flux = np.concatenate([flux, phot_table['flux'].value], axis = 1 )
        flux_err = np.concatenate([flux_err, phot_table['flux_err'].value], axis = 1 )
        sky_flux = np.concatenate([sky_flux, phot_table['sky_flux'].value], axis = 1 )
        SNR = np.concatenate([SNR, phot_table['SNR'].value], axis = 1 )


    phot_table['flux'] =  np.array(flux)
    phot_table['flux_err'] = np.array(flux_err)
    phot_table['sky_flux'] = np.array(sky_flux)
    phot_table['SNR'] = np.array(SNR)

    self.phot_table = phot_table
    return phot_table