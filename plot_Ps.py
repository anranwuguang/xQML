import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import timeit

import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var
from xqml.simulation import extrapolpixwin


def plot_Ps(output_dir, clth_in, spec,nside,lmax,dell,fsky):
    # # ############## Initial parameters ###############
    if fsky < 0.3:
        lmin = int((3+dell)/2)+2*dell
    elif fsky >= 0.3:
        lmin = 2
    ellval = np.arange(lmin, lmax-dell//2, dell)
    lth = np.arange(np.min(ellval), np.max(ellval)+1)
    factor1 = lth*(lth+1)/2./np.pi
    factor2 = ellval*(ellval+1)/2./np.pi
    print(ellval)
    # # ################################################# 
    ##################  Input model  #####################
    hcla  = np.loadtxt(output_dir+"/results/hcla.txt", unpack=True)
    scla  = np.loadtxt(output_dir+"/results/scla.txt", unpack=True)

    clth = clth_in[:,lth]
    clth_bin = clth_in[:,ellval]

    if len(hcla.shape) == 1:
        hcla = np.array([hcla])
        scla = np.array([scla]) 
    if fsky>0.3:
          hcla = hcla[:,:len(ellval)]
          scla = scla[:,:len(ellval)]   
    elif fsky<0.3:
          hcla = hcla[:,2:len(ellval)+2]
          scla = scla[:,2:len(ellval)+2]    
    nspec = hcla.shape[0]          
    ##############################
    stokes, spec, istokes, ispecs = getstokes( spec=spec)
    print(stokes, spec, istokes, ispecs)
    label=['TT','EE','BB','TE','TB','EB']
    cspecs = [
          'yellowgreen',
          'gold',
          'royalblue',
          'magenta',
          'purple',
          'peru',
          'darkorange',
          'red',
          '#1f77b4',
          '#fa00FF',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#6f6f6f',
          '#17becf',
          '#1a55FF',
          #'olive',
          #'lime',
          #'lawngreen',
          #'dodgerblue',
          #'forestgreen'
          ]

    [nrows, ncols]  = [nspec,1] if nspec%2 else [nspec//2, 2]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(6*ncols,4*nrows))
    for n in np.arange(nspec):
        ithfig = n+1
        formula_errors=cal_errors(clth,n,lth,dell,fsky)
        plt.subplot(nrows, ncols, ithfig)
        plt.plot(lth, factor1*clth[ispecs][n].T, '--k')
        plt.errorbar(ellval, factor2*hcla[n], yerr=factor2*scla[n], ls='',ms=2,fmt='o',elinewidth=1.5,capsize=2, color='C%i' % ispecs[n], label=r"$%s\_ errors$" % spec[n])
        #plt.errorbar(lth, lth*(lth+1)/2./np.pi*clth[ispecs][n].T, yerr=formula_errors[ispecs][n]*lth*(lth+1)/2./np.pi, ls='',ms=2,fmt='s',elinewidth=1.5,capsize=2, color=cspecs[n], label=r"$formula \_ errors$")
        plt.fill_between(lth, factor1*(clth[ispecs][n].T+formula_errors) \
            , factor1*(clth[ispecs][n].T-formula_errors),facecolor='gray',alpha=0.2)
        plt.xlabel(r"$\ell$")
        plt.ylabel(r'$\mathcal{D}^{%s}_\ell$ [in  $ \mu$K${}^2$]' % label[n])
        plt.legend(fontsize="14")
    plt.savefig(output_dir+'/results/cls.pdf');
    print("Plot Ps finished! \n")

def cal_errors(clth,n,lth,dell,fsky):
  if n< 3:
      formula_errors = np.sqrt((2*clth[n]*clth[n])/(2*lth+1)/dell/fsky)
  elif n==3:
      formula_errors = np.sqrt((clth[3]*clth[3]+clth[0]*clth[1])/(2*lth+1)/dell/fsky)
  elif n==4:    
      formula_errors = np.sqrt((clth[4]*clth[4]+clth[0]*clth[2])/(2*lth+1)/dell/fsky)
  elif n==5:    
      formula_errors = np.sqrt((clth[5]*clth[5]+clth[1]*clth[2])/(2*lth+1)/dell/fsky)
  print
  return formula_errors   
