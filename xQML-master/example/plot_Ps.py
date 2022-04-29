import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import timeit

import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var
from xqml.simulation import extrapolpixwin


def plot_Ps(output_dir, MODELFILE, spec,nside,lmax):
    # # ############## Initial parameters ###############
    
    dell=1
    fsky=0.7
    lth = np.arange(2, lmax+1)
    ellval = np.arange((3+dell)/2, lmax+1, dell)
    # # ################################################# 
    ##################  Input model  #####################
    clth = np.array(hp.read_cl(MODELFILE))
    Clthshape = np.zeros(((6,)+np.shape(clth)[1:]))
    Clthshape[:4] = clth
    clth = Clthshape[:,lth]
    ##############################


    stokes, spec, istokes, ispecs = getstokes( spec=spec)
    print(stokes, spec, istokes, ispecs)
    #cspecs=np.array(ispecs)+3
    cspecs = [
          'yellowgreen',
          #'olive',
          'gold',
          'royalblue',
          #'lime',
          #'lawngreen',
          #'dodgerblue',
          #'forestgreen',
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
          '#1a55FF']
    hcla  = np.loadtxt(output_dir+"hcla.txt", unpack=True)
    scla  = np.loadtxt(output_dir+"scla.txt", unpack=True)
    #hnla  = np.loadtxt(output_dir+"hnla.txt", unpack=True)
    if len(hcla.shape) == 1:
        hcla = np.array([hcla])
        scla = np.array([scla])
     #   hnla = np.array([hnla])
        

    hcla = hcla[:,:len(ellval)]
    scla = scla[:,:len(ellval)]
    #hnla = hnla[:,:len(ellval)]
    nspec = hcla.shape[0]
    formula_errors = np.zeros_like(clth)


    formula_errors[0] = np.sqrt((clth[0]*clth[0]+clth[0]*clth[0])/(2*lth+1)/dell/fsky)
    formula_errors[1] = np.sqrt((clth[1]*clth[1]+clth[1]*clth[1])/(2*lth+1)/dell/fsky)
    formula_errors[2] = np.sqrt((clth[2]*clth[2]+clth[2]*clth[2])/(2*lth+1)/dell/fsky)
    formula_errors[3] = np.sqrt((clth[3]*clth[3]+clth[0]*clth[1])/(2*lth+1)/dell/fsky)
    formula_errors[4] = np.sqrt((clth[4]*clth[4]+clth[0]*clth[2])/(2*lth+1)/dell/fsky)
    formula_errors[5] = np.sqrt((clth[5]*clth[5]+clth[1]*clth[2])/(2*lth+1)/dell/fsky)

    np.savetxt(output_dir+"formula_errors.txt",np.transpose(formula_errors), "%10.6e")
    [nrows, ncols]  = [nspec,1] if nspec%2 else [nspec//2, 2]


    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(6*ncols,4*nrows))
    for n in np.arange(nspec):
        ithfig = n+1
        plt.subplot(nrows, ncols, ithfig)
        plt.plot(lth, (lth*(lth+1)/2./np.pi)*clth[ispecs][n].T, '--k')
        plt.errorbar(ellval, ellval*(ellval+1)/2./np.pi*hcla[n], yerr=scla[n]*ellval*(ellval+1)/2./np.pi, ls='',ms=2,fmt='o',elinewidth=1.5,capsize=2, color='C%i' % ispecs[n], label=r"$%s\_ errors$" % spec[n])
        plt.errorbar(lth+dell/3, lth*(lth+1)/2./np.pi*hcla[n], yerr=formula_errors[ispecs][n]*lth*(lth+1)/2./np.pi, ls='',ms=2,fmt='s',elinewidth=1.5,capsize=2, color=cspecs[n], label=r"$formula \_ errors$")
        plt.xlabel(r"$\ell$")
        plt.ylabel(r'$\mathcal{D}^{%s}_\ell$ [in  $ \mu$K${}^2$]' % spec[n])
        plt.legend(fontsize="14")
    plt.savefig(output_dir+'cls.eps')
    print("Plot Ps finished! \n")
