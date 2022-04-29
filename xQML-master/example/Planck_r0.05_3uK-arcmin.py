  #!/usr/bin/env python
"""
Test script for xQML

Author: Vanneste
"""

from __future__ import division

import numpy as np
import healpy as hp
from pylab import *
import astropy.io.fits as fits
import timeit
import sys
import os
import tools
from plot_Ps import plot_Ps 

import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var
from xqml.simulation import extrapolpixwin
ion()
show()

input_dir  = "data/"
output_dir = "Planck_r0.05_3uK-arcmin/"
os.system("mkdir " + output_dir)
os.system("mkdir " + output_dir + "code")
os.system("cp example/Planck_r0.05_3uK-arcmin.py " + output_dir + "code")
os.system("cp example/plot_Ps.py " + output_dir + "code")
os.system("cp example/tools.py " + output_dir + "code")



##########################  estimator  ##########################
nside  = 16
dell   = 1
lmax   = 3 * nside - 1
fwhm   = 0.0
spec   = ['TT']
#spec  = ['TT','EE','BB','TE','TB','EB']
pixwin = False
cos_bl = True

##########################  Clth  ##########################
MODELFILE = input_dir + "fits2cl/r005K2.fits" 
##########################  Mask  ##########################
#MASKFILE  = ''
MASKFILE  = input_dir+'maskfile/mask.fits'
apo_scale = 2


glat  = None
########################## Sample ##########################
nsimu = 1000
#maps_dir   = ''
#noises_dir = ''
maps_dir   = input_dir + "degrade_maps_sig_noise/s_n_"
noises_dir = input_dir + "degrade_maps_noise/n_"
############################################################




s0 = timeit.default_timer()
##############################
stokes, spec, istokes, ispecs = getstokes( spec=spec)
print(stokes, spec, istokes, ispecs)
nspec = len(spec)
nstoke = len(stokes)

clth = tools.read_clth(MODELFILE)
if cos_bl==True:
    bl = tools.cos_bl(nside)
else:
    bl= None

mask = tools.Creat_mask(MASKFILE,apo_scale,nside,glat)
npix = sum(mask)
if noises_dir !='':
    noises = tools.read_maps(nsimu, noises_dir)
    NoiseVar = tools.get_maps_covariance(noises, istokes, mask)
else:
    muKarcmin = 0.1
    pixvar = Karcmin2var(muKarcmin*1e-6, nside)
    varmap = ones((nstoke * npix)) * pixvar
    NoiseVar = np.diag(varmap)
##############################

# ############## Initialise xqml class ###############
ellbins = np.arange(2, lmax + 2, dell)
ellbins[-1] = lmax+1
esti = xqml.xQML(mask, ellbins, clth, NA=NoiseVar, NB=NoiseVar, lmax=lmax, fwhm=fwhm, spec=spec,pixwin=pixwin,bell=bl)
ellval = esti.lbin()
# ############## Construct MC ###############

if maps_dir != '':
    maps   = tools.read_maps(nsimu, maps_dir)
    allcla = []
    allcl = []
    t = []
    for n in np.arange(nsimu):
        progress_bar(n, nsimu)
        cmb=maps[n]    
        cmbm   = cmb[istokes][:, mask]
        dmA = cmbm 
        dmB = cmbm 
        s1 = timeit.default_timer()
        allcl.append(esti.get_spectra(dmA, dmB))
        t.append( timeit.default_timer() - s1)
        allcla.append(esti.get_spectra(dmA))
    s2 = timeit.default_timer()
    print( "get_spectra: %.2f (%.2f sec)" % (s2-s0,mean(t)))
    s1=s2

    hcl  = np.mean(allcl, 0)
    scl  = np.std(allcl, 0)
    hcla = np.mean(allcla, 0)
    scla = np.std(allcla, 0)
    np.savetxt(output_dir+"/hcl.txt",np.transpose(hcl),"%10.6e")
    np.savetxt(output_dir+"/scl.txt",np.transpose(scl),"%10.6e")
    np.savetxt(output_dir+"/hcla.txt",np.transpose(hcla),"%10.6e")
    np.savetxt(output_dir+"/scla.txt",np.transpose(scla),"%10.6e")
    if noises_dir !='':
        allnla = []
        allnl = []
        for n in np.arange(nsimu):
            progress_bar(n, nsimu)
            noise=noises[n]
            noisem   = noise[istokes][:, mask]
            dnA = noisem 
            dnB = noisem 
            s1 = timeit.default_timer()
            allnl.append(esti.get_spectra(dnA, dnB))
            t.append( timeit.default_timer() - s1)
            allnla.append(esti.get_spectra(dnA))
        s2 = timeit.default_timer()
        print( "get_spectra: %.2f (%.2f sec)" % (s2-s0,mean(t)))
        s1=s2
        hnl  = np.mean(allnl, 0)
        hnla = np.mean(allnla, 0)
        np.savetxt(output_dir+"/hnl.txt",np.transpose(hnl),"%10.6e")
        np.savetxt(output_dir+"/hnla.txt",np.transpose(hnla),"%10.6e")


else:
    allcla = []
    allcl = []
    t = []
    Slmax = lmax
    bl = hp.gauss_beam(deg2rad(fwhm), lmax=Slmax)
    fpixw = extrapolpixwin(nside, Slmax, pixwin=pixwin)
    for n in np.arange(nsimu):
        progress_bar(n, nsimu)
        cmb = np.array(hp.synfast(clth[:, :len(fpixw)]*(fpixw*bl)**2, nside,
                       pixwin=False, lmax=Slmax, fwhm=0.0, new=True, verbose=False))
        cmbm = cmb[istokes][:, mask]
        dmA = cmbm + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
        dmB = cmbm + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
        s1 = timeit.default_timer()
        allcl.append(esti.get_spectra(dmA, dmB))
        t.append( timeit.default_timer() - s1)
        allcla.append(esti.get_spectra(dmA))

    print( "get_spectra: %.2f (%.2f sec)" % (timeit.default_timer()-s0,mean(t)))

    hcl  = np.mean(allcl, 0)
    scl  = np.std(allcl, 0)
    hcla = np.mean(allcla, 0)
    scla = np.std(allcla, 0)
    np.savetxt(output_dir+"/hcl.txt",np.transpose(hcl),"%24.16e")
    np.savetxt(output_dir+"/scl.txt",np.transpose(scl),"%24.16e")
    np.savetxt(output_dir+"/hcla.txt",np.transpose(hcla),"%24.16e")
    np.savetxt(output_dir+"/scla.txt",np.transpose(scla),"%24.16e")

# ############## Plot results ###############
lth = arange(2, lmax+1)
figure(figsize=[12, 8])
clf()
Delta = (ellbins[1:] - ellbins[:-1])/2.

subplot(2, 2, 1)
title("Cross")
plot(lth, (lth*(lth+1)/2./np.pi)[:, None]*clth[ispecs][:, lth].T, '--k')
for s in np.arange(nspec):
    errorbar(ellval, ellval*(ellval+1)/2./np.pi*hcl[s], yerr=scl[s], xerr=Delta, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" % spec[s])
semilogy()
ylabel(r"$D_\ell$")
legend(loc=4, frameon=True)

subplot(2, 2, 2)
title("Auto")
plot(lth,(lth*(lth+1)/2./np.pi)[:, None]*clth[ispecs][:, lth].T, '--k')
for s in np.arange(nspec):
    errorbar(ellval, ellval*(ellval+1)/2./np.pi*hcla[s], yerr=scla[s], xerr=Delta, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" % spec[s])
semilogy()
grid()

savefig(output_dir+"./cls_QML.eps")
s2 = timeit.default_timer()
print( "construct covariance: %.2f sec (%.2f)" % (s2-s0,s2-s1))

plot_Ps(output_dir,MODELFILE,spec,nside,lmax=2*nside)





## if __name__ == "__main__":
##     """
##     Run the doctest using

##     python simulation.py

##     If the tests are OK, the script should exit gracefuly, otherwise the
##     failure(s) will be printed out.
##     """
##     import doctest
##     if np.__version__ >= "1.14.0":
##         np.set_printoptions(legacy="1.13")
##     doctest.testmod()
