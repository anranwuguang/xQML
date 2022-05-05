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

from pymaster import mask_apodization

import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var
from xqml.simulation import extrapolpixwin
ion()
show()

input_dir  = "input_dir"
output_dir = "output_dir"
os.system("mkdir " + output_dir)
os.system("mkdir " + output_dir + "/results")


exp = "Small"
if len(sys.argv) > 1:
    if sys.argv[1].lower()[0] == "s":
        exp = "Small"

if exp == "Big":
    nside = 8
    dell = 1
    #glat = 15
elif exp == "Small":
    nside = 32
    dell = 11
    #glat = 80
else:
    print( "Need a patch !")

nside_maps =512
lmax = 3 * nside - 1
nsimu = 100
MODELFILE = input_dir + "/Cls/PCP18_r0.05wl_K2.fits" 
Slmax = 2*nside_maps

s0 = timeit.default_timer()

# provide list of specs to be computed, and/or options
spec = ['TT','EE','BB','TE','TB','EB']
pixwin = False
ellbins = np.arange(2, lmax + 2, dell)
ellbins[-1] = lmax+1

muKarcmin = 1.0
fwhm = 0.0


bl=np.ones(3*nside)
l=np.arange(nside+1,3*nside)
bl[nside+1:3*nside]=0.5*(1+np.sin(l*np.pi/2/nside))
##############################
#input model
clth = np.array(hp.read_cl(MODELFILE))
Clthshape = zeros(((6,)+shape(clth)[1:]))
Clthshape[:4] = clth
clth = Clthshape
lth = arange(2, lmax+1)
##############################



##############################
# Create mask

t, p = hp.pix2ang(nside, range(hp.nside2npix(nside)))

#mask = np.ones(hp.nside2npix(nside), bool)
#if exp == "Big":
#    mask[abs(90 - rad2deg(t)) < glat] = False
#elif exp == "Small":
#    mask[(90 - rad2deg(t)) < glat] = False

mask_in  = hp.read_map(input_dir+'/Mask/AliCPT_512.fits',field=0,dtype=bool,verbose=0)
mask_apo = mask_apodization(mask_in, 2, apotype='C2')
mask = tools.degrade_mask(nside, mask_apo)

fsky = np.mean(mask)
npix = sum(mask)
print("fsky=%.2g %% (npix=%d)" % (100*fsky,npix))
toGB = 1024. * 1024. * 1024.
emem = 8.*(npix*2*npix*2) * ( len(lth)*2 ) / toGB
print("mem=%.2g Gb" % emem)
##############################



stokes, spec, istokes, ispecs = getstokes( spec=spec)
print(stokes, spec, istokes, ispecs)
nspec = len(spec)
nstoke = len(stokes)


# ############## Generate Noise ###############
pixvar = Karcmin2var(muKarcmin*1e-6, nside)
varmap = ones((nstoke * npix)) * pixvar
NoiseVar = np.diag(varmap)

noise = (randn(len(varmap)) * varmap**0.5).reshape(nstoke, -1)



# ############## Initialise xqml class ###############
esti = xqml.xQML(mask, ellbins, clth, lmax=lmax, fwhm=fwhm, spec=spec,pixwin=pixwin,bell=bl)
s1 = timeit.default_timer()
print( "Init: %.2f sec (%.2f)" % (s1-s0,s1-s0))

esti.NA = NoiseVar
esti.NB = NoiseVar

invCa = xqml.xqml_utils.pd_inv(esti.S + esti.NA)
invCb = xqml.xqml_utils.pd_inv(esti.S + esti.NB)

s2 = timeit.default_timer()
print( "Inv C: %.2f sec (%.2f)" % (s2-s0,s2-s1))
s1 = s2

meth = "classic"
#meth = "long"

if meth == "classic":
    esti.El = xqml.estimators.El(invCa, invCb, esti.Pl)
    s2 = timeit.default_timer()
    print( "Construct El: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2

    Wll = xqml.estimators.CrossWindowFunction(esti.El, esti.Pl)

    s2 = timeit.default_timer()
    print( "Construct W: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1=s2
    esti.Pl = 0.

    esti.bias = xqml.estimators.biasQuadEstimator(esti.NA, esti.El)
    s2 = timeit.default_timer()
    print( "Construct bias: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2

else:
    nl = len(esti.Pl)
    CaPl = [np.dot(invCa, P) for P in esti.Pl]
    CbPl = [np.dot(invCb, P) for P in esti.Pl]
    esti.Pl = 0
    Wll = np.asarray([np.trace(np.dot(CaP,CbP)) for CaP in CaPl for CbP in CbPl]).reshape(nl,nl)
    np.savetxt(output_dir+"Wll.txt",Wll,"%10.6e")
    s2 = timeit.default_timer()
    print( "Construct Wll: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2
    CbPl = 0

    esti.El = [np.dot(CaP, invCb) for CaP in CaPl]
    s2 = timeit.default_timer()
    print( "Construct El: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2
    CaPl =0

    esti.bias = xqml.estimators.biasQuadEstimator(esti.NA, esti.El)
    s2 = timeit.default_timer()
    print( "Construct bias: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2


esti.invW = linalg.inv(Wll)
s2 = timeit.default_timer()
print( "inv W: %.2f sec (%.2f)" % (s2-s0,s2-s1))
s1=s2
ellval = esti.lbin()


# ############## Construct MC ###############
allcla = []
allcl = []
t = []
bl = hp.gauss_beam(deg2rad(fwhm), lmax=Slmax)
fpixw = extrapolpixwin(nside_maps, Slmax, pixwin=pixwin)
for n in np.arange(nsimu):
    progress_bar(n, nsimu)
    cmb_in = np.array(hp.synfast(clth[:, :len(fpixw)]*(fpixw*bl)**2, nside_maps,
                   pixwin=False, lmax=Slmax, fwhm=0.0, new=True, verbose=False))
    cmb= tools.degrade_maps(nside, cmb_in, pol=True, qml_mask=mask)
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



np.savetxt(output_dir+"/results/hcl.txt",np.transpose(hcl),"%24.16e")
np.savetxt(output_dir+"/results/scl.txt",np.transpose(scl),"%24.16e")
np.savetxt(output_dir+"/results/hcla.txt",np.transpose(hcla),"%24.16e")
np.savetxt(output_dir+"/results/scla.txt",np.transpose(scla),"%24.16e")
# ############## Plot results ###############

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

savefig(output_dir+"/results/cls_QML.eps")
s2 = timeit.default_timer()
print( "construct covariance: %.2f sec (%.2f)" % (s2-s0,s2-s1))

#plot_Ps(output_dir,MODELFILE,spec,nside,lmax-nside)

plot_Ps(output_dir, clth, spec,nside,2*nside+dell,dell,fsky)



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
