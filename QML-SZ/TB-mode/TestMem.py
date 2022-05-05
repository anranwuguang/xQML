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
from plot_Ps import *

import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var, pixvar2nl
from xqml.simulation import extrapolpixwin
ion()
show()
output_dir="results/"
os.system("rm -rf results")
os.system("mkdir results")
exp = "Big"
if len(sys.argv) > 1:
    if sys.argv[1].lower()[0] == "s":
        exp = "Small"

if exp == "Big":
    nside = 16
    dell = 1
    glat = 15
elif exp == "Small":
    nside = 64
    dell = 10
    glat = 80
else:
    print( "Need a patch !")

lmax = 3*nside-1
nsimu = 200
MODELFILE = 'r005K2.fits'
MODELFILE_ESTI = 'r005K2.fits'
Slmax = lmax

s0 = timeit.default_timer()

# provide list of specs to be computed, and/or options
spec = ['TT'] 
pixwin = True
ellbins = np.arange(2, lmax + 2, dell)
ellbins[-1] = lmax+1

muKarcmin = 0.1
fwhm = 0.0

bl=np.ones(3*nside)
l=np.arange(nside+1,3*nside)
bl[nside+1:3*nside]=0.5*(1+np.sin(l*np.pi/2/nside))
print("\n bl = ", bl)

##############################
#input model
clth = np.array(hp.read_cl(MODELFILE))
Clthshape = zeros(((6,)+shape(clth)[1:]))
Clthshape[:4] = clth
clth = Clthshape
lth = arange(2, lmax+1)


clth_esti = np.array(hp.read_cl(MODELFILE_ESTI))
lth_esti = np.arange(0, clth_esti.shape[1])
Nl2 = (lth_esti+2)*(lth_esti+1)*lth_esti*(lth_esti-1)
bell=np.sqrt(Nl2)
Clthshape_esti = zeros(((6,)+shape(clth_esti)[1:]))
Clthshape_esti[:4] = clth_esti
Clthshape_esti[4]  = clth_esti[0]
Clthshape_esti[0]  = np.zeros_like(clth_esti[3])*np.sqrt(Nl2)
clth_esti = Clthshape_esti


clth_EE = np.array(hp.read_cl(MODELFILE))
clth_EE[0]=clth_EE[3]
Clthshape_EE = zeros(((6,)+shape(clth_EE)[1:]))
Clthshape_EE[:4] = clth_EE
clth_EE = Clthshape_EE
print(clth)
print(clth_esti)
##############################



##############################
# Create mask

t, p = hp.pix2ang(nside, range(hp.nside2npix(nside)))
mask = np.ones(hp.nside2npix(nside), bool)

if exp == "Big":
    mask[abs(90 - rad2deg(t)) < glat] = False
elif exp == "Small":
    mask[(90 - rad2deg(t)) < glat] = False

fsky = np.mean(mask)
npix = sum(mask)
print("fsky=%.2g %% (npix=%d)" % (100*fsky,npix))
toGB = 1024. * 1024. * 1024.
emem = 8.*(npix*2*npix*2) * ( len(lth)*2 ) / toGB
print("mem=%.2g Gb" % emem)
##############################



stokes, spec, istokes, ispecs = getstokes( spec=spec)
print(type(ispecs))
print(stokes, spec, istokes, ispecs)
nspec = len(spec)
nstoke = len(stokes)


# ############## Generate Noise ###############
pixvar = Karcmin2var(muKarcmin*1e-6, nside)
nl=pixvar2nl(pixvar,nside)
nlth=np.ones_like(clth_esti[:nspec,:(lmax-1)])*nl
varmap = ones((nstoke * npix)) * pixvar
NoiseVar = np.diag(varmap)


# ############## Initialise xqml class ###############

esti = xqml.xQML(mask, ellbins, clth_esti, lmax=lmax, fwhm=fwhm, spec=spec, pixwin=pixwin)
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
#    nl = len(esti.El)
#    Wll = np.asarray( [np.sum(E * P) for E in esti.El for P in esti.Pl] ).reshape(nl,nl)
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
    Wll = np.asarray([np.sum(CaP * CbP) for CaP in CaPl for CbP in CbPl]).reshape(nl,nl)
    s2 = timeit.default_timer()
    print( "Construct Wll: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2
    CbPl = 0

    esti.El = [np.dot(CaP, invCb) for CaP in CaPl]
#    esti.El = xqml.estimators.El(invCa, invCb, esti.Pl)
    s2 = timeit.default_timer()
    print( "Construct El: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2
    CaPl =0

    esti.bias = xqml.estimators.biasQuadEstimator(esti.NA, esti.El)
    print("esti.bias = ",esti.bias)
    s2 = timeit.default_timer()
    print( "Construct bias: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2


esti.invW = linalg.inv(Wll)
print("invW.shape = ",esti.invW.shape)
s2 = timeit.default_timer()
print( "inv W: %.2f sec (%.2f)" % (s2-s0,s2-s1))
s1=s2

#esti.construct_esti( NA=NoiseVar, NB=NoiseVar)
#s2 = timeit.default_timer()
#print( "Construct esti: %.2f sec (%.2f)" % (s2-s0,s2-s1))
ellval = esti.lbin()

# ############## Construct MC ###############
# maps_dir = "/home/jm/data/Data_common/TQUmaps/r0_05_lmax1024/cmb"
# maps_in=tools.read_maps(nsimu, maps_dir)
allcla = []
allcl = []
t = []
bl = hp.gauss_beam(deg2rad(fwhm), lmax=Slmax)
fpixw = extrapolpixwin(nside, Slmax, pixwin=pixwin)
print(clth[:, :len(fpixw)]*(fpixw*bl)**2)
for n in np.arange(nsimu):
    progress_bar(n, nsimu)
    cmbA = np.array(hp.synfast(clth[:, :len(fpixw)]*(fpixw*bl)**2, nside,
                pixwin=False, lmax=Slmax, fwhm=0.0, new=True, verbose=False))
    # cmbB = np.array(hp.synfast(clth[:, :len(fpixw)]*(fpixw*bl)**2, nside,
    #                pixwin=False, lmax=Slmax, fwhm=0.0, new=True, verbose=False))
    # cmbA = hp.ud_grade(maps_in[n],nside)
    # cmbB = hp.ud_grade(maps_in[n],nside)
    map_TT = cmbA[0]
    #map_EE = tools.degrade_maps_qu2pure(nside,cmbA, pol=True, ispec=1)
    map_BB = tools.degrade_maps_qu2pure(nside,cmbA, pol=True, ispec=2)
    #hp.write_map(output_dir+"ud_maps_%d.fits"%n,map_EE,dtype=np.float64)
    # alm_TQU = hp.map2alm(cmbA, lmax=Slmax,pol=True,use_weights=True,iter=3)
    # alm_TT = alm_TQU[0]
    # alm_EE = alm_TQU[1]
    # alm_BB = alm_TQU[2]
    # map_out = hp.alm2map(alm_TQU, nside= nside, pixwin=False) 
    # map_TT = hp.alm2map(alm_TT, nside= nside, pixwin=False) 
    # map_EE = hp.alm2map(alm_EE, nside= nside, pixwin=False) 
    # map_BB = hp.alm2map(alm_BB, nside= nside, pixwin=False) 
    # hp.write_map(output_dir+"ud_maps_%d.fits"%n,map_out,dtype=np.float64)
    # # cmbmA   = map_out[[0]][:, mask]
    # # cmbmB   = map_out[[0]][:, mask]
    cmbmA   = map_TT[mask]
    cmbmB   = map_BB[mask]

    dmA = cmbmA + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
    dmB = cmbmB + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
    s1 = timeit.default_timer()
    allcl.append(tools.get_spectra(esti, dmA, dmB))
    t.append( timeit.default_timer() - s1)
    
s2 = timeit.default_timer()
print( "get_spectra: %.2f (%.2f sec)" % (s2-s0,mean(t)))
s1=s2
allcl=allcl/np.sqrt(Nl2)[2:(lmax+1)]
hcl  = np.mean(allcl, 0)
scl  = np.std(allcl, 0)



np.savetxt(output_dir+"hcl.txt",np.transpose(hcl),"%24.16e")
np.savetxt(output_dir+"scl.txt",np.transpose(scl),"%24.16e")

# # ############## Compute Analytical variance ###############
# V  = esti.get_covariance(cross=True )
# Va = esti.get_covariance(cross=False)
# s2 = timeit.default_timer()
# print( "construct covariance: %.2f sec (%.2f)" % (s2-s0,s2-s1))
# ############## Plot results ###############

figure(figsize=[12, 8])
clf()
Delta = (ellbins[1:] - ellbins[:-1])/2.

subplot(2, 1, 1)
title("Cross")
plot(lth, (lth*(lth+1)/2./np.pi)[:, None]*clth_esti[ispecs][:, lth].T, '--k')
for s in np.arange(nspec):
    errorbar(ellval, ellval*(ellval+1)/2./np.pi*hcl[s], yerr=scl[s], xerr=Delta, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" % spec[s])
semilogy()
ylabel(r"$D_\ell$")
legend(loc=4, frameon=True)

# subplot(2, 2, 2)
# title("Auto")
# plot(lth,(lth*(lth+1)/2./np.pi)[:, None]*clth[ispecs][:, lth].T, '--k')
# for s in np.arange(nspec):
#     errorbar(ellval, ellval*(ellval+1)/2./np.pi*hcla[s], yerr=scla[s], xerr=Delta, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" % spec[s])
# semilogy()

# subplot(3, 2, 3)
# for s in np.arange(nspec):
#     plot(ellval, scl[s], '--', color='C%i' % ispecs[s], label=r"$\sigma^{%s}_{\rm MC}$" % spec[s])
#     plot(ellval, sqrt(diag(V)).reshape(nspec, -1)[s], 'o', color='C%i' % ispecs[s])
# ylabel(r"$\sigma(C_\ell)$")
# semilogy()

# subplot(3, 2, 4)
# for s in np.arange(nspec):
#     plot(ellval, scla[s], ':', color='C%i' % ispecs[s], label=r"$\sigma^{%s}_{\rm MC}$" % spec[s])
#     plot(ellval, sqrt(diag(Va)).reshape(nspec, -1)[s], 'o', color='C%i' % ispecs[s])
# semilogy()

subplot(2, 1, 2)
for s in np.arange(nspec):
    plot(ellval, (hcl[s]-esti.BinSpectra(clth_esti)[s])/(esti.BinSpectra(clth_esti)[s])*100, '--o', color='C%i' % ispecs[s])
ylabel(r"$R[C_\ell] \%$")
xlabel(r"$\ell$")
#ylim(-3, 3)
grid()

# subplot(2, 2, 4)
# for s in np.arange(nspec):
#     plot(ellval, (hcla[s]-esti.BinSpectra(clth)[s])/(esti.BinSpectra(clth)[s])*100, '--o', color='C%i' % ispecs[s])
# xlabel(r"$\ell$")
#ylim(-3, 3)
grid()

#show()
savefig("./cls.eps")
s2 = timeit.default_timer()
print( "construct covariance: %.2f sec (%.2f)" % (s2-s0,s2-s1))


plot_Ps(output_dir, clth_esti,spec, nside,2*nside-1,dell,fsky)
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
