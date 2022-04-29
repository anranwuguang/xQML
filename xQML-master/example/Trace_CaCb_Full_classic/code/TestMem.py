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
from plot_Ps import plot_Ps 

import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var
from xqml.simulation import extrapolpixwin
ion()
show()

output_dir="Trace_CaCb_Full_classic/"
os.system("mkdir " + output_dir)
os.system("mkdir " + output_dir + "code")
os.system("cp TestMem.py " + output_dir + "code")
os.system("cp plot_Ps.py " + output_dir + "code")

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

#lmax = nside
lmax = 2 * nside - 1
nsimu = 100
MODELFILE = 'planck_base_planck_2015_TTlowP.fits'
Slmax = lmax

s0 = timeit.default_timer()

# provide list of specs to be computed, and/or options
spec = ['TT','EE','BB','TE','TB','EB']#, 'TE', 'TB']
#spec = ['TT']#, 'TE', 'TB']
pixwin = True
ellbins = np.arange(2, lmax + 2, dell)
ellbins[-1] = lmax+1

muKarcmin = 1.0
fwhm = 0.0



##############################
#input model
clth = np.array(hp.read_cl(MODELFILE))
Clthshape = zeros(((6,)+shape(clth)[1:]))
Clthshape[:4] = clth
clth = Clthshape
np.savetxt(output_dir+"clth.txt",np.transpose(clth),"%10.6e")
lth = arange(2, lmax+1)
##############################



##############################
# Create mask

t, p = hp.pix2ang(nside, range(hp.nside2npix(nside)))
mask = np.ones(hp.nside2npix(nside), bool)
# import random
# random.shuffle(mask)

if exp == "Big":
#    mask[abs(90 - rad2deg(t)) < glat] = False
    mask[abs(90 - rad2deg(t)) < glat] = False
elif exp == "Small":
    mask[(90 - rad2deg(t)) < glat] = False

#print("mask = ",mask)
vc = np.array(hp.pix2vec(nside=nside,ipix=0))
vr = np.array(hp.pix2vec(nside=nside,ipix=1))
print("vc, vr = ",vc, vr)
print(np.dot(vc,vr))
print(np.sum(vc*vr))

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
esti = xqml.xQML(mask, ellbins, clth, lmax=lmax, fwhm=fwhm, spec=spec)
s1 = timeit.default_timer()
print( "Init: %.2f sec (%.2f)" % (s1-s0,s1-s0))

esti.NA = NoiseVar
esti.NB = NoiseVar

invCa = xqml.xqml_utils.pd_inv(esti.S + esti.NA)
invCb = xqml.xqml_utils.pd_inv(esti.S + esti.NB)
#invCa = linalg.inv(esti.S + esti.NA)
#invCb = linalg.inv(esti.S + esti.NB)
np.savetxt(output_dir+"S.txt",np.array(esti.S),"%10.6e",delimiter=',')
np.savetxt(output_dir+"NA.txt",np.array(esti.S),"%10.6e",delimiter=',')
np.savetxt(output_dir+"invCa.txt",np.array(invCa),"%10.6e",delimiter=',')
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
    #Wll = np.asarray([np.sum(CaP * CbP) for CaP in CaPl for CbP in CbPl]).reshape(nl,nl)
    Wll = np.asarray([np.trace(np.dot(CaP,CbP)) for CaP in CaPl for CbP in CbPl]).reshape(nl,nl)
    np.savetxt(output_dir+"Wll.txt",Wll,"%10.6e")
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
    s2 = timeit.default_timer()
    print( "Construct bias: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2


esti.invW = linalg.inv(Wll)
np.savetxt(output_dir+"invW.txt",np.array(esti.invW),"%10.6e")
s2 = timeit.default_timer()
print( "inv W: %.2f sec (%.2f)" % (s2-s0,s2-s1))
s1=s2

#esti.construct_esti( NA=NoiseVar, NB=NoiseVar)
#s2 = timeit.default_timer()
#print( "Construct esti: %.2f sec (%.2f)" % (s2-s0,s2-s1))
ellval = esti.lbin()


# ############## Construct MC ###############
allcla = []
allcl = []
t = []
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

# # ############## Compute Analytical variance ###############
# V  = esti.get_covariance(cross=True )
# Va = esti.get_covariance(cross=False)
# s2 = timeit.default_timer()
# print( "construct covariance: %.2f sec (%.2f)" % (s2-s0,s2-s1))
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

# subplot(2, 2, 3)
# for s in np.arange(nspec):
#     plot(ellval, (hcl[s]-esti.BinSpectra(clth)[s])/(esti.BinSpectra(clth)[s])*100, '--o', color='C%i' % ispecs[s])
# ylabel(r"$R[C_\ell] \%$")
# xlabel(r"$\ell$")
# #ylim(-3, 3)
# grid()

# subplot(2, 2, 4)
# for s in np.arange(nspec):
#     plot(ellval, (hcla[s]-esti.BinSpectra(clth)[s])/(esti.BinSpectra(clth)[s])*100, '--o', color='C%i' % ispecs[s])
# xlabel(r"$\ell$")
# #ylim(-3, 3)
grid()

#show()
savefig(output_dir+"./cls_QML.eps")
s2 = timeit.default_timer()
print( "construct covariance: %.2f sec (%.2f)" % (s2-s0,s2-s1))

plot_Ps(output_dir,MODELFILE,spec)





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
