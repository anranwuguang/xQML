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
from pymaster import mask_apodization
import tools

import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var
from xqml.simulation import extrapolpixwin
from xqml.simulation import pixvar2nl
ion()
show()

input_dir  = "test_dir"
output_dir = "output_dir"
os.system("mkdir output_dir")
os.system("mkdir " + output_dir + "/results")
os.system("mkdir "+output_dir+"/EBmaps")

exp = "Big"
if len(sys.argv) > 1:
    if sys.argv[1].lower()[0] == "s":
        exp = "Small"

if exp == "Big":
    nside = 16
    dell = 1
    lmax = 3 * nside - 1  
    smoothing_deg = 8         # optional, Gaussian smoothing size
    bl_smoothing= hp.gauss_beam(deg2rad(smoothing_deg),lmax=lmax)
    lth_esti = np.arange(2,lmax+1)
    Nl2 = (lth_esti+2)*(lth_esti+1)*lth_esti*(lth_esti-1)
elif exp == "Small":
    nside = 64
    dell = 11
    lmax = 3 * nside - 1
    glat = 80
    lth_esti = np.arange((3+dell)/2,lmax-5 ,dell)
    Nl2 = (lth_esti+2)*(lth_esti+1)*lth_esti*(lth_esti-1)
else:
    print( "Need a patch !")

#Simulatin parameters   
nside_maps=512
nsimu = 100
MODELFILE = input_dir + "/Cls/PCP18_r0.05wl_K2.fits" 
Slmax = 2*nside_maps-1

s0 = timeit.default_timer()

# provide list of specs to be computed, and/or options
spec = ['TT']#, 'TE', 'TB']
pixwin = False  
ellbins = np.arange(2, lmax + 2, dell)
ellbins[-1] = lmax+1

muKarcmin = 0.5

fwhm = 0.0



##############################
#input model
clth = np.array(hp.read_cl(MODELFILE))
lth = arange(0, clth.shape[1])

clth_EE=np.zeros(clth.shape)    #clth_EE   = C^EE_\ell
clth_EE[0]=clth[1]

clth_esti=np.zeros(clth.shape)  #clth_esti = N^2_\ell C^EE_\ell
clth_esti[0]=clth[1]*(lth+2)*(lth+1)*lth*(lth-1)

pixvar = Karcmin2var(muKarcmin*1e-6, nside)
nl=pixvar2nl(pixvar,nside)
clth_Nl=np.ones(clth.shape)
clth_Nl[:]=clth_Nl[:]*nl*(lth+2)*(lth+1)*lth*(lth-1)
##############################



##############################
# Create mask
mask_in  = hp.read_map(input_dir+"/mask/Planck_512.fits",field=0,dtype=bool,verbose=0)
mask_apo = mask_apodization(mask_in, 0.3 , apotype='Smooth')
mask = tools.degrade_mask(nside, mask_apo)
#hp.write_map(output_dir+"/EBmaps/mask_apo.fits", mask_apo, dtype=np.float64)

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
esti_Nl = xqml.xQML(mask, ellbins, clth_Nl, lmax=lmax, fwhm=fwhm, spec=spec,bell=bl_smoothing, pixwin=pixwin)
print(clth_Nl)
print(esti_Nl.S)
varmap = np.diag(esti_Nl.S)
NoiseVar = np.diag(varmap)
noise = (randn(len(varmap)) * varmap**0.5).reshape(nstoke, -1)
# ############## Initialise xqml class ###############

esti = xqml.xQML(mask, ellbins, clth_esti, lmax=lmax, fwhm=fwhm, spec=spec, bell=bl_smoothing, pixwin=pixwin)
print(clth_esti)

print(esti.S)
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
fpixw = extrapolpixwin(nside_maps, Slmax, pixwin=pixwin)
Emaps = tools.read_maps(nsimu, output_dir+"/EBmaps/Emap_out", pol=False)
for n in np.arange(nsimu):
    progress_bar(n, nsimu)
    #cmb_in = np.array(hp.synfast(clth[:, :len(fpixw)]*(fpixw*bl)**2, nside_maps,
    #              pixwin=False, lmax=Slmax, fwhm=0.0, new=True, verbose=False,pol=True))
    #cmbE,cmbB = tools.QU_to_SZ(nside, cmb_in, pol=True, mask_apo=mask_apo,smoothing_deg=smoothing_deg,qml_mask=mask, output_dir=output_dir,n_index=n)
    
    cmbE = Emaps[n]
    cmbm = cmbE[mask]
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

hcl=hcl/Nl2
scl=scl/Nl2
hcla=hcla/Nl2
scla=scla/Nl2

np.savetxt(output_dir+"/results/hcl.txt",np.transpose(hcl),"%24.16e")
np.savetxt(output_dir+"/results/scl.txt",np.transpose(scl),"%24.16e")
np.savetxt(output_dir+"/results/hcla.txt",np.transpose(hcla),"%24.16e")
np.savetxt(output_dir+"/results/scla.txt",np.transpose(scla),"%24.16e")

# # ############## Compute Analytical variance ###############
# V  = esti.get_covariance(cross=True )
# Va = esti.get_covariance(cross=False)
# s2 = timeit.default_timer()
# print( "construct covariance: %.2f sec (%.2f)" % (s2-s0,s2-s1))
# ############## Plot results ###############

savefig(output_dir+"/results/cls_QML.eps")
s2 = timeit.default_timer()
print( "construct covariance: %.2f sec (%.2f)" % (s2-s0,s2-s1))
plot_Ps(output_dir,clth_EE,spec,nside,2*nside,dell,fsky)





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
