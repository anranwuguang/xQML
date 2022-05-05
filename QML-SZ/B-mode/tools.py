import numpy as np
import healpy as hp
from pylab import *
import os
import xqml
from xqml.xqml_utils import progress_bar
def read_maps(nsimu,  maps_dir, pol=True):
    """cut high multipoles of input map and degrade it to low resolution
    Parameters
    ----------
    nsimu：integer
        mumber of samples

    qml_nside : integer
       nisde of  low multipoles maps

    mapsA, mapsB: numpy.ndarray
             
        mapsA noise maps of chanel A
        mapsB noise maps of chanel B（optional）

    Returns
    ----------
    maps:
        Return a degrade maps with Nside = qml_nside 
    """
    print()
    print("Read maps from %s"%maps_dir)

    maps=[]
    for n in np.arange(nsimu):
        progress_bar(n, nsimu)
        if pol == True:
            map_in = hp.read_map(maps_dir+"%d.fits"%n,field=(0,1,2),dtype=float,verbose=0)
        else:
            map_in = hp.read_map(maps_dir+"%d.fits"%n,field=0,dtype=float,verbose=0)
        maps.append(map_in)
    maps=np.array(maps)

    return maps

def degrade_maps_alm(qml_nside, maps_in, mask,smoothing_deg=0,qml_mask=(),output_dir=None, n1=None):
    """cut high multipoles of input map and degrade it to low resolution
    Parameters
    ----------
    nsimu：integer
        mumber of samples

    qml_nside : integer
       nisde of  low multipoles maps

    mapsA, mapsB: numpy.ndarray
             
        mapsA noise maps of chanel A
        mapsB noise maps of chanel B（optional）

    Returns
    ----------
    maps:
        Return a degrade maps with Nside = qml_nside
    """
    # print()
    # print("down grade maps to Nside = %d"%qml_nside)
    if len(maps_in.shape) <=2:
        nsimu=1
        maps_in=np.array([maps_in])
    else:
        nsimu=maps_in.shape[0]
    nside=hp.npix2nside(maps_in.shape[-1])
    Slmax_old=3*nside-1
    Slmax_new=3*qml_nside-1
    if (Slmax_new> Slmax_old):
        print("Slmax_new must smaller than Slmax_old!")
        exit()
    
    npix= hp.nside2npix(qml_nside)
    if qml_mask == ():
        qml_mask=np.ones(npix,bool)
    else:
        if npix != len(qml_mask):
            print("The nside of qml_mask inconsistent with qml_nside.")

    ALM = hp.Alm
    maps=[]
    for n in np.arange(nsimu): 
        map_in=maps_in[0][0]
        map_in=map_in*mask  
        # hp.mollview(map_in, cmap=plt.cm.jet, title='Q map output')
        # plt.savefig(output_dir+'/map_out_%d.pdf'%n1,bbox_inches='tight',pad_inches=0.1)
        alm_old=hp. map2alm(map_in,lmax=Slmax_old,pol=False)
        if smoothing_deg>0:
            hp.smoothalm(alm_old,fwhm=deg2rad(smoothing_deg))
        alm_new=np.zeros_like(np.array(alm_old)[:ALM.getsize(Slmax_new)])
        for l in np.arange(Slmax_new+1) :
            for m in np.arange(l+1) :
                idx_new = ALM.getidx(Slmax_new, l, m)
                idx_old = ALM.getidx(Slmax_old, l, m) 
                alm_new[idx_new]=alm_old[idx_old]
        map_out = hp.alm2map(alm_new, nside= qml_nside, pixwin=False) 
        # if output_dir != None:
        #     hp.write_map(output_dir+"/Tmaps/Tmap_out%d.fits"%n1,map_out,dtype=np.float64)
        
        #maps.append(map_out)
    return map_out

def degrade_mask(qml_nside,  mask_in):
    """down grade mask to low resolution
    Parameters
    ----------
    qml_nside : integer
        nisde of  low multipoles maps

    mapsA, mapsB: numpy.ndarray
        mapsA noise maps of chanel A
        mapsB noise maps of chanel B（optional）

    Returns
    ----------
    map:
        Return a degrade map with Nside = qml_nside
    """
    print()
    print("degrade mask to Nside = %d"%qml_nside)
    mask_out = hp.ud_grade(mask_in,qml_nside)
    index=np.where(mask_out<0.999)
    mask_out[index]=0
    index=np.where(mask_out>=0.999)
    mask_out[index]=1
    mask_out=np.asarray(mask_out,bool)
    return mask_out

def QU_to_SZ(qml_nside, maps_in, pol, mask_apo,smoothing_deg=0,qml_mask=(),output_dir=None,n_index=0):
    """cut high multipoles of input map and degrade it to low resolution
    Parameters
    ----------
    nsimu：integer
        mumber of samples

    qml_nside : integer
       nisde of  low multipoles maps
 
    mapsA, mapsB: numpy.ndarray
             
        mapsA noise maps of chanel A
        mapsB noise maps of chanel B（optional）

    Returns
    ----------
    maps:
        Return a degrade maps with Nside = qml_nside
    """
    
    #print("\n QU_to_SZ")
    #print(maps_in.shape)
    if len(maps_in.shape) ==2 and maps_in.shape[0]==3:
        nsimu=1
        maps_in = np.array([maps_in])
    else:
        nsimu=maps_in.shape[0]
    nside = hp.npix2nside(maps_in.shape[-1])
    #print("nside = ",nside)
    Slmax_old = 2*nside-1
    Slmax_new = 3*qml_nside-1
    
    # npix= hp.nside2npix(qml_nside)
    # if qml_mask == ():
    #     qml_mask=np.ones(npix,bool)
    # else:
    #     if npix != len(qml_mask):
    #         print("The nside of qml_mask inconsistent with qml_nside.")

    W0, W1_1, W1_2, W2_1, W2_2 = Trans_windowmap(mask_apo)
    #print("Designed by Chen Jiming!")
    ALM = hp.Alm
    Emaps=[]
    Bmaps=[]
    snl1 = np.zeros(ALM.getsize(lmax=Slmax_old,mmax=Slmax_old))
    snl2 = np.zeros(ALM.getsize(lmax=Slmax_old,mmax=Slmax_old))
    ll = np.zeros(ALM.getsize(lmax=Slmax_old,mmax=Slmax_old))   
    mm = np.zeros(ALM.getsize(lmax=Slmax_old,mmax=Slmax_old))
    for l in np.arange(0,Slmax_old + 1) :
        for m in np.arange(l+1) :            
            idx = ALM.getidx(Slmax_old, l, m)
            ll[idx]=l*1.0
            mm[idx]=m*1.0
            snl1[idx] = np.sqrt((l+1)*l)
            snl2[idx] = np.sqrt((l+2)*(l+1)*l*(l-1))
    smap0 = np.zeros([2,hp.nside2npix(nside)])
    smap1 = np.zeros([2,hp.nside2npix(nside)])
    smap2 = np.zeros([2,hp.nside2npix(nside)])
    for n in np.arange(nsimu):
        #progress_bar(n, nsimu)
        n1 = n+1
        cmbin=maps_in[n]
        if smoothing_deg>0:
            bl= hp.gauss_beam(deg2rad(smoothing_deg),lmax=Slmax_old)
            cmbi = hp.smoothing(cmbin, beam_window= bl,pol=True,lmax=Slmax_old)
        else:
            cmbi = cmbin
            
            


        smap0[0] = W2_1*cmbi[1] + W2_2*cmbi[2]
        smap0[1] = W2_1*cmbi[2] - W2_2*cmbi[1]
        smap1[0] = W1_1*cmbi[1] + W1_2*cmbi[2]
        smap1[1] = W1_1*cmbi[2] - W1_2*cmbi[1]
        smap2[0] = W0*cmbi[1]
        smap2[1] = W0*cmbi[2]
     

        alm0 = -hp.map2alm(smap0, lmax=Slmax_old,pol=False)
        #alm0 = hp.map2alm_spin(smap1, spin=0, lmax=Slmax_old)
        alm1 = hp.map2alm_spin(smap1, spin=1, lmax=Slmax_old)
        alm2 = hp.map2alm_spin(smap2, spin=2, lmax=Slmax_old)

        
        Elm = alm0[0] + 2*snl1*alm1[0] + snl2*alm2[0]
        Blm = alm0[1] + 2*snl1*alm1[1] + snl2*alm2[1]
        # Elm =   alm0[0]
        # Blm =   alm0[1]
        # Elm =   snl2*alm2[0]
        # Blm =   snl2*alm2[1]
        
        # Emap = hp.alm2map(Elm, nside= nside) 
        # Bmap = hp.alm2map(Blm, nside= nside) 
        # visu_Bmaps(Emap, outfile="Emap.png")
        # visu_Bmaps(Bmap, outfile="Bmap.png")
        
        # Emap_out = hp.ud_grade(Emap,nside_out = qml_nside)
        # Bmap_out = hp.ud_grade(Bmap,nside_out = qml_nside)

        # Emap_out = hp.alm2map(Elm, nside= nside) 
        # Bmap_out = hp.alm2map(Blm, nside= nside) 
        Elm_new=np.zeros_like(np.array(Elm)[:ALM.getsize(Slmax_new)])
        Blm_new=np.zeros_like(np.array(Blm)[:ALM.getsize(Slmax_new)])
        for l in np.arange(Slmax_new+1) :
            for m in np.arange(l+1) :
                idx_new = ALM.getidx(Slmax_new, l, m)
                idx_old = ALM.getidx(Slmax_old, l, m)
                Elm_new[idx_new]=Elm[idx_old]
                Blm_new[idx_new]=Blm[idx_old]
        Emap_out = hp.alm2map(Elm_new, nside= qml_nside, pixwin=False) 
        Bmap_out = hp.alm2map(Blm_new, nside= qml_nside, pixwin=False) 
        # Emap_out=Emap_out*qml_mask
        # Bmap_out=Bmap_out*qml_mask

        # if n==0:
        #     visu_Bmaps(smap0[0], outfile="smap0_1.png")
        #     visu_Bmaps(smap0[1], outfile="smap0_2.png")
        #     visu_Bmaps(smap1[0], outfile="smap1_1.png")
        #     visu_Bmaps(smap1[1], outfile="smap1_2.png")
        #     visu_Bmaps(smap2[0], outfile="smap2_1.png")
        #     visu_Bmaps(smap2[1], outfile="smap2_2.png")
        #     visu_Bmaps(Emap_out, outfile="Emap.png")
        #     visu_Bmaps(Bmap_out, outfile="Bmap.png")

        # alm_old=hp.map2alm(map_in,lmax=Slmax_old,pol=pol,use_weights=True,iter=0)
        # if smoothing_deg>0:
        #     hp.smoothalm(alm_old,fwhm=deg2rad(smoothing_deg))
        # mapin=hp.alm2map(alm_old, nside= 512, pixwin=False) 
        # mapin=mapin*mask
        # alm_old=hp.map2alm(mapin,lmax=Slmax_old,pol=pol,use_weights=False,iter=0)
        # alm_new=np.zeros_like(np.array(alm_old)[1,:ALM.getsize(Slmax_new)])
        # for l in np.arange(Slmax_new+1) :
        #     for m in np.arange(l+1) :
        #         idx_new = ALM.getidx(Slmax_new, l, m)
        #         idx_old = ALM.getidx(Slmax_old, l, m)
        #         if l<2:
        #             alm_new[idx_new]=alm_old[ispec,idx_old]
        #         elif l<= Slmax_new:
        #             alm_new[idx_new]=alm_old[ispec,idx_old]#*np.sqrt((l+2)*(l+1)*l*(l-1))
        
        # map_in=map_in*mask            
        # #hp.write_map("data/map_in_%d.fits"%n1,map_in,dtype=np.float64)
        # alm_old=hp.map2alm(map_in,lmax=Slmax_old,pol=pol,use_weights=True,iter=0)
        # if smoothing_deg>0:
        #     hp.smoothalm(alm_old,fwhm=deg2rad(smoothing_deg))
        # #mapin=hp.alm2map(alm_old, nside= 512, pixwin=False) 
        # #mapin=mapin*mask
        # #alm_old=hp.map2alm(mapin,lmax=Slmax_old,pol=pol,use_weights=False,iter=0)
        # alm_new=np.zeros_like(np.array(alm_old)[1,:ALM.getsize(Slmax_new)])
        # for l in np.arange(Slmax_new+1) :
        #     for m in np.arange(l+1) :
        #         idx_new = ALM.getidx(Slmax_new, l, m)
        #         idx_old = ALM.getidx(Slmax_old, l, m)
        #         if l<2:
        #             alm_new[idx_new]=alm_old[ispec,idx_old]
        #         elif l<= Slmax_new:
        #             alm_new[idx_new]=alm_old[ispec,idx_old]#*np.sqrt((l+2)*(l+1)*l*(l-1))

        #map_in=map_in*mask            
        # alm_old=hp.map2alm(map_in,lmax=Slmax_old,pol=pol,use_weights=True,iter=0)
        # # if smoothing_deg>0:
        # #     hp.smoothalm(alm_old,fwhm=deg2rad(smoothing_deg))

        # alm_new=np.zeros_like(np.array(alm_old)[:ALM.getsize(Slmax_new)])
        # for l in np.arange(Slmax_new+1) :
        #     for m in np.arange(l+1) :
        #         idx_new = ALM.getidx(Slmax_new, l, m)
        #         idx_old = ALM.getidx(Slmax_old, l, m)
        #         if l<2:
        #             alm_new[idx_new]=alm_old[idx_old]
        #         elif l<= Slmax_new:
        #             alm_new[idx_new]=alm_old[idx_old]#*np.sqrt((l+2)*(l+1)*l*(l-1))

        # map_out = hp.alm2map(alm_new, nside= qml_nside, pixwin=False) 
        #print(np.array(map_out).shape)


        # hp.write_map(output_dir+"/map_out_%d.fits"%n1,map_out,dtype=np.float64)
        # hp.mollview(map_out[1], cmap=plt.cm.jet, title='Q map output')
        # plt.savefig(output_dir+'/map_out_%d.eps'%n1,bbox_inches='tight',pad_inches=0.1)
        if output_dir != None:
            hp.write_map(output_dir+"/EBmaps/Emap_out%d.fits"%n_index,Emap_out,dtype=np.float64)
            hp.write_map(output_dir+"/EBmaps/Bmap_out%d.fits"%n_index,Bmap_out,dtype=np.float64)

    #     Emaps.append(Emap_out)
    #     Bmaps.append(Bmap_out)

    # Emaps=np.array(Emaps)
    # Bmaps=np.array(Bmaps)
    return np.array(Emap_out), np.array(Bmap_out)
    
def Trans_windowmap(mask_apo):
    """Trans windowmap to W0,W1,W2
    Parameters
    ----------    
    mask_apo: numpy.ndarray
             
        apodized window map
    Returns
    ----------
    Windowmaps:numpy.ndarray
        Return 5 maps: W0[1,npix],W1[2,npix],W2[2,npix]
    """
    if isinstance(mask_apo, (np.ndarray)):
        npix      = mask_apo.shape[-1]
        nside     = hp.npix2nside(npix)
        Slmax_old = 2*nside
        #print("nside of mask_apo = ", nside)
    else:
        print("ERROR: Only mollweide (moll) or orthographic (orth) projections available.")
        exit()

    alm_mask   = hp.map2alm(mask_apo,lmax=Slmax_old,pol=False,use_weights=True)
    der1       = hp.alm2map_der1(alm_mask,nside,lmax=Slmax_old)
    
    alm_der1_1 = hp.map2alm(der1[1],lmax=Slmax_old,pol=False,use_weights=True)
    der21      = hp.alm2map_der1(alm_der1_1,nside,lmax=Slmax_old)
    
    alm_der1_2 = hp.map2alm(der1[2],lmax=Slmax_old,pol=False,use_weights=True)
    der22      = hp.alm2map_der1(alm_der1_2,nside,lmax=Slmax_old)

    W0 = mask_apo
    
    W1_1  = -der1[1]
    W1_2  = -der1[2]

    theta,phi = hp.pix2ang(nside, np.arange(npix))
    W2_1  = -np.cos(theta)/np.sin(theta)*der1[1] + der21[1] - der22[2]
    W2_2  = -np.cos(theta)/np.sin(theta)*der1[2] + der21[2] + der22[1]
    
    return W0, W1_1, W1_2, W2_1, W2_2