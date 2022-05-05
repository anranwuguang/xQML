import numpy as np
import healpy as hp

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
    #hp.write_map("qml_mask.fits",mask_out,dtype=bool)
    mask_out=np.asarray(mask_out,bool)
    return mask_out

def degrade_maps(qml_nside, maps_in, pol, qml_mask=()):
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
    if len(maps_in.shape) ==2:
        nsimu=1
        maps_in=np.array([maps_in])
    else:
        nsimu=maps_in.shape[0]
    nside=hp.npix2nside(maps_in.shape[-1])
    Slmax_old=3*nside
    Slmax_new=3*qml_nside

    npix= hp.nside2npix(qml_nside)
    if qml_mask == ():
        qml_mask=np.ones(npix,bool)
    else:
        if npix != len(qml_mask):
            print("The nside of qml_mask inconsistent with qml_nside.")

    ALM = hp.Alm
    #maps=[]
    for n in np.arange(nsimu):
        n1=n+1
        #map_out = hp.read_map(maps_dir+"%d.fits"%n1,field=(0,1,2),dtype=float,verbose=0)
        #map_in = hp.read_map(maps_dir+"%d.fits"%n1,field=(0,1,2),dtype=np.float64,verbose=0)
        map_in=maps_in[n]
        
        alm_old=hp.map2alm(map_in,lmax=Slmax_old,pol=pol,use_weights=True,iter=3)
        alm_new=np.zeros_like(np.array(alm_old)[:,:ALM.getsize(Slmax_new)])
        for l in np.arange(Slmax_new+1) :
            for m in np.arange(l+1) :
                idx_new = ALM.getidx(Slmax_new, l, m)
                idx_old = ALM.getidx(Slmax_old, l, m)
                if l<=qml_nside:
                    alm_new[:,idx_new]=alm_old[:,idx_old]
                elif l<=Slmax_new:
                    alm_new[:,idx_new]=alm_old[:,idx_old]*0.5*(1+np.sin(l*np.pi/2/qml_nside))
        map_out = hp.alm2map(alm_new, nside= qml_nside, pixwin=False) 
        map_out = map_out*qml_mask


        # hp.write_map(output_dir+"/map_out_%d.fits"%n1,map_out,dtype=np.float64)
        # hp.mollview(map_out[1], cmap=plt.cm.jet, title='Q map output')
        # plt.savefig(output_dir+'/map_out_%d.eps'%n1,bbox_inches='tight',pad_inches=0.1)
        
        #maps.append(map_out)


    #maps=np.array(maps)
    return map_out

def get_maps_covariance (maps, istokes, qml_mask):
    """get maps pixel covariance
    Parameters
    ----------
    nsimu：integer
        mumber of samples
        
    maps：
        degraded maps

    mode:

    targets:

    Returns
    ----------
    map:
        Return a degrade map with Nside = qml_nside
    """
    print()
    print("get pixel covariance of maps.")
    nsimu = maps.shape[0]
    masked_maps=maps[:,istokes][:,:, qml_mask]
    print("nsimu = %d"%nsimu)
    print("maps.shape = ",maps.shape)
    print("masked_maps.shape = ",masked_maps.shape)
    reshaped_masked_maps = masked_maps.reshape(masked_maps.shape[0],masked_maps.shape[1]*masked_maps.shape[2])
    print("reshaped_masked_maps.shape = ", reshaped_masked_maps.shape)
    rank=reshaped_masked_maps.shape[1]
    NoiseVar=np.cov(np.array(reshaped_masked_maps).T)

    for i in np.arange(rank):
         for j in np.arange(rank):
            if not(j-1 < i< j+1) :
                NoiseVar[i,j]=0
               

    print("NoiseVar.shape = ", NoiseVar.shape)
    
    return NoiseVar

    

print("import tools successed! \n")
