import numpy as np 
from tesi_slm.my_tools import open_fits_file
import matplotlib.pyplot as plt

def get_contrast():
    fpath = "C:\\Users\\labot\\Desktop\\misure_tesi_slm\\alignment\\"
    fname = "240530fringe_due_cmm1bs013.fits"
    
    hdr, data = open_fits_file(fpath+fname)
    
    ima = data[0].data
    plt.figure()
    plt.clf()
    plt.imshow(ima, cmap='Greys_r')
    plt.colorbar(label='ADU')
    
    plt.figure()
    plt.plot(ima[200,:])
    
    Imax = 755
    Imin = 447
    
    c_meas = (Imax-Imin)/(Imax+Imin)
    
    r = 0.55
    t = 0.45
    ar = 0.005
    
    i1 = t*r
    i2 = t*r*ar 
    imax = i1+i2+2*np.sqrt(i1*i2)
    imin = i1+i2-2*np.sqrt(i1*i2)
    
    c_t = (imax-imin)/(imax+imin)
     
    return c_meas,c_t