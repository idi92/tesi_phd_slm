import numpy as np

def rect(x):
    return np.where(abs(x) <= 0.5, 1, 0)


def main(phi = np.pi*2):
    
    D = 9.2e-3
    d = 9.2e-6
    a = 9e-6
    Npixel = int(D/d)
    Nstep = 4
    
    Npt = 100000
    xi = np.linspace(0, D, Npt)
    
    phase = np.zeros(Npt)
    
    for idx, x in enumerate(xi):
        pixel_term = 0
        step_term = 0
    
        for n in range(Npixel):
            pixel_term += rect((x-(n+0.5)*d)/a)
            
        for k in range(Nstep):
            step_term += rect((x-(k+0.5)*D/Nstep)/(D/Nstep))*k*phi/Nstep
        
        phase[idx] = pixel_term * step_term
        
    return xi,phase 