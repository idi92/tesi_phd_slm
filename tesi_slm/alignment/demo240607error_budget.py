import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

def WFC_error(fname, max_stroke = 635e-9):
    
    bmp_image = open_8bitmap(fname)
    gray_map = np.array(bmp_image, dtype=np.uint8)
    wrapped_wf_map = max_stroke/256 * gray_map
    return wrapped_wf_map

def open_8bitmap(fname_bitmap):
    bmp_image = Image.open(fname_bitmap)
    gray_map = np.array(bmp_image, dtype=np.uint8)
    return gray_map

def unwrap_wf(wrapped_wf, max_stroke = 635e-9):
    
    dstroke = max_stroke/256
    jump = max_stroke - dstroke
    
    unwrapped_wf = np.unwrap(wrapped_wf, axis = 1, discont = jump, period = max_stroke)
    
    plt.figure()
    plt.clf()
    plt.imshow(unwrapped_wf/1e-6, cmap = 'jet')
    plt.colorbar(label = 'um')
    plt.title("%g m rms"%unwrapped_wf.std())
    