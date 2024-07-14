from astropy.io import fits
import numpy as np


def save(fname, dataCube, header_dict={}, vector_dict={}):
    """
    Save dataCube and associated metadata into a FITS file.

    Parameters:
    fname (str): Filename of the FITS file to which the data will be saved.
    dataCube (numpy.ndarray or numpy.ma.MaskedArray): Main data cube to be saved.
        If dataCube is a MaskedArray, it will be saved with separate HDUs for data and mask.
    header_dict (dict, optional): Dictionary containing header keywords and their values
        to be saved in the primary HDU header. Default is an empty dictionary ({}).
    vector_dict (dict, optional): Dictionary where keys are HDU names and values are 1D arrays
        to be saved as separate HDUs. Default is an empty dictionary ({}).

    Returns:
    None

    Notes:
    - If dataCube is a MaskedArray, it will be saved with separate HDUs for data ('DATA') and mask ('MASK').
    - Additional 1D arrays in vector_dict are saved as separate HDUs with their respective keys as names.

    """

    # Saving par_dict in the header
    primary_hdu = fits.PrimaryHDU()
    # Add header keywords and values from header_dict to the primary header
    for key, value in header_dict.items():
        primary_hdu.header[key] = value

    # Save masked array data and mask as separate HDUs if dataCube is a masked
    # array
    if np.ma.is_masked(dataCube):
        data_hdu = fits.ImageHDU(dataCube.data, name='DATA')
        mask_hdu = fits.ImageHDU(dataCube.mask.astype(np.uint8), name='MASK')
        hdul = fits.HDUList([primary_hdu, data_hdu, mask_hdu])

    else:
        # Save non-masked array data
        data_hdu = fits.ImageHDU(dataCube, name='DATA')
        hdul = fits.HDUList([primary_hdu, data_hdu])

    # Save additional 1D arrays from vector_dict as separate HDUs
    for key, array in vector_dict.items():
        hdul.append(fits.ImageHDU(array, name=key))

    try:
        hdul.writeto(fname, overwrite=True)
    finally:
        hdul.close()
        hdul = None  # Ensure the FITS file is properly closed


def read_info(fname):
    """
    Display information about a FITS file, including its structure and primary header.

    Parameters:
    fname (str): Filename of the FITS file to read and display information from.

    Returns:
    None. Prints file structure and primary header information to the console.

    """

    with fits.open(fname) as hdul:
        print("\nFile INFO:\n")
        hdul.info()
        print("\nPrimary Header:\n")
        print(repr(hdul[0].header))


def load(fname, displayInfo=False):
    """
    Load data and associated metadata from a FITS file.

    Parameters:
    fname (str): Filename of the FITS file to load.
    displayInfo (bool, optional): If True, display information about the FITS file.
        Default is False.

    Returns:
    dataCube (numpy.ndarray or numpy.ma.MaskedArray): Main data cube loaded from the 'DATA' HDU.
        If 'MASK' HDU exists, dataCube is a MaskedArray; otherwise, it's a regular ndarray.
    header_dict (dict): Dictionary containing header keywords and their values from the primary HDU header.
    vector_dict (dict): Dictionary where keys are HDU names and values are 1D arrays (numpy.ndarray)
        loaded from additional HDUs.

    """

    with fits.open(fname, memmap=False) as hdul:

        if displayInfo is not False:
            print("\nFile INFO:\n")
            hdul.info()
            print("\nPrimary Header:\n")
            print(repr(hdul[0].header))

        # Access the primary HDU and read parameters from the header
        primary_header = hdul[0].header
        header_dict = {key: primary_header[key] for key in primary_header[4:]}

        # Check if 'MASK' HDU exists
        mask_hdu_exists = 'MASK' in hdul

        if mask_hdu_exists:

            # Access the masked array data and mask
            data_from_fits = hdul['DATA'].data
            mask_from_fits = hdul['MASK'].data.astype(bool)
            dataCube = np.ma.masked_array(data_from_fits, mask=mask_from_fits)
            idx = 3  # Starting index for additional HDUs

        else:
            # Access the non-masked array data
            data_from_fits = hdul['DATA'].data
            dataCube = np.array(data_from_fits)
            idx = 2  # Starting index for additional HDUs

        # Access the additional 1D arrays from HDUs
        vector_dict = {}
        for hdu in hdul[idx:]:  # Iterate over additional HDUs
            vector_dict[hdu.name] = hdu.data

    hdul.close()
    hdul = None

    return dataCube, header_dict, vector_dict
