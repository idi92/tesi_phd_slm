

def startup():
    LB_ROOT = '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/.shortcut-targets-by-id/1SPpwbxlHyuuXmzaajup9lXpg_qSHjBX4/phd_slm_edo/misure_tesi_slm/bronte'
    EB_ROOT = 'sdf'

    from tesi_slm.bronte.package_data import set_data_root_dir
    set_data_root_dir(LB_ROOT)

    from tesi_slm.bronte import factory
    return factory.BronteFactory()
