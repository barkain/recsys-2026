from mcrs.crs_system import CRS_SYSTEM


def load_crs_system(**kwargs) -> CRS_SYSTEM:
    return CRS_SYSTEM(**kwargs)


# Alias for baseline compatibility
load_crs_baseline = load_crs_system
