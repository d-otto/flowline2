from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
GDATA_DIR = Path(__file__).parent.parent.parent.parent / Path('glacier_data')

# Custom exceptions
class FlowlineModelError(Exception):
    """Base exception for flowline model errors"""
    pass

class GeometryError(FlowlineModelError):
    """Errors related to geometry setup"""
    pass

class NumericalInstabilityError(FlowlineModelError):
    """Errors from numerical instabilities"""
    pass
