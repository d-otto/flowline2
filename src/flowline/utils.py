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


# Object comparison utilities
def objects_equal(obj1, obj2):
    """Check if two configuration objects are equal."""
    if type(obj1) != type(obj2):
        return False
    
    if hasattr(obj1, '__dict__') and hasattr(obj2, '__dict__'):
        return obj1.__dict__ == obj2.__dict__
    
    return obj1 == obj2


def object_hash(obj):
    """Generate hash for configuration objects."""
    if hasattr(obj, '__dict__'):
        return hash(str(sorted(obj.__dict__.items())))
    return hash(str(obj))
