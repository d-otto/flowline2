import itertools
import importlib
from copy import deepcopy

def import_from_string(path_string):
    """Import a function or class from a string path."""
    module_path, obj_name = path_string.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, obj_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import '{obj_name}' from '{module_path}': {e}")

