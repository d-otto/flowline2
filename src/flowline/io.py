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

def generate_run_params(config):
    """Generate individual run parameter dictionaries from a sweep config."""
    base_params = config.get('base_parameters', {})
    sweep_params = config.get('sweep_parameters', {})

    if not sweep_params:
        yield base_params
        return

    sweep_keys = list(sweep_params.keys())
    sweep_value_lists = [sweep_params[key] for key in sweep_keys]

    # Create all combinations of sweep values
    for combination in itertools.product(*sweep_value_lists):
        run_params = deepcopy(base_params)
        for i, key in enumerate(sweep_keys):
            # Nested parameter update (e.g., config.mu or geometry.parameters.amplitude)
            parts = key.split('.')
            d = run_params
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = combination[i]
        
        yield run_params
