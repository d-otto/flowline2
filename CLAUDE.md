# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Installation
```bash
pip install -e .                    # Editable installation for development
pip install -e ".[dev]"            # Include development dependencies
```

### Running Tests
```bash
pytest                              # Run all tests
pytest tests/test_examples.py       # Run specific test file
pytest -m "not slow"               # Skip slow tests
pytest -m integration              # Run integration tests only
pytest -m numerical                # Run numerical tests only
pytest -v                          # Verbose output
```

### Running Models
```bash
# Single simulation
python examples/basic_run/run.py

# Parameter sweep (new unified approach)
python examples/parameter_sweep/run.py
python examples/parameter_sweep/run.py -o output_dir --workers 4

# Advanced parameter sweep with RNG and distributions
python examples/advanced_sweep_demo/run.py

# Spinup sweeps (legacy approach)
python examples/spinup_sweep/run.py
python examples/spinup_sweep_gamma/run.py

# Auto steady-state with FlowlineSpinup (new 4-object architecture)
python examples/auto_steady_state_demo/run.py --workers 4
```

### Development Workflow
- Use `pytest` for testing - no specific linting/formatting tools configured
- Test with small parameter sets first, then scale up
- Check QC plots in output directories for validation
- When running one of the example scripts, make sure the contents of the `output` folder are cleared ahead of execution.
- Examples are run as part of the pytest suite. When you want to test/rerun an example, you should do it by using pytest.

## Architecture Overview

### Core Architecture Design
PyFlowline uses a modular architecture with clean object separation:

#### Three-Component Design (Basic)
1. **Configuration** (`FlowlineConfig`) - Physical and numerical parameters
2. **Geometry** (`FlowlineGeometry`) - Spatial setup and bed topology  
3. **Forcing** (`MassBalanceForcing`) - Climate and mass balance

#### Four-Component Design (Advanced with Auto Steady-State)
For auto-generating steady-state profiles and response testing:
4. **Spinup** (`FlowlineSpinup`) - Steady-state generation and perturbations

Each run has Config, Geometry, Forcing, and Spinup objects providing complete specification.

### Key Architectural Patterns

#### Parameter Sweep Architecture
The advanced workflow uses `FlowlineSweep` with:
- **Direct object passing** - Pass `FlowlineConfig`, `FlowlineGeometry`, and `MassBalanceForcing` objects directly
- **Unified run scripts** - Each experiment is a single Python file with config, execution, and post-processing
- **Dask parallel processing** for distributed execution
- **CLI utilities** - Shared argument parsing via `flowline.cli.utils`
- **Full Python power** - Complex parameter generation, RNG objects, statistical distributions, custom geometries

#### Spinup System
Two approaches for steady-state initialization:

**Legacy Approach** (mode-based):
- **Separate spinup runs** executed before main simulations
- **Grid consistency requirements** between spinup and main runs  
- **State transfer** from spinup to main run initial conditions
- **Automatic result validation** and error handling

**New Approach** (`FlowlineSpinup` objects):
- **Auto-generated steady-state profiles** for each parameter set
- **Target matching** to achieve comparable glacier states
- **Lambda-based perturbations** for response testing
- **4-object architecture** (Config, Geometry, Forcing, Spinup)

#### Data Flow Patterns
- **NetCDF-based I/O** (not pickle) for all persistent data
- **xarray Datasets** for labeled, multi-dimensional arrays
- **Provenance tracking** with git hashes and environment snapshots
- **Chunked processing** for large parameter sweeps

### Important Implementation Details

#### Parameter Sweep Pattern (New)
**Modern approach** - Unified config+run scripts:
```python
# Create objects directly in the run script
base_config = FlowlineConfig(ts=0, tf=100, delx=25, ...)
base_geometry = FlowlineGeometry(x_gr, zb_gr, w_geom, ...)
base_forcing = TemperaturePrecipitationForcing(ts=0, tf=100, ...)

# Define sweep parameters
sweep_parameters = {'forcing.T0': [7.0, 8.0, 9.0]}

# Run sweep
sweep = FlowlineSweep(
    base_config=base_config,
    base_geometry=base_geometry, 
    base_forcing=base_forcing,
    sweep_parameters=sweep_parameters
)
sweep.run()
```

#### Auto Steady-State Pattern (New)
**FlowlineSpinup approach** - 4-object architecture:
```python
from src.flowline.spinup import FlowlineSpinup

# Create FlowlineSpinup objects for each parameter set
spinup_objects = {}
for mu in [0.5, 0.6, 0.7]:
    run_id = f"run_{len(spinup_objects):04d}"
    
    spinup_config = FlowlineConfig(tf=1000, ...)
    spinup_forcing = TemperaturePrecipitationForcing(mu=mu, T0=8.0, ...)
    
    spinup_obj = FlowlineSpinup(
        config=spinup_config,
        geometry=base_geometry,
        forcing=spinup_forcing,
        target_matching={
            'target_length': 8000,              # Target glacier length
            'adjustment_parameter': 'forcing.T0',
            'adjustment_function': lambda mu: 8.0 + (mu-0.6)*3.0
        }
    )
    spinup_objects[run_id] = spinup_obj

# Option 1: Dict of spinup objects (different spinups per run)
spinup_objects = {
    'run_0000': spinup_obj1,
    'run_0001': spinup_obj2
}

# Option 2: Single shared spinup (same spinup for all runs)
# spinup_objects = shared_spinup_obj

# Create experimental perturbations (separate from spinup)
experimental_perturbations = {
    'run_0000': {'forcing.T0': lambda T0: T0 + 1.0},  # +1°C warming
    'run_0001': {'forcing.T0': lambda T0: T0 + 2.0}   # +2°C warming
}

# Run sweep with 4-object architecture
sweep = FlowlineSweep(
    base_config=response_config,
    base_geometry=base_geometry,
    base_forcing=response_forcing,
    sweep_parameters={},  # No additional parameter sweeps
    spinup_objects=spinup_objects,  # Flexible: single object or dict
    experimental_perturbations=experimental_perturbations  # Experimental changes
)
sweep.run()
```

#### Forcing Classes
Forcing classes are defined in `src/flowline/forcing.py` and imported into `src/flowline/flowline2d.py`:
- `TemperaturePrecipitationForcing` - Climate-based mass balance with temperature and precipitation
- `DirectMassBalanceForcing` - Direct mass balance specification

These classes are instantiated directly in sweep scripts (no `mode` parameter needed).

#### Constructor Patterns
The `flowline2d` constructor has **legacy compatibility paths**:
- Modern usage: Pass `FlowlineConfig`, `FlowlineGeometry`, `MassBalanceForcing` objects
- Legacy usage: Pass individual parameters (automatically wrapped in config object)

#### Grid Management
- **Spatial grids must be consistent** between all components
- **Interpolation happens in geometry setup** from high-res to model grid
- **Gradient calculations** are cached in geometry object

### Critical Development Gotchas

1. **Test isolation**: Some tests create files in the test directory - clean up between runs
2. **Parameter validation**: Config validation happens in `__post_init__`, not at assignment
3. **Spinup grid consistency**: Spinup and main runs must use identical spatial grids
4. **Dynamic loading**: Geometry function names in YAML must match actual function names
5. **NetCDF chunking**: Large sweeps may need custom chunking strategies

### Configuration Patterns

#### Unified Run Script Structure
```python
# Standard pattern for parameter sweep scripts
from src.flowline.sweep import FlowlineSweep
from src.flowline.cli.utils import parse_sweep_cli_args, get_sweep_cli_kwargs
from src.flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from src.flowline.geometry import FlowlineGeometry
import src.flowline.geometry as geometry_module

def main():
    # Parse CLI arguments
    args = parse_sweep_cli_args("Description of this sweep")
    
    # Create base objects
    base_config = FlowlineConfig(...)
    base_geometry = FlowlineGeometry(...)
    base_forcing = TemperaturePrecipitationForcing(ts=0, tf=100, P0=2.0, T0=8.0, ...)  # No 'mode' parameter needed
    
    # Define sweep parameters
    sweep_parameters = {'forcing.T0': [7.0, 8.0, 9.0]}
    
    # Run sweep
    sweep = FlowlineSweep(
        base_config=base_config,
        base_geometry=base_geometry,
        base_forcing=base_forcing,
        sweep_parameters=sweep_parameters,
        **get_sweep_cli_kwargs(args)
    )
    sweep.run()
    
    # Custom post-processing here
    
if __name__ == "__main__":
    main()
```

#### Advanced Parameter Generation
```python
# Example: Using RNG for reproducible random parameters
import numpy as np
from scipy import stats

rng = np.random.RandomState(42)
temp_values = stats.norm(loc=8.0, scale=1.0).rvs(10, random_state=rng)
sweep_parameters = {'forcing.T0': temp_values.tolist()}
```

### Testing Structure
- **Unit tests**: Component-level testing
- **Integration tests**: Full workflow testing  
- **Numerical tests**: Scientific validation
- **Slow tests**: Long-running parameter sweeps (marked with `@pytest.mark.slow`)
- **Sweep tests**: Test object creation and basic sweep functionality

### Common Workflows

#### Creating a New Parameter Sweep
**Basic Sweep:**
1. Copy an existing example (e.g., `examples/parameter_sweep/run.py`)
2. Modify base objects (`FlowlineConfig`, `FlowlineGeometry`, `MassBalanceForcing`)
3. Define `sweep_parameters` dictionary with parameter paths and values
4. Add custom post-processing and visualization
5. Test with small parameter sets first

**Auto Steady-State Sweep:**
1. Copy `examples/auto_steady_state_demo/run.py` as template
2. Create `FlowlineSpinup` objects for each parameter set
3. Configure target matching for comparable initial states
4. Define perturbations using lambda functions
5. Use `spinup_objects` parameter in `FlowlineSweep`

#### Adding New Forcing
1. Create new class inheriting from `MassBalanceForcing` in `flowline2d.py`
2. Implement `get_mass_balance()` and `get_climate_vars()` methods
3. Add to forcing selection logic in `flowline2d` constructor
4. Add configuration parameters to `FlowlineConfig` if needed
5. Update sweep scripts to use new forcing class

#### Adding New Geometry
1. Create function in `geometry.py` following naming pattern `create_*`
2. Function should return `(x_gr, zb_gr, w_geom)` tuple
3. Test with both single runs and parameter sweeps
4. Use in sweep scripts via `geometry_module.create_*()` calls

#### Advanced Parameter Generation
1. Use numpy RNG objects for reproducible randomness
2. Generate parameters from statistical distributions
3. Create complex geometries programmatically
4. Pass objects with state (e.g., RNG seeds) to forcing classes

#### Using FlowlineSpinup for Auto Steady-State
1. Create `FlowlineSpinup` objects with 3-object architecture (Config, Geometry, Forcing)
2. Configure target matching for comparable glacier lengths across parameter sets
3. Define experimental perturbations separately using `experimental_perturbations` parameter
4. Use lambda functions for relative perturbations (`lambda T0: T0 + 1.0`)
5. Flexible spinup specification: single object (shared) or dict (per-run)
6. Automatic run count inference from `spinup_objects` or `experimental_perturbations`
7. Clean separation: `FlowlineSpinup` for steady-state, `FlowlineSweep` for experiments

#### Debugging Parameter Sweeps
1. Check individual NetCDF files in output directory
2. Use QC plots for visual validation
3. Check Dask dashboard for parallel processing issues
4. Verify parameter hashes for reproducibility
5. Test object creation before running full sweep

### Development Best Practices
- Always use deltout = 1

### Simulation Recommendations
- When running a simulation from the command line, you should specify the number of dask workers as either 4 or 8. If the simulation is small, use 4. Otherwise, use 8. Never use less than 4 workers.

### Critical Development Guidelines
- **NEVER hard code default values, except for when they are the default in a function signature. Hardcoded default values are a SILENT FAILURE, which should ALWAYS be avoided. Let the model throw an error if something goes wrong!**

## Memories
- You should NOT maintain backwards compatibility unless by default. You should ask if it seems necessary.
- Do NOT follow the pattern of providing default arguments in the model. If a required argument is missing, let the function fail. Do NOT create any silent fail-states, or scenarios where the behavior would be unexpected.
- When plotting subplots with matplotlib, use plot mosaic.
- If you need to look inside a model results file that is too large, you should use xarray functions to print & analyze its structure.
- When you run scripts from the command line that run the model, you should use the flag to disable output of the progress bar.
- To avoid race conditions with dask, you should use fig.savefig() and plt.close(fig).
- Don't put info about claude code in commit messages.