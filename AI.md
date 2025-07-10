# README for AI Assistants

This document provides a high-level overview of the `PyFlowline` codebase to help AI assistants navigate the project, understand its architecture, and make effective changes.

## 1. Project Goal

PyFlowline is a 2D glacier flowline model based on the Shallow Ice Approximation (SIA), designed for scientific research and parameter-sweep experiments.

## 2. Core Architecture

The model is built on a modular design. A simulation is orchestrated by the `flowline2d` class, which brings together three main components:

-   **`FlowlineConfig` (`src/flowline/config.py`)**: A dataclass holding all numerical and physical parameters for a simulation (e.g., grid spacing `delx`, time stepping `delt`, ice physics `fd`).
-   **`FlowlineGeometry` (`src/flowline/geometry.py`)**: Defines the physical domain of the glacier. It handles the model grid, interpolation of bed topography (`zb`) and width (`w`), and setting the initial ice thickness (`h0`). It can be initialized from functions (e.g., `create_uniform_slope`) or from the output of a previous run (a "spin-up" profile).
-   **`MassBalanceForcing` (`src/flowline/flowline2d.py`)**: An abstract base class defining how climate drives glacier mass changes. Key implementations are:
    -   `TemperaturePrecipitationForcing`: A mass balance model driven by temperature and precipitation.
    -   `DirectMassBalanceForcing`: A simpler model where mass balance is specified directly.

The main `flowline2d` class (`src/flowline/flowline2d.py`) takes these three components, runs the simulation in a time-stepping loop (`_run_model`), solves the SIA equations (`space_loop`), and stores the results in output arrays.

## 3. Key Workflows

### A. Running a Single Simulation
This is the most basic operation. The canonical example is `examples/example_basic_run.py`.
The process is:
1.  Instantiate `FlowlineConfig` with model parameters.
2.  Instantiate `FlowlineGeometry` with bed and initial ice shape.
3.  Instantiate a `MassBalanceForcing` object (e.g., `TemperaturePrecipitationForcing`).
4.  Pass these three objects to the `flowline2d` constructor.
5.  Call `model.run()`.
6.  The result is a `flowline2d` object containing the output data, which can be saved (e.g., `result.to_xarray()`) or analyzed (`diagnostics.calc_diag(result)`).

### B. Running a Parameter Sweep
This is the primary advanced workflow, used for sensitivity analysis.
-   **Configuration**: Sweeps are defined in a YAML file (see `sweep_config.yml`). It has `base_parameters` and `sweep_parameters`.
-   **Execution**: Sweeps are launched from the command line using the `flowline-sweep` command, which is an entrypoint defined in `pyproject.toml`.
-   **Core Logic**:
    1.  The `flowline-sweep` command calls `main()` in `src/cli/run_sweep.py`.
    2.  This script uses the `FlowlineSweep` class from `src/flowline/sweep.py`, which is the high-level manager for the entire sweep process.
    3.  `FlowlineSweep` generates all parameter combinations, sets up a Dask `LocalCluster` for parallel processing, and creates a task for each simulation.
    4.  Each Dask task calls the worker function `run_flowline_simulation` in `src/flowline/entrypoints.py`.
    5.  This worker function is the bridge between a dictionary of parameters (from the YAML) and the Python model objects (`FlowlineConfig`, `FlowlineGeometry`, `MassBalanceForcing`). It runs one simulation and saves the output to a NetCDF file.
    6.  After all runs are complete, `FlowlineSweep` combines the individual NetCDF files into a single `combined_results.nc` and generates summary plots using `visualization.plot_sweep_qc`.

## 4. Codebase Map

-   `src/flowline/flowline2d.py`: **The core of the model.** Contains the `flowline2d` class, the `space_loop` SIA solver, and the currently-used forcing class definitions.
-   `src/flowline/config.py`: `FlowlineConfig` dataclass.
-   `src/flowline/geometry.py`: `FlowlineGeometry` class and helper functions to create bed shapes.
-   `src/flowline/sweep.py`: `FlowlineSweep` class, the high-level orchestrator for parameter sweeps.
-   `src/flowline/entrypoints.py`: The Dask worker function `run_flowline_simulation`.
-   `src/cli/run_sweep.py`: The `click`-based CLI script for `flowline-sweep`.
-   `src/flowline/diagnostics.py`: Post-processing functions (`calc_diag`) and physical calculations (`calc_ela`).
-   `src/flowline/visualization.py`: Plotting functions for single runs (`plot_run_qc`) and sweeps (`plot_sweep_qc`).
-   `src/flowline/analysis/core.py`: Standalone analysis functions for exploring parameter space outside of full model runs.
-   `examples/`: Example scripts showing how to use the model. `example_basic_run.py` is the key reference for single runs.

## 5. Architectural Notes & Gotchas

-   **Duplicate Forcing Classes**: The forcing classes (`MassBalanceForcing`, `TemperaturePrecipitationForcing`, etc.) are defined inside `src/flowline/flowline2d.py` and are **also** defined in `src/flowline/forcing.py`. The code currently **only uses the classes defined inside `flowline2d.py`**. The file `src/flowline/forcing.py` appears to be part of an incomplete or abandoned refactoring and is currently **unused**. Be careful to modify the correct classes inside `flowline2d.py`.
-   **Legacy `__init__`**: The `flowline2d` constructor has a legacy path (`_init_legacy`). It is designed for backward compatibility with an older, monolithic-style constructor. All new code and sweep runs use the modern constructor that accepts separate `config`, `geometry`, and `forcing` objects.
-   **Obsolete `run_sweep.py`**: The file `run_sweep.py` at the project root is obsolete and non-functional. The modern way to run sweeps is via the `flowline-sweep` command, which uses `src/cli/run_sweep.py` and `src/flowline/sweep.py`. Do not use or modify the root-level script.
-   **Dynamic Geometry Functions**: In sweeps, the geometry-creation function (e.g., `flowline.geometry.create_uniform_slope`) is specified as a string in the YAML config. The `run_flowline_simulation` entrypoint uses `import_from_string` (from `src/flowline/io.py`) to dynamically import and call this function.

## 6. Conventions

-   **Indexing**: Do not use `np.clip` to prevent out-of-bounds indexing. This practice can hide underlying bugs related to incorrect array sizes or logic errors. Array sizes and indices should be handled correctly so that clipping is not necessary.

## 7. Architectural Log

This section tracks significant architectural decisions and changes over time.

*   **2025-07-09: Serialization changed from Pickle to xarray/NetCDF.**
    The primary method for saving model results for analysis and subsequent runs is `to_xarray()`, which creates an `xarray.Dataset` that is typically saved to a NetCDF file. The `to_pickle` method on the `flowline2d` class has been removed. The standard for data interchange is now NetCDF via `xarray`.
