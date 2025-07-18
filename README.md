# PyFlowline: A 2D Glacier Flowline Model

PyFlowline is a scientific modeling tool for simulating the dynamics of glacier flowlines in two dimensions. It is built on the Shallow Ice Approximation (SIA) and is designed with a modular architecture to facilitate research and experimentation with different glacier geometries, climate forcings, and model parameters.

## Key Features

- **Shallow Ice Approximation (SIA)**: The model core solves the SIA equations for ice flow, which is suitable for many valley glaciers.
- **Modular Design**: The model is structured into distinct components for `Configuration`, `Geometry`, and `Forcing`, allowing for easy extension and modification.
- **Auto Steady-State**: Advanced `FlowlineSpinup` system for auto-generating steady-state profiles with target matching and perturbation testing.
- **Flexible Forcing**: Supports multiple mass balance models, including temperature-precipitation-based schemes and direct mass balance inputs.
- **Parameter Sweeps**: Integrated with Dask to perform parallelized parameter sweeps efficiently across multiple CPU cores. This is ideal for sensitivity studies and model calibration.
- **Standardized Outputs**: Simulation results are saved in NetCDF format (`.nc`), complete with metadata for reproducibility. QC plots are automatically generated for single runs and sweeps.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/pyflowline.git
    cd pyflowline
    ```

2.  Install the package in editable mode. This allows you to run the scripts from the project root and have your changes to the source code immediately reflected.
    ```bash
    pip install -e .
    ```

### Dependencies

The model requires the following Python packages. Installing the project with `pip` will handle these automatically.

- `click`
- `dask[distributed]`
- `dill`
- `matplotlib`
- `numba`
- `numpy`
- `pandas`
- `pyyaml`
- `scipy`
- `seaborn`
- `tqdm`
- `xarray`

## Usage

### Running a Single Simulation

A simple example is provided in `examples/example_basic_run.py`. This script demonstrates how to configure and run a single glacier simulation from start to finish.

To run it, execute the following command from the project root:
```bash
python examples/example_basic_run.py
```

This will produce two output files in the `examples/example_outputs/` directory:
- `basic_run_result.nc`: A NetCDF file containing the full simulation results.
- `basic_run_qc.png`: A QC (Quality Control) plot summarizing the glacier's evolution.

### Running a Parameter Sweep

The model supports two approaches for parameter sweeps:

#### Basic Parameter Sweep
Traditional sweeps are configured using a YAML file and executed via the command-line interface.

1.  **Configure the sweep**: An example is provided in `sweep_config.yml`. Edit it to define the base parameters and the sweep parameters. The script will generate a run for every possible combination of the sweep parameter values.

    ```yaml
    # sweep_config.yml
    base_parameters:
      config:
        tf: 500
        # ... other base parameters
      geometry:
        # ... geometry setup
      forcing:
        P0: 2.0

    sweep_parameters:
      # Sweep over a range of temperatures
      forcing.T0: [7.0, 7.5, 8.0, 8.5, 9.0]
      # And/or other parameters
      forcing.mu: [0.5, 0.6]
    ```

2.  **Execute the sweep**: Run the sweep from the project root using the `flowline-sweep` command. The results will be saved in a new, timestamped directory (e.g., `sweep_output_1672531200`).

    ```bash
    flowline-sweep sweep_config.yml
    ```
    This command will:
    - Create an output directory for the sweep.
    - Save reproducibility information (git hash, environment, config file).
    - Run all simulation combinations in parallel using Dask.
    - Combine the individual NetCDF results into a single `combined_results.nc` file.
    - Generate QC plots summarizing the sweep results (e.g., `sweep_qc_length.png`).

    You can specify a custom output directory and number of workers:
    ```bash
    flowline-sweep sweep_config.yml -o my_sweep_results --workers 4
    ```

#### Auto Steady-State Sweep
For advanced experiments requiring auto-generated steady-state profiles with target matching:

```bash
python examples/auto_steady_state_demo/run.py --workers 4
```

This approach:
- Generates steady-state profiles for each parameter set
- Applies target matching to achieve comparable glacier lengths
- Tests response to climate perturbations (e.g., +1°C warming)
- Uses the new 4-object architecture (Config, Geometry, Forcing, Spinup)

See `examples/auto_steady_state_demo/run.py` for a complete example using `FlowlineSpinup` objects.

## Project Structure

- `src/flowline/`: The core source code for the flowline model.
    - `flowline2d.py`: The main model class and SIA solver.
    - `config.py`, `geometry.py`, `forcing.py`: Modules defining the modular components.
    - `sweep.py`: Logic for managing and executing parameter sweeps.
    - `spinup.py`: FlowlineSpinup class for auto steady-state generation.
    - `entrypoints.py`: Worker function for running simulations.
    - `visualization.py`: Plotting functions.
- `src/cli/`: Command-line interfaces for running parts of the model (e.g., sweeps).
- `examples/`: Example scripts and configurations demonstrating model usage.
- `tests/`: Unit and integration tests for the model.
- `projects/`: A directory for specific research projects or case studies using the model.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
