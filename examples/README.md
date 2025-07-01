# Flowline Model Examples

This directory contains example scripts demonstrating how to use the `flowline` model package.

To run these examples, ensure you have installed the package in editable mode from the project root:
```bash
pip install -e .
```
Then, you can run any example script from the project root, for example:
```bash
python examples/example_basic_run.py
```

The scripts will generate plots and save them to the `examples/example_outputs` directory.

## Scripts

- `example_basic_run.py`: A simple, commented script showing the end-to-end process of setting up and running a single flowline simulation with temperature-precipitation forcing.

- `example_geometry_variations.py`: Demonstrates how to use different bed geometry functions (e.g., uniform slope, concave profile, variable width) and compares their effects on the glacier's evolution.

- `example_forcing_variations.py`: Shows how to configure different mass balance scenarios, including using direct mass balance with step changes or noise, and compares the glacier's response. This script also shows how to perform a spin-up run to achieve a steady state.

- `example_sweep.py`: Demonstrates how to programmatically set up and execute a parameter sweep using the `FlowlineSweep` class, without using the command-line interface. It creates a sweep configuration on the fly, runs the sweep, and plots the combined results.
