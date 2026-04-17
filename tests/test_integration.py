import xarray as xr
import pytest

from flowline.sweep import FlowlineSweep
from flowline.flowline2d import FlowlineConfig, TemperaturePrecipitationForcing
from flowline.geometry import FlowlineGeometry, create_uniform_slope

def test_sweep_integration(tmp_path):
    """
    Integration test that runs a small sweep using modern object-oriented approach.
    """
    output_dir = tmp_path / "test_sweep_output"
    output_dir.mkdir(exist_ok=True)

    # Create base objects using modern approach
    base_config = FlowlineConfig(
        ts=0,
        tf=10,  # Short run time for testing
        delx=100,  # Coarser grid for speed
        delt=0.0125/2,  # Larger time step
        deltout=5.0,
        min_thick=1.0,
        mu=0.65
    )
    
    # Create geometry
    x_gr, zb_gr, w_geom = create_uniform_slope(
        bed_characteristic_length=10000,
        domain_extent=12000,
        x_gr_points=61,
        width=1000,
        elevation_drop=1000
    )
    
    # Create reasonable initial ice thickness profile
    scale = 100
    length = 5000
    h_init = [(scale * (1 - x / length)) if x < length else 0 for x in x_gr]
    h_init = [max(0, h) for h in h_init]
    
    base_geometry = FlowlineGeometry(
        x_gr=x_gr,
        zb_gr=zb_gr, 
        w_geom=w_geom,
        x_init=x_gr,
        h_init=h_init
    )
    
    # Create forcing
    base_forcing = TemperaturePrecipitationForcing(
        ts=0,
        tf=10,
        T0=8.0,
        P0=2.0,
        gamma=6.5e-3,
        mu=0.65
    )
    
    # Define sweep parameters - same as in test_sweep_config.yml
    sweep_parameters = {
        'config.mu': [0.65, 0.70]  # 2 runs
    }
    
    # Run the sweep
    sweep = FlowlineSweep(
        base_config=base_config,
        base_geometry=base_geometry,
        base_forcing=base_forcing,
        sweep_parameters=sweep_parameters,
        output_dir=str(output_dir),
        workers=2
    )
    
    sweep.run()

    # Check outputs
    assert output_dir.exists()
    
    # The sweep has 2 runs in it
    run_files = list(output_dir.glob("run_*.nc"))
    assert len(run_files) == 2, f"Expected 2 individual run output files, but found {len(run_files)}"
    
    # Check for combined results file
    combined_file = output_dir / "combined_results.nc"
    assert combined_file.exists(), "Combined results file was not created"
    
    # Check contents of the combined file
    with xr.open_dataset(combined_file) as ds:
        assert 'config_mu' in ds.coords
        assert len(ds['config_mu']) == 2
        # Use tolist() for clean comparison
        assert sorted(ds['config_mu'].values.tolist()) == [0.65, 0.70]
        assert 'h' in ds.data_vars
        # Check that the swept dimension is the first dimension
        assert ds['h'].dims[0] == 'config_mu'
        assert ds['h'].shape[0] == 2
    
    # Check for run info files
    assert (output_dir / "run_info.txt").exists()
    assert (output_dir / "requirements.txt").exists()
