import subprocess
import sys
from pathlib import Path
import xarray as xr
import pytest

def test_sweep_integration(tmp_path):
    """
    An integration test that runs a small sweep and checks the output.
    """
    # 1. Setup paths relative to this test file
    project_root = Path(__file__).parent.parent
    test_config_path = project_root / "tests/test_sweep_config.yml"
    output_dir = tmp_path / "test_sweep_output"
    # The entry point is now a module within the `cli` package
    cli_script_module = "cli.run_sweep"

    # 2. Run the sweep as a subprocess using the new click-based CLI
    cmd = [
        sys.executable, "-m", cli_script_module,
        str(test_config_path),
        "--output-dir", str(output_dir),
        "--workers", "2"
    ]
    
    # Running from project root; pytest is configured with pythonpath="src"
    # so the `cli` module should be found.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=project_root)
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout, file=sys.stdout)
        print("STDERR:", result.stderr, file=sys.stderr)
        pytest.fail(f"CLI script failed with exit code {result.returncode}", pytrace=False)

    # 3. Check outputs
    assert output_dir.exists()
    
    # The config has 2 runs in it.
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
    assert (output_dir / "config.yml").exists()
    assert (output_dir / "run_info.txt").exists()
    assert (output_dir / "requirements.txt").exists()
