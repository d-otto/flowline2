import pytest
from pathlib import Path
import subprocess
import sys

# Discover example scripts to be tested.
# We only include scripts for which content was provided.
EXAMPLE_DIR = Path(__file__).resolve().parent.parent
example_scripts = [
    EXAMPLE_DIR / 'examples/example_spinup_sweep.py',
    EXAMPLE_DIR / 'examples/example_sweep.py'
]

# Filter out scripts that might not exist to prevent errors.
example_scripts = [p for p in example_scripts if p.exists()]

# Create readable IDs for the tests.
ids = [p.name for p in example_scripts]

@pytest.mark.parametrize("script_path", example_scripts, ids=ids)
def test_example_script_runs_successfully(script_path, tmp_path):
    """
    Runs an example script as a subprocess to ensure it executes without errors.
    Outputs are redirected to a temporary directory provided by pytest.
    """
    # Use a unique subdirectory for each parametrised test case.
    output_dir = tmp_path / script_path.stem
    output_dir.mkdir()

    command = [
        sys.executable,
        str(script_path),
        '--output-dir',
        str(output_dir)
    ]

    # Execute the script as a subprocess.
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False  # We check the return code manually for better error reporting.
    )

    # Assert that the script ran successfully (exit code 0).
    assert result.returncode == 0, (
        f"Script {script_path.name} failed with exit code {result.returncode}.\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}"
    )
