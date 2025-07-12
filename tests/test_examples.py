import pytest
from pathlib import Path
import subprocess
import sys
import glob

# Discover all example run scripts.
EXAMPLE_DIR = Path(__file__).resolve().parent.parent / 'examples'
example_scripts = sorted([Path(p) for p in glob.glob(str(EXAMPLE_DIR / '*/run.py'))])

# Create readable IDs for the tests from their parent directory names.
ids = [p.relative_to(EXAMPLE_DIR).parent.name for p in example_scripts]

@pytest.mark.parametrize("script_path", example_scripts, ids=ids)
def test_example_script_runs_successfully(script_path):
    """
    Runs an example script as a subprocess to ensure it executes without errors.
    Outputs are redirected to a temporary directory provided by pytest.
    """
    command = [
        sys.executable,
        str(script_path),
    ]

    # The example scripts now handle their own default config paths, so no
    # special logic is needed here to provide a --config argument.

    # Execute the script as a subprocess.
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False  # We check the return code manually for better error reporting.
    )

    # Assert that the script ran successfully (exit code 0).
    assert result.returncode == 0, (
        f"Script '{script_path.relative_to(EXAMPLE_DIR)}' failed with exit code {result.returncode}.\n"
        f"--- COMMAND ---\n{' '.join(command)}\n"
        f"--- STDOUT ---\n{result.stdout}\n"
        f"--- STDERR ---\n{result.stderr}"
    )
