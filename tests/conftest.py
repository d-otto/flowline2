import pytest
import shutil
from pathlib import Path


@pytest.fixture(autouse=True)
def clear_example_output_folders(request):
    """
    Automatically clear the output folder for a specific example script test.
    This fixture identifies tests parameterized with `script_path` (like those
    in test_examples.py) and clears the 'output' directory in that script's
    parent folder before the test runs.
    """
    if hasattr(request.node, "callspec"):
        if 'script_path' in request.node.callspec.params:
            script_path = request.node.callspec.params['script_path']
            output_dir = script_path.parent / 'output'
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(exist_ok=True)
