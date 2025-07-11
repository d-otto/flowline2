import pytest
import shutil
from pathlib import Path


@pytest.fixture(autouse=True)
def clear_example_output_folders():
    """
    Automatically clear output folders in example directories before each test.
    This fixture runs before every test to ensure clean state.
    """
    # Find all example directories
    example_dir = Path(__file__).resolve().parent.parent / 'examples'
    
    if example_dir.exists():
        # Clear output folders in all example directories
        for example_subdir in example_dir.iterdir():
            if example_subdir.is_dir() and example_subdir.name != '__pycache__':
                output_dir = example_subdir / 'output'
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                    output_dir.mkdir(exist_ok=True)