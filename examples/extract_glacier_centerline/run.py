import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from skimage import graph
from shapely.geometry import Polygon, Point, LineString
import warnings

from flowline.topo import extract_glacier_centerlines

# Example with synthetic glacier data
if __name__ == "__main__":
    # Create synthetic glacier outline (simple valley glacier)
    theta = np.linspace(0, 2 * np.pi, 100)
    outline = np.column_stack(
        [500 + 300 * np.cos(theta) + 100 * np.cos(3 * theta), 500 + 400 * np.sin(theta)]
    )

    # Create synthetic DEM (sloping surface)
    x = np.arange(0, 1000, 10)
    y = np.arange(0, 1000, 10)
    X, Y = np.meshgrid(x, y)

    # Simple elevation model: higher in the north, with some variation
    dem = 3000 + Y * 0.5 + 50 * np.sin(X / 100) * np.sin(Y / 100)

    # Extract centerlines
    
    centerlines, branches = extract_glacier_centerlines(outline, dem, resolution=10)

    print(f"Extracted {len(centerlines)} centerlines")
    print(f"Identified {len(branches)} branches")

    for i, branch in enumerate(branches):
        print(
            f"Branch {i + 1}: Order={branch['order']}, "
            f"Length={branch['length']:.1f}m, "
            f"Main={branch['is_main']}"
        )
        
        
    fig, ax = plt.subplots()
    ax.imshow(dem)
    ax.plot(centerlines)
    