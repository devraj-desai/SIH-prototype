"""
map_utils.py
------------
Frontend-only helpers that turn a 2D temperature field into a smooth,
semi-transparent PNG overlay draped over a real coastline basemap
(windy.com-style filled contours), plus a matching colorbar legend.

Nothing in this file talks to the model — it only handles visual styling
of whatever array model_interface.py returns.
"""

import base64
import io

import branca.colormap as bcm
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

CMAP_NAME = "turbo"  # blue -> green -> yellow -> orange -> red, close to the reference map


def _get_cmap(name):
    """matplotlib.cm.get_cmap() is deprecated/removed across recent matplotlib
    versions; this works on both old and new releases."""
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)


def field_to_overlay_image(field, vmin, vmax, upsample=4, blur_sigma=1.2, alpha=190):
    """
    Convert a 2D temperature field (rows = latitude south->north, cols =
    longitude west->east) into a smooth RGBA image suitable for a folium
    ImageOverlay.

    Parameters
    ----------
    field : np.ndarray, shape (nlat, nlon)
    vmin, vmax : float          color scale bounds
    upsample : int              grid up-sampling factor for smoothness
    blur_sigma : float          gaussian blur strength (higher = smoother/blobbier)
    alpha : int (0-255)         overlay opacity, so the basemap shows through

    Returns
    -------
    PIL.Image (RGBA)
    """
    smooth = zoom(field, upsample, order=3)
    smooth = gaussian_filter(smooth, sigma=blur_sigma)

    norm = np.clip((smooth - vmin) / (vmax - vmin + 1e-9), 0, 1)
    cmap = _get_cmap(CMAP_NAME)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3] = alpha

    # PIL images have row 0 = top of image = NORTH; our field row 0 = south, so flip
    rgba = np.flipud(rgba)
    return Image.fromarray(rgba, mode="RGBA")


def image_to_data_url(img):
    """Encode a PIL image as a base64 data: URL folium can embed directly."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def make_colormap_legend(vmin, vmax, n_stops=12, caption="Temperature (°C)"):
    """Build a branca colorbar legend matching the overlay's color scale."""
    cmap = _get_cmap(CMAP_NAME)
    colors = [mcolors.to_hex(cmap(i / (n_stops - 1))) for i in range(n_stops)]
    return bcm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=caption)
