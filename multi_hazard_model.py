#!/usr/bin/env python
# coding: utf-8

# # 🗺️ Multi-Hazard Susceptibility Model (Clean Version)
# This notebook aligns all raster layers, applies AHP weighting, and produces a final hazard map.

# In[1]:



## Step 1: Load Required Libraries

We begin by importing the essential Python libraries used throughout this analysis:

1 -rasterio – for reading and writing raster data (GeoTIFFs)
2 -rasterio.warp.reproject and Resampling – for aligning rasters to a common grid
3 -numpy – for numerical operations and array manipulation
4 -matplotlib.pyplot – for plotting the final maps
5 -Path from `pathlib – for handling file paths across operating systems

These tools form the backbone of our geospatial workflow in Python.

In this step, we load all input raster layers such as rainfall, slope, and fault lines.
We check their dimensions and ensure they are aligned spatially before further processing.

import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# In[2]:


# Step 2: Define Raster Alignment Function
def align_raster_to_reference(src_path, ref_path, out_path):
    with rasterio.open(ref_path) as ref:
        ref_meta = ref.meta.copy()
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_meta = src.meta.copy()
        aligned_data = np.empty(ref_shape, dtype=src_meta['dtype'])

        reproject(
            source=src_data,
            destination=aligned_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )

        ref_meta.update({
            'driver': 'GTiff',
            'height': ref_shape[0],
            'width': ref_shape[1],
            'transform': ref_transform,
            'crs': ref_crs
        })

        with rasterio.open(out_path, 'w', **ref_meta) as dst:
            dst.write(aligned_data, 1)


# In[4]:


import rasterio
import numpy as np
from rasterio.enums import Resampling

def calculate_slope(dem_path, slope_path):
    with rasterio.open(dem_path) as src:
        elevation = src.read(1, resampling=Resampling.bilinear)
        transform = src.transform
        cellsize_x = transform.a
        cellsize_y = -transform.e  # negative due to top-left origin

        # Handle no-data values
        elevation = np.where(elevation == src.nodata, np.nan, elevation)

        # Central difference method
        dzdx = (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) / (2 * cellsize_x)
        dzdy = (np.roll(elevation, -1, axis=0) - np.roll(elevation, 1, axis=0)) / (2 * cellsize_y)
        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))

        slope_deg = np.degrees(slope_rad)

        # Replace NaNs with 0 or keep as is
        slope_deg = np.where(np.isnan(slope_deg), 0, slope_deg)

        # Save slope raster
        profile = src.profile
        profile.update(dtype=rasterio.float32, nodata=0)

        with rasterio.open(slope_path, 'w', **profile) as dst:
            dst.write(slope_deg.astype(rasterio.float32), 1)

# 🏔️ Generate slope raster
calculate_slope("thesis_data/dem.tif", "thesis_data/slope_degree.tif")


# In[7]:


import rasterio
with rasterio.open("thesis_data/slope_degree.tif") as src:
    print("Slope shape:", src.shape)
#slope_deg = np.where(np.isnan(slope_deg), 0, slope_deg)


# In[8]:


# Step 3: Align All Layers (with error handling)
from pathlib import Path

input_dir = Path("thesis_data")
aligned_dir = Path("aligned_data")
aligned_dir.mkdir(exist_ok=True)

reference_raster = input_dir / "cyclone_path.tif"

# List of raster files (excluding the reference raster)
raster_files = [
    "river.tif",
    "rainfall.tif",
    "faultline.tif",
    "slope_degree.tif",
    "road_proximity.tif",
    "cyclone_shelters.tif"
]

# Loop through each raster file and align it
for filename in raster_files:
    print(f"🔄 Aligning: {filename}")
    src_path = input_dir / filename
    out_path = aligned_dir / filename
    try:
        align_raster_to_reference(src_path, reference_raster, out_path)
        print(f"✅ Success: {filename}")
    except Exception as e:
        print(f"❌ Skipped: {filename} — Error: {e}")

# Copy the reference raster to the aligned_data folder
ref_path = aligned_dir / "cyclone_path.tif"
with open(input_dir / "cyclone_path.tif", "rb") as src_file:
    with open(ref_path, "wb") as dst_file:
        dst_file.write(src_file.read())

# Set the data directory for the next steps
data_dir = aligned_dir


# In[10]:


# Step 3.5: AHP Pairwise Comparison & Weight Derivation

import numpy as np
import pandas as pd

criteria = [
    "Rainfall", "Slope", "River", "Faultline", 
    "Roads", "CycloneP", "CycloneS"
]

# Your pairwise matrix
matrix = np.array([
    [1,    3,    3,    3,    2,    2,    2],
    [1/3,  1,    2,    3,    4,    2,    2],
    [1/3, 1/2,   1,    4,    2,    2,    2],
    [1/2, 1/2, 1/4,    1,    2,    2,    2],
    [1/2, 1/4, 1/2,  1/2,    1,    2,    2],
    [1/2, 1/2, 1/2,  1/2,  1/2,    1,  0.5],
    [1/2, 1/2, 1/2,  1/2,  1/2,    2,    1]
])

# Normalize columns
col_sums = matrix.sum(axis=0)
normalized_matrix = matrix / col_sums

# Calculate priority weights
weights = normalized_matrix.mean(axis=1)

# Create weight table
weights_df = pd.DataFrame({
    "Criteria": criteria,
    "Weight (Decimal)": weights,
    "Weight (%)": weights * 100
}).sort_values(by="Weight (Decimal)", ascending=False)

# Print weight table
print("✅ AHP-Derived Weights:")
display(weights_df)

# Consistency Check
# λ_max = (A*w)/w
Aw = matrix @ weights
lambda_max = (Aw / weights).mean()

n = len(criteria)
CI = (lambda_max - n) / (n - 1)

# Random Index (RI) values
RI_dict = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
    5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
    9: 1.45, 10: 1.49
}

RI = RI_dict[n]
CR = CI / RI

print(f"\n🧠 Consistency Index (CI): {CI:.4f}")
print(f"📏 Consistency Ratio (CR): {CR:.4f}")
if CR < 0.1:
    print("✅ Judgments are consistent (CR < 0.10)")
else:
    print("⚠️ Judgments may be inconsistent (CR ≥ 0.10)")


# In[26]:


from pathlib import Path
import rasterio
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Load slope_degree.tif as mask
mask_path = Path("aligned_data") / "slope_degree.tif"
with rasterio.open(mask_path) as mask_src:
    slope_mask = mask_src.read(1)
    nodata_val = mask_src.nodata
    if nodata_val is not None:
        slope_mask = np.where(slope_mask == nodata_val, 0, slope_mask)

# Mask: keep only where slope exists (> 0)
masked_score = np.where(slope_mask > 0, final_score, np.nan)

# Optional: rescale for better visual stretch (clipping extremes)
vmin = np.nanpercentile(masked_score, 2)   # 2nd percentile
vmax = np.nanpercentile(masked_score, 98)  # 98th percentile

# Masked plot
masked_score = ma.masked_invalid(masked_score)
cmap = cm.get_cmap("RdYlGn_r")
cmap_colors = cmap(np.linspace(0, 1, cmap.N))
new_cmap = ListedColormap(cmap_colors)
new_cmap.set_bad(color='white')

plt.figure(figsize=(10, 6), facecolor='white')
img = plt.imshow(masked_score, cmap=new_cmap, vmin=vmin, vmax=vmax)
cbar = plt.colorbar(img, label="Hazard Susceptibility Score")
cbar.ax.tick_params(labelsize=10)
plt.title("Multi-Hazard Susceptibility Map (Masked by Slope)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()


# In[29]:


import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib import cm
from matplotlib.colors import ListedColormap

# ✅ Apply final mask (already done)
masked_score = ma.masked_invalid(masked_score)

# ✅ Green-to-red gradient (reversed)
cmap = cm.get_cmap("RdYlGn_r")
cmap_colors = cmap(np.linspace(0, 1, cmap.N))
new_cmap = ListedColormap(cmap_colors)
new_cmap.set_bad(color='white')

# ✅ Plot with tight layout and no axes
plt.figure(figsize=(10, 6), facecolor='white')
img = plt.imshow(masked_score, cmap=new_cmap, vmin=0, vmax=1)
cbar = plt.colorbar(img, label="Hazard Susceptibility Score", shrink=0.8, pad=0.02)
cbar.ax.tick_params(labelsize=10)

plt.title("Multi-Hazard Susceptibility Map (AHP-Based)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()


# In[28]:


import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy as np
import numpy.ma as ma

# ------------------------
# Step 1: Classification
# ------------------------

# You can adjust these thresholds based on your results
breaks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
colors = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']  # green to red
cmap = ListedColormap(colors)

# Classify and mask
classified = np.digitize(masked_score, bins=breaks, right=True)
classified_masked = ma.masked_where(np.isnan(masked_score), classified)

# ------------------------
# Step 2: Plot Map
# ------------------------

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

# Map with discrete colors
img = ax.imshow(classified_masked, cmap=cmap, vmin=1, vmax=len(colors))

# Turn off axis
ax.axis('off')
ax.set_title("Multi-Hazard Susceptibility Map", fontsize=14)

# ------------------------
# Step 3: Custom Legend
# ------------------------

legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))]

# Place legend BELOW the map
fig.legend(handles=legend_elements,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=len(labels),
           fontsize=10,
           title="Susceptibility Level")

plt.tight_layout()
plt.show()


# In[21]:


# Step 4: Combine Raster Layers Using AHP Weights

from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from matplotlib import cm
from matplotlib.colors import ListedColormap

# 📁 Folder with aligned data
data_dir = Path("aligned_data")

# AHP Weights (replace with yours if different)
criteria_weights = {
    "rainfall.tif": 0.2709,
    "slope_degree.tif": 0.1917,
    "river.tif": 0.1723,
    "faultline.tif": 0.1202,
    "road_proximity.tif": 0.0968,
    "cyclone_path.tif": 0.0816,
    "cyclone_shelters.tif": 0.0665
}

# Cost vs. Benefit
criteria_types = {
    "rainfall.tif": "benefit",
    "slope_degree.tif": "benefit",
    "river.tif": "benefit",
    "faultline.tif": "benefit",
    "road_proximity.tif": "cost",
    "cyclone_path.tif": "benefit",
    "cyclone_shelters.tif": "cost"
}

# 🔁 Combine layers
final_score = None

for fname, weight in criteria_weights.items():
    fpath = data_dir / fname
    with rasterio.open(fpath) as src:
        data = src.read(1).astype(float)

        # ✅ Handle NoData properly
        nodata_val = src.nodata
        if nodata_val is not None:
            data[data == nodata_val] = np.nan
        data[np.isinf(data)] = np.nan

        # Normalize (0–1)
        if criteria_types[fname] == "benefit":
            norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        else:
            norm = (np.nanmax(data) - data) / (np.nanmax(data) - np.nanmin(data))

        # Weighted sum
        weighted = norm * weight
        final_score = weighted if final_score is None else final_score + weighted

# 📦 Save final raster
output_path = data_dir / "multi_hazard_score.tif"
with rasterio.open(fpath) as ref:
    profile = ref.profile
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(final_score.astype(rasterio.float32), 1)

# 🎨 Mask and Plot with green-to-red gradient, white for no-data
masked = ma.masked_invalid(final_score)
cmap = cm.get_cmap("RdYlGn_r")
cmap_colors = cmap(np.linspace(0, 1, cmap.N))
new_cmap = ListedColormap(cmap_colors)
new_cmap.set_bad(color='white')  # white background for NaN

plt.figure(figsize=(10, 6), facecolor='white')
img = plt.imshow(masked, cmap=new_cmap, vmin=0, vmax=1)
cbar = plt.colorbar(img, label="Hazard Susceptibility Score")
cbar.ax.tick_params(labelsize=10)

plt.title("Multi-Hazard Susceptibility Map (AHP-Based)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()


# In[23]:


import rasterio
import numpy as np

fpath = Path("aligned_data") / "rainfall.tif"  # change to another file if needed

with rasterio.open(fpath) as src:
    data = src.read(1).astype(float)
    print("Nodata value:", src.nodata)
    print("Top-left 5x5:")
    print(data[:5, :5])  # top-left corner
    print("Bottom-right 5x5:")
    print(data[-5:, -5:])  # bottom-right corner
    import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Mask NaN and 0 values (assuming 0 = meaningless background)
masked_score = ma.masked_where((final_score == 0) | np.isnan(final_score), final_score)

# Create green-to-red colormap and set masked (bad) to white
cmap = cm.get_cmap("RdYlGn_r")
cmap_colors = cmap(np.linspace(0, 1, cmap.N))
new_cmap = ListedColormap(cmap_colors)
new_cmap.set_bad(color='white')

# Plot
plt.figure(figsize=(10, 6), facecolor='white')
img = plt.imshow(masked_score, cmap=new_cmap, vmin=0, vmax=1)
cbar = plt.colorbar(img, label="Hazard Susceptibility Score")
cbar.ax.tick_params(labelsize=10)

plt.title("Multi-Hazard Susceptibility Map (AHP-Based)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()


# In[24]:


import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# 👇 Threshold all very low values (including 0) as invalid
threshold = 0.01  # You can adjust this if needed
masked_score = ma.masked_where((final_score < threshold) | np.isnan(final_score), final_score)

# 🎨 Green-to-red colormap with white for masked areas
cmap = cm.get_cmap("RdYlGn_r")
cmap_colors = cmap(np.linspace(0, 1, cmap.N))
new_cmap = ListedColormap(cmap_colors)
new_cmap.set_bad(color='white')

# 🖼️ Plot
plt.figure(figsize=(10, 6), facecolor='white')
img = plt.imshow(masked_score, cmap=new_cmap, vmin=0, vmax=1)
cbar = plt.colorbar(img, label="Hazard Susceptibility Score")
cbar.ax.tick_params(labelsize=10)

plt.title("Multi-Hazard Susceptibility Map (AHP-Based)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()


# In[15]:


# 📊 Plot result with green-to-red and masked no-data
import numpy.ma as ma
from matplotlib import cm

plt.figure(figsize=(10, 6), facecolor='white')

# Mask NaNs
masked = ma.masked_invalid(final_score)

# Colormap setup
cmap = cm.get_cmap("RdYlGn_r")
cmap = cm.ScalarMappable(cmap=cmap).cmap
cmap.set_bad(color='white')

# Plot
img = plt.imshow(masked, cmap=cmap, vmin=0, vmax=1)
cbar = plt.colorbar(img, label="Hazard Susceptibility Score")
cbar.ax.tick_params(labelsize=10)

plt.title("Multi-Hazard Susceptibility Map (AHP-Based)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()


# In[16]:


import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import cm

# Step 1: Mask NaNs in the final hazard score
masked_score = ma.masked_invalid(final_score)

# Step 2: Get the green-to-red reversed colormap
cmap = cm.get_cmap("RdYlGn_r").with_extremes(bad='white')

# Step 3: Plot
plt.figure(figsize=(10, 6), facecolor='white')
img = plt.imshow(masked_score, cmap=cmap, vmin=0, vmax=1)

cbar = plt.colorbar(img, label="Hazard Susceptibility Score")
cbar.ax.tick_params(labelsize=10)

plt.title("Multi-Hazard Susceptibility Map (AHP-Based)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()


# In[17]:


import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Step 1: Mask NaNs
masked_score = ma.masked_invalid(final_score)

# Step 2: Get green-to-red colormap and set NaNs to white
cmap = cm.get_cmap("RdYlGn_r")
cmap_colors = cmap(np.linspace(0, 1, cmap.N))
new_cmap = ListedColormap(cmap_colors)
new_cmap.set_bad(color='white')  # NaNs will be white

# Step 3: Plot
plt.figure(figsize=(10, 6), facecolor='white')
img = plt.imshow(masked_score, cmap=new_cmap, vmin=0, vmax=1)
cbar = plt.colorbar(img, label="Hazard Susceptibility Score")
cbar.ax.tick_params(labelsize=10)

plt.title("Multi-Hazard Susceptibility Map (AHP-Based)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()

