# import rasterio
# import matplotlib.pyplot as plt
# import numpy as np

# # Helper function to load a single band
# def load_band(file_path):
#     with rasterio.open(file_path) as src:
#         band = src.read(1)  # Read first (and usually only) band
#         profile = src.profile  # Metadata
#     return band, profile

# # Load Red, Green, and NIR from Scene 1
# green1, profile = load_band("./scene1/B2.tif")
# red1, _ = load_band("scene1/B3.tif")
# nir1, _ = load_band("scene1/B4.tif")

# # (Optional) Load bands from Scene 2 as well
# green2, _ = load_band("scene2/B2.tif")
# red2, _ = load_band("scene2/B3.tif")
# nir2, _ = load_band("scene2/B4.tif")

# # Visualize one band (e.g., Red)
# plt.imshow(red1, cmap='gray')
# plt.title("Scene 1 - Red Band")
# plt.colorbar()
# plt.show()

# # Step 1: Calculate NDVI for both scenes
# def compute_ndvi(nir, red):
#     ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-6)  # Add small value to avoid zero division
#     return np.clip(ndvi, -1, 1)

# ndvi1 = compute_ndvi(nir1, red1)
# ndvi2 = compute_ndvi(nir2, red2)

# # Step 2: NDVI Difference Map
# ndvi_diff = ndvi2 - ndvi1

# # Optional: Clip values to the expected NDVI range
# ndvi_diff = np.clip(ndvi_diff, -1, 1)

# # Step 3: Apply Threshold to Filter Changes
# # Threshold NDVI change to isolate strong changes
# change_map = np.zeros_like(ndvi_diff, dtype=np.int8)

# # You can adjust thresholds as needed (e.g., ±0.2 or ±0.3)
# gain_threshold = 0.2
# loss_threshold = -0.2

# change_map[ndvi_diff >= gain_threshold] = 1   # Significant vegetation gain
# change_map[ndvi_diff <= loss_threshold] = -1  # Significant vegetation loss

# # Visualize thresholded change map
# plt.figure(figsize=(6, 6))
# plt.imshow(change_map, cmap='bwr', vmin=-1, vmax=1)
# plt.title("Thresholded Change Map (-1 = Loss, 0 = No Change, +1 = Gain)")
# plt.colorbar(label="Change Category")
# plt.show()
    

# # Visualize the difference
# plt.figure(figsize=(6, 6))
# plt.imshow(ndvi_diff, cmap='bwr', vmin=-1, vmax=1)
# plt.title("NDVI Change (Scene2 - Scene1)")
# plt.colorbar(label="NDVI Difference")
# plt.show()


# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(ndvi1, cmap='RdYlGn')
# plt.title("NDVI - Scene 1")
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(ndvi2, cmap='RdYlGn')
# plt.title("NDVI - Scene 2")
# plt.colorbar()

# plt.tight_layout()
# plt.show()


# from rasterio.mask import mask
# import geopandas as gpd

# def clip_to_aoi(band_array, profile, aoi_path='aoi.geojson'):
#     aoi = gpd.read_file(aoi_path)
#     geom = [aoi.geometry[0].__geo_interface__]

#     with rasterio.MemoryFile() as memfile:
#         with memfile.open(**profile) as dataset:
#             dataset.write(band_array, 1)
#             clipped, clipped_transform = mask(dataset, geom, crop=True)

#     clipped_profile = profile.copy()
#     clipped_profile.update({
#         "height": clipped.shape[1],
#         "width": clipped.shape[2],
#         "transform": clipped_transform
#     })

#     return clipped[0], clipped_profile

# import rasterio
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import matplotlib.patches as mpatches
# from rasterio.enums import Resampling
# # -----------------------------
# # CLOUD/SHADOW SEGMENTATION MODEL
# # -----------------------------
# import torch
# import torch.nn as nn

# class SimpleUNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 1, 1)
#         )

#     def forward(self, x):
#         return self.net(x)

# model = SimpleUNet()
# model.eval()


# # --------------------------
# # CONFIGURATION
# # --------------------------
# USE_CLOUD_MASK = False  # Set to True if you have cloud masks (1 = cloud)
# GAIN_STD_MULTIPLIER = 1  # Multiplier for std-dev thresholding

# # --------------------------
# # Helper Function to Load a Band
# # --------------------------
# def load_band(file_path):
#     with rasterio.open(file_path) as src:
#         band = src.read(1)
#         profile = src.profile
#     return band, profile

# def prepare_input_for_model(red, green, nir):
#     red_norm = red / 10000.0
#     green_norm = green / 10000.0
#     nir_norm = nir / 10000.0
#     stacked = np.stack([red_norm, green_norm, nir_norm], axis=0)
#     return stacked.astype(np.float32)


# def predict_cloud_mask(model, red, green, nir):
#     input_tensor = prepare_input_for_model(red, green, nir)
#     input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)

#     with torch.no_grad():
#         output = model(input_tensor)[0, 0].numpy()

#     mask = (output > 0.5).astype(np.uint8)
#     return mask

# # --------------------------
# # Resample Scene 2 to Match Scene 1
# # --------------------------
# def resample_to_match(src_array, src_profile, match_profile):
#     scale_x = match_profile['transform'][0] / src_profile['transform'][0]
#     scale_y = -match_profile['transform'][4] / src_profile['transform'][4]

#     dst_height = match_profile['height']
#     dst_width = match_profile['width']

#     with rasterio.MemoryFile() as memfile:
#         with memfile.open(
#             driver='GTiff',
#             height=src_array.shape[0],
#             width=src_array.shape[1],
#             count=1,
#             dtype=src_array.dtype,
#             crs=src_profile['crs'],
#             transform=src_profile['transform']
#         ) as dataset:
#             dataset.write(src_array, 1)

#             resampled = dataset.read(
#                 out_shape=(1, dst_height, dst_width),
#                 resampling=Resampling.bilinear
#             )[0]

#     return resampled


# # Clip all bands to AOI
# red1, profile = clip_to_aoi(red1, profile)
# green1, profile = clip_to_aoi(green1, profile)
# nir1, profile = clip_to_aoi(nir1, profile)

# red2, _ = clip_to_aoi(red2, profile)
# green2, _ = clip_to_aoi(green2, profile)
# nir2, _ = clip_to_aoi(nir2, profile)


# # --------------------------
# # Optional: Cloud Masking
# # --------------------------
# def apply_mask(band, mask):
#     return np.where(mask == 1, np.nan, band)  # 1 = cloud/shadow

# # --------------------------
# # Load Scene 1
# # --------------------------
# green1, profile = load_band("scene1/B2.tif")
# red1, _ = load_band("scene1/B3.tif")
# nir1, _ = load_band("scene1/B4.tif")

# # --------------------------
# # Load Scene 2 (raw)
# # --------------------------
# green2, profile2 = load_band("scene2/B2.tif")
# red2, _ = load_band("scene2/B3.tif")
# nir2, _ = load_band("scene2/B4.tif")

# # --------------------------
# # Resample Scene 2 Bands to Match Scene 1
# # --------------------------
# green2 = resample_to_match(green2, profile2, profile)
# red2 = resample_to_match(red2, profile2, profile)
# nir2 = resample_to_match(nir2, profile2, profile)

# # Optional: Cloud Mask Resampling
# if USE_CLOUD_MASK:
#     cloud_mask1, _ = load_band("scene1/cloudmask.tif")
#     cloud_mask2, profile2 = load_band("scene2/cloudmask.tif")
#     cloud_mask2 = resample_to_match(cloud_mask2, profile2, profile)

#     red1 = apply_mask(red1, cloud_mask1)
#     nir1 = apply_mask(nir1, cloud_mask1)
#     red2 = apply_mask(red2, cloud_mask2)
#     nir2 = apply_mask(nir2, cloud_mask2)

# # --------------------------
# # Compute NDVI
# # --------------------------
# def compute_ndvi(nir, red):
#     ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-6)
#     return np.clip(ndvi, -1, 1)

# ndvi1 = compute_ndvi(nir1, red1)
# ndvi2 = compute_ndvi(nir2, red2)

# # --------------------------
# # NDVI Difference
# # --------------------------
# ndvi_diff = ndvi2 - ndvi1
# ndvi_diff = np.clip(ndvi_diff, -1, 1)

# # --------------------------
# # Thresholding (dynamic)
# # --------------------------
# mean_diff = np.nanmean(ndvi_diff)
# std_diff = np.nanstd(ndvi_diff)

# gain_threshold = mean_diff + GAIN_STD_MULTIPLIER * std_diff
# loss_threshold = mean_diff - GAIN_STD_MULTIPLIER * std_diff

# change_map = np.zeros_like(ndvi_diff, dtype=np.int8)
# change_map[ndvi_diff >= gain_threshold] = 1
# change_map[ndvi_diff <= loss_threshold] = -1

# # --------------------------
# # Export GeoTIFFs
# # --------------------------
# def export_geotiff(filename, array, profile, dtype='float32'):
#     profile.update(dtype=dtype, count=1, nodata=np.nan if 'float' in dtype else 0)
#     with rasterio.open(filename, 'w', **profile) as dst:
#         dst.write(array.astype(dtype), 1)

# os.makedirs("output", exist_ok=True)
# export_geotiff("output/ndvi_scene1.tif", ndvi1, profile)
# export_geotiff("output/ndvi_scene2.tif", ndvi2, profile)
# export_geotiff("output/ndvi_difference.tif", ndvi_diff, profile)
# export_geotiff("output/change_threshold.tif", change_map, profile, dtype='int8')

# # --------------------------
# # Alert System
# # --------------------------
# loss_pixels = np.count_nonzero(change_map == -1)
# gain_pixels = np.count_nonzero(change_map == 1)
# total_pixels = change_map.size

# percent_loss = (loss_pixels / total_pixels) * 100
# percent_gain = (gain_pixels / total_pixels) * 100

# alert_lines = [
#     "🔔 ALERT REPORT",
#     f"Vegetation Loss Pixels: {loss_pixels} ({percent_loss:.2f}%)",
#     f"Vegetation Gain Pixels: {gain_pixels} ({percent_gain:.2f}%)"
# ]

# if percent_loss > 5:
#     alert_lines.append("⚠️ Alert: Significant vegetation loss detected in AOI!")

# print("\n".join(alert_lines))
# with open("output/alert_report.txt", "w", encoding="utf-8") as f:
#     f.write("\n".join(alert_lines))


# # --------------------------
# # Visualization
# # --------------------------
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(ndvi1, cmap='RdYlGn', vmin=-1, vmax=1)
# plt.title("NDVI - Scene 1")
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(ndvi2, cmap='RdYlGn', vmin=-1, vmax=1)
# plt.title("NDVI - Scene 2")
# plt.colorbar()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.imshow(ndvi_diff, cmap='bwr', vmin=-1, vmax=1)
# plt.title("NDVI Change (Scene2 - Scene1)")
# plt.colorbar(label="NDVI Difference")
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.imshow(change_map, cmap='bwr', vmin=-1, vmax=1)
# plt.title("Thresholded Change Map")

# red_patch = mpatches.Patch(color='blue', label='Vegetation Loss')
# green_patch = mpatches.Patch(color='red', label='Vegetation Gain')
# plt.legend(handles=[red_patch, green_patch], loc='lower right')

# plt.colorbar(label="Change Category")
# plt.show()

# plt.figure(figsize=(10, 4))
# plt.hist(ndvi1.flatten(), bins=50, alpha=0.5, label="Scene 1", color='green')
# plt.hist(ndvi2.flatten(), bins=50, alpha=0.5, label="Scene 2", color='brown')
# plt.title("NDVI Distribution Comparison")
# plt.xlabel("NDVI Value")
# plt.ylabel("Pixel Count")
# plt.legend()
# plt.show()

# # --------------------------
# # Overlay Change Map on Scene 2 NDVI
# # --------------------------
# from matplotlib.colors import ListedColormap

# # Create a custom transparent colormap for overlay
# overlay = np.zeros((*change_map.shape, 4))  # RGBA
# overlay[change_map == 1] = [0, 1, 0, 0.4]   # Green for gain
# overlay[change_map == -1] = [1, 0, 0, 0.4]  # Red for loss

# plt.figure(figsize=(8, 8))
# plt.imshow(ndvi2, cmap='RdYlGn', vmin=-1, vmax=1)
# plt.imshow(overlay, interpolation='none')
# plt.title("NDVI Scene 2 with Change Overlay (Gain/Loss)")
# plt.colorbar(label="NDVI Value")

# # Add Legend
# gain_patch = mpatches.Patch(color='green', label='Vegetation Gain')
# loss_patch = mpatches.Patch(color='red', label='Vegetation Loss')
# plt.legend(handles=[gain_patch, loss_patch], loc='lower right')

# plt.tight_layout()
# plt.show()

# analyze_satellite.py

# ✅ FINAL CLEANED FILE: cloud_shadow_ndvi.py
# -----------------------------
# PHASE 1, 2, 3 Fully Integrated with Inline Comments
# -----------------------------
# ✅ FINAL CLEANED & FIXED FILE: cloud_shadow_ndvi.py
# ✅ FINAL CLEANED & FIXED FILE: cloud_shadow_ndvi.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
import geopandas as gpd
from matplotlib.colors import ListedColormap

# -----------------------------
# AUTO-GENERATE AOI FROM RASTER
# -----------------------------
def generate_aoi_from_raster(raster_path, output_geojson="aoi.geojson"):
    from shapely.geometry import box
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
    geom = box(*bounds)
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=crs)
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"✅ AOI saved to {output_geojson}")

# -----------------------------
# CONFIGURATION
# -----------------------------
USE_CLOUD_MASK = True
GAIN_STD_MULTIPLIER = 1
AOI_PATH = 'aoi.geojson'

# -----------------------------
# SIMPLE UNET MODEL (Placeholder)
# -----------------------------
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleUNet()
model.eval()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_band(file_path):
    with rasterio.open(file_path) as src:
        band = src.read(1)
        profile = src.profile
    return band, profile

def prepare_input_for_model(red, green, nir):
    stacked = np.stack([
        red / 10000.0,
        green / 10000.0,
        nir / 10000.0
    ], axis=0)
    return stacked.astype(np.float32)

def predict_cloud_mask(model, red, green, nir):
    input_tensor = prepare_input_for_model(red, green, nir)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)

    tile_size = 512
    h, w = input_tensor.shape[2:]
    mask = np.zeros((h, w), dtype=np.uint8)

    with torch.no_grad():
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                tile = input_tensor[:, :, i:i+tile_size, j:j+tile_size]
                if tile.shape[2] == 0 or tile.shape[3] == 0:
                    continue
                out = model(tile)[0, 0].numpy()
                mask[i:i+tile.shape[2], j:j+tile.shape[3]] = (out > 0.5).astype(np.uint8)

    return mask

def apply_mask(band, mask):
    if band.shape != mask.shape:
        from skimage.transform import resize
        mask = resize(mask, band.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    return np.where(mask == 1, np.nan, band)

def resample_to_match(src_array, src_profile, match_profile):
    dst_height, dst_width = match_profile['height'], match_profile['width']
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=src_array.shape[0],
            width=src_array.shape[1],
            count=1,
            dtype=src_array.dtype,
            crs=src_profile['crs'],
            transform=src_profile['transform']
        ) as dataset:
            dataset.write(src_array, 1)
            resampled = dataset.read(
                out_shape=(1, dst_height, dst_width),
                resampling=Resampling.bilinear
            )[0]
    return resampled

def clip_to_aoi(band_array, profile, aoi_path=AOI_PATH):
    aoi = gpd.read_file(aoi_path)
    aoi = aoi.to_crs(profile['crs'])
    geom = [aoi.geometry[0].__geo_interface__]
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(band_array, 1)
            try:
                clipped, transform = mask(dataset, geom, crop=True)
            except ValueError:
                raise ValueError("\u274c AOI does not overlap raster. Fix AOI bounds/CRS.")
    profile.update({
        'height': clipped.shape[1],
        'width': clipped.shape[2],
        'transform': transform
    })
    return clipped[0], profile

def compute_ndvi(nir, red):
    ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-6)
    return np.clip(ndvi, -1, 1)

def export_geotiff(filename, array, profile, dtype='float32'):
    profile.update(dtype=dtype, count=1, nodata=np.nan if 'float' in dtype else 0)
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(array.astype(dtype), 1)

# -----------------------------
# EXECUTION PIPELINE
# -----------------------------
generate_aoi_from_raster("scene1/B3.tif")

green1, profile = load_band("scene1/B2.tif")
red1, _ = load_band("scene1/B3.tif")
nir1, _ = load_band("scene1/B4.tif")

green2, profile2 = load_band("scene2/B2.tif")
red2, _ = load_band("scene2/B3.tif")
nir2, _ = load_band("scene2/B4.tif")

red2 = resample_to_match(red2, profile2, profile)
green2 = resample_to_match(green2, profile2, profile)
nir2 = resample_to_match(nir2, profile2, profile)

red1, profile = clip_to_aoi(red1, profile)
green1, profile = clip_to_aoi(green1, profile)
nir1, profile = clip_to_aoi(nir1, profile)

red2, _ = clip_to_aoi(red2, profile)
green2, _ = clip_to_aoi(green2, profile)
nir2, _ = clip_to_aoi(nir2, profile)

if USE_CLOUD_MASK:
    cloud_mask1 = predict_cloud_mask(model, red1, green1, nir1)
    cloud_mask2 = predict_cloud_mask(model, red2, green2, nir2)
    red1 = apply_mask(red1, cloud_mask1)
    nir1 = apply_mask(nir1, cloud_mask1)
    red2 = apply_mask(red2, cloud_mask2)
    nir2 = apply_mask(nir2, cloud_mask2)

ndvi1 = compute_ndvi(nir1, red1)
ndvi2 = compute_ndvi(nir2, red2)
ndvi_diff = np.clip(ndvi2 - ndvi1, -1, 1)

mean_diff = np.nanmean(ndvi_diff)
std_diff = np.nanstd(ndvi_diff)
gain_thresh = mean_diff + GAIN_STD_MULTIPLIER * std_diff
loss_thresh = mean_diff - GAIN_STD_MULTIPLIER * std_diff

change_map = np.zeros_like(ndvi_diff, dtype=np.int8)
change_map[ndvi_diff >= gain_thresh] = 1
change_map[ndvi_diff <= loss_thresh] = -1

os.makedirs("output", exist_ok=True)
export_geotiff("output/ndvi_scene1.tif", ndvi1, profile)
export_geotiff("output/ndvi_scene2.tif", ndvi2, profile)
export_geotiff("output/ndvi_difference.tif", ndvi_diff, profile)
export_geotiff("output/change_threshold.tif", change_map, profile, dtype='int8')

if USE_CLOUD_MASK:
    export_geotiff("output/cloud_mask_scene1.tif", cloud_mask1, profile, dtype='uint8')
    export_geotiff("output/cloud_mask_scene2.tif", cloud_mask2, profile, dtype='uint8')

loss_pixels = np.count_nonzero(change_map == -1)
gain_pixels = np.count_nonzero(change_map == 1)
total_pixels = change_map.size

percent_loss = (loss_pixels / total_pixels) * 100
percent_gain = (gain_pixels / total_pixels) * 100

alert_lines = [
    "🔔 ALERT REPORT",
    f"Vegetation Loss Pixels: {loss_pixels} ({percent_loss:.2f}%)",
    f"Vegetation Gain Pixels: {gain_pixels} ({percent_gain:.2f}%)"
]
if percent_loss > 5:
    alert_lines.append("⚠️ Alert: Significant vegetation loss detected in AOI!")

print("\n".join(alert_lines))
with open("output/alert_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(alert_lines))

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(ndvi1, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title("NDVI - Scene 1")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(ndvi2, cmap='RdYlGn', vmin=-1, vmax=1)
plt.title("NDVI - Scene 2")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(ndvi_diff, cmap='bwr', vmin=-1, vmax=1)
plt.title("NDVI Change (Scene2 - Scene1)")
plt.colorbar()
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(change_map, cmap='bwr', vmin=-1, vmax=1)
plt.title("Thresholded Change Map")
plt.colorbar(label="Change Category")
plt.legend([
    mpatches.Patch(color='green', label='Vegetation Gain'),
    mpatches.Patch(color='red', label='Vegetation Loss')
])
plt.show()

plt.figure(figsize=(10, 4))
plt.hist(ndvi1.flatten(), bins=50, alpha=0.5, label="Scene 1", color='green')
plt.hist(ndvi2.flatten(), bins=50, alpha=0.5, label="Scene 2", color='brown')
plt.title("NDVI Distribution Comparison")
plt.xlabel("NDVI Value")
plt.ylabel("Pixel Count")
plt.legend()
plt.show()

overlay = np.zeros((*change_map.shape, 4))
overlay[change_map == 1] = [0, 1, 0, 0.4]
overlay[change_map == -1] = [1, 0, 0, 0.4]

plt.figure(figsize=(8, 8))
plt.imshow(ndvi2, cmap='RdYlGn', vmin=-1, vmax=1)
plt.imshow(overlay, interpolation='none')
plt.title("NDVI Scene 2 with Gain/Loss Overlay")
plt.legend([
    mpatches.Patch(color='green', label='Vegetation Gain'),
    mpatches.Patch(color='red', label='Vegetation Loss')
])
plt.colorbar()
plt.tight_layout()
plt.show()

