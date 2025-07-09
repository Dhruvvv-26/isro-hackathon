import rasterio
import geopandas as gpd
from shapely.geometry import box

def generate_aoi_from_raster(raster_path="scene1/B3.tif", output_geojson='aoi.geojson'):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        print(f"Bounds: {bounds}")
        print(f"CRS: {crs}")

        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs=crs)
        gdf = gdf.to_crs("EPSG:4326")  # Convert to lat/lon
        gdf.to_file(output_geojson, driver='GeoJSON')
        print(f"✅ AOI saved to {output_geojson}")

# Run it
if __name__ == "__main__":
    generate_aoi_from_raster()
