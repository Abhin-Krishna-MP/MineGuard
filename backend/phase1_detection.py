import ee
import geemap
import os

# Import Helpers
try:
    from phase2_tin_viz import generate_tin_visualization
    from report_generator import generate_pdf_report
except ImportError:
    generate_tin_visualization = None
    generate_pdf_report = None

# --- CONFIGURATION ---
PROJECT_ID = 'minesector'
KEY_PATH = "gee-key.json"
START_DATE = '2024-01-01'
END_DATE = '2024-04-30'
CLOUD_THRESHOLD = 20
OPTICAL_THRESHOLD = 0.07
RADAR_THRESHOLD = 0.5
MIN_DEPTH_THRESHOLD = 2.0
DEM_SOURCE = 'COPERNICUS/DEM/GLO30' 

try:
    if os.path.exists(KEY_PATH):
        # Authenticate using the Service Account Key
        service_account = "mineguard-sa@minesector.iam.gserviceaccount.com" # Replace with YOUR SA Email if needed
        ee.Initialize(credentials=ee.ServiceAccountCredentials(service_account, KEY_PATH))
        print("✅ GEE Authenticated via Service Account.")
    else:
        # Fallback for local testing (Requires 'gcloud' installed)
        print("⚠️ Service Key not found. Attempting default auth...")
        ee.Initialize(project=PROJECT_ID)
except Exception as e:
    # If standard init fails, try triggering the flow (only works locally, not in Docker)
    try:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
    except Exception as e2:
        print(f"❌ GEE Auth Failed: {e2}")
        
def run_unified_detection(lease_geojson=None, filename="Manual_Input", output_dir="output"):
    """
    API-Ready Pipeline: Returns results dict instead of printing to stdout.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # --- A. INPUT GEOMETRY ---
    if lease_geojson:
        try:
            roi = ee.Geometry(lease_geojson)
        except Exception:
            roi = ee.Geometry.Polygon([[86.40, 23.70], [86.45, 23.70], [86.45, 23.75], [86.40, 23.75], [86.40, 23.70]])
    else:
        roi = ee.Geometry.Polygon([[86.40, 23.70], [86.45, 23.70], [86.45, 23.75], [86.40, 23.75], [86.40, 23.70]])

    search_zone = roi.buffer(3000)

    # --- B. DETECTION ---
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate(START_DATE, END_DATE).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD)).select(['B4', 'B3', 'B2', 'B8', 'B11']) 
    s2_image = s2.median().clip(search_zone)
    ndbi = s2_image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    optical_mask = ndbi.gt(OPTICAL_THRESHOLD)

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi).filterDate(START_DATE, END_DATE).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.eq('instrumentMode', 'IW'))
    s1_linear = s1.median().clip(search_zone).select('VV').max(0.001)
    s1_db = s1_linear.log10().multiply(10.0).rename('VV_dB')
    roughness = s1_db.reduceNeighborhood(reducer=ee.Reducer.stdDev(), kernel=ee.Kernel.square(3))
    radar_mask = roughness.gt(RADAR_THRESHOLD)
    candidate_mask = optical_mask.And(radar_mask)

    # --- C. VERIFICATION ---
    dem = ee.ImageCollection(DEM_SOURCE).select('DEM').mosaic().clip(search_zone)
    lease_image = ee.Image.constant(0).byte().paint(roi, 1)
    outside_mask = lease_image.eq(0)
    illegal_candidates = candidate_mask.And(outside_mask)

    rim_mask = illegal_candidates.focal_max(60, 'circle', 'meters').And(illegal_candidates.Not())
    rim_stats = dem.updateMask(rim_mask).reduceRegion(reducer=ee.Reducer.mean(), geometry=search_zone, scale=30, maxPixels=1e9)
    lid_elevation = rim_stats.get('DEM').getInfo() or 0.0

    raw_depth = ee.Image.constant(lid_elevation).subtract(dem)
    verified_depth_mask = raw_depth.gt(MIN_DEPTH_THRESHOLD)
    final_illegal_mask = candidate_mask.And(verified_depth_mask).And(outside_mask)

    # --- D. QUANTIFICATION ---
    area_stats = final_illegal_mask.multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=search_zone, scale=10, maxPixels=1e9)
    illegal_area_m2 = area_stats.values().get(0).getInfo() or 0.0

    depth_map_final = raw_depth.updateMask(final_illegal_mask).rename('depth')
    vol_stats = depth_map_final.multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=search_zone, scale=30, maxPixels=1e9)
    total_volume_m3 = vol_stats.get('depth').getInfo() or 0.0
    
    avg_depth_m = total_volume_m3 / illegal_area_m2 if illegal_area_m2 > 0 else 0.0

    # --- E. GENERATE ARTIFACTS ---
    
    # 1. 2D Map
    Map = geemap.Map()
    Map.centerObject(roi, 13)
    Map.addLayer(s2_image, {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}, 'Satellite Image')
    Map.addLayer(final_illegal_mask.selfMask(), {'palette': 'red'}, 'VERIFIED Illegal Mines')
    Map.addLayer(roi, {'color': 'blue', 'width': 3}, 'Legal Lease Boundary')
    
    map_filename = "map_2d.html"
    map_full_path = os.path.join(output_dir, map_filename)
    Map.to_html(map_full_path)

    # 2. 3D TIN
    tin_filename = "model_3d.html"
    tin_full_path = os.path.join(output_dir, tin_filename)
    if illegal_area_m2 > 0 and generate_tin_visualization:
        generate_tin_visualization(depth_map_final, search_zone, illegal_area_m2, output_path=tin_full_path)

    # 3. PDF Report
    pdf_filename = "report.pdf"
    pdf_full_path = os.path.join(output_dir, pdf_filename)
    if generate_pdf_report:
        report_data = {
            "start_date": START_DATE, "end_date": END_DATE, "dem_source": DEM_SOURCE,
            "filename": os.path.basename(filename),
            "illegal_area": illegal_area_m2, "lid_elevation": lid_elevation,
            "avg_depth": avg_depth_m, "volume": total_volume_m3,
            "trucks": int(total_volume_m3 / 15) if total_volume_m3 else 0
        }
        try:
            generate_pdf_report(report_data, output_path=pdf_full_path)
        except Exception as e:
            print(f"PDF Error: {e}")

    # --- F. RETURN RESULTS ---
    return {
        "status": "success",
        "metrics": {
            "illegal_area_m2": round(illegal_area_m2, 2),
            "volume_m3": round(total_volume_m3, 2),
            "avg_depth_m": round(avg_depth_m, 2),
            "truckloads": int(total_volume_m3 / 15)
        },
        "artifacts": {
            "map_url": map_filename,
            "model_url": tin_filename if illegal_area_m2 > 0 else None,
            "report_url": pdf_filename
        }
    }