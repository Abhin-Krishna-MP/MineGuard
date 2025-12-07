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
# Default dates (can be overridden by API)
DEFAULT_START = '2024-01-01'
DEFAULT_END = '2024-04-30'
CLOUD_THRESHOLD = 20
OPTICAL_THRESHOLD = 0.07
RADAR_THRESHOLD = 0.5
MIN_DEPTH_THRESHOLD = 2.0
DEM_SOURCE = 'COPERNICUS/DEM/GLO30' 

try:
    if os.path.exists(KEY_PATH):
        service_account = "mineguard-sa@minesector.iam.gserviceaccount.com"
        ee.Initialize(credentials=ee.ServiceAccountCredentials(service_account, KEY_PATH))
    else:
        ee.Initialize(project=PROJECT_ID)
except:
    try:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
    except:
        pass

def run_unified_detection(lease_geojson=None, filename="Manual_Input", output_dir="output", start_date=DEFAULT_START, end_date=DEFAULT_END):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- A. INPUT ---
    if lease_geojson:
        try:
            roi = ee.Geometry(lease_geojson)
        except:
            roi = ee.Geometry.Polygon([[86.40, 23.70], [86.45, 23.70], [86.45, 23.75], [86.40, 23.75], [86.40, 23.70]])
    else:
        roi = ee.Geometry.Polygon([[86.40, 23.70], [86.45, 23.70], [86.45, 23.75], [86.40, 23.75], [86.40, 23.70]])

    search_zone = roi.buffer(3000)

    # --- B. DETECTION (Total Mining Signature) ---
    print(f"🚀 Step 1: Scanning Surface ({start_date} to {end_date})...")
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate(start_date, end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD)).select(['B4', 'B3', 'B2', 'B8', 'B11']) 
    s2_image = s2.median().clip(search_zone)
    ndbi = s2_image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    optical_mask = ndbi.gt(OPTICAL_THRESHOLD)

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi).filterDate(start_date, end_date).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.eq('instrumentMode', 'IW'))
    s1_linear = s1.median().clip(search_zone).select('VV').max(0.001)
    s1_db = s1_linear.log10().multiply(10.0).rename('VV_dB')
    roughness = s1_db.reduceNeighborhood(reducer=ee.Reducer.stdDev(), kernel=ee.Kernel.square(3))
    radar_mask = roughness.gt(RADAR_THRESHOLD)
    
    candidate_mask = optical_mask.And(radar_mask)

    # --- C. VERIFICATION (Depth Check) ---
    print("🚀 Step 2: Verifying Topography...")
    dem = ee.ImageCollection(DEM_SOURCE).select('DEM').mosaic().clip(search_zone)
    
    rim_mask = candidate_mask.focal_max(60, 'circle', 'meters').And(candidate_mask.Not())
    rim_stats = dem.updateMask(rim_mask).reduceRegion(reducer=ee.Reducer.mean(), geometry=search_zone, scale=30, maxPixels=1e9)
    lid_elevation = rim_stats.get('DEM').getInfo() or 0.0

    raw_depth = ee.Image.constant(lid_elevation).subtract(dem)
    verified_depth_mask = raw_depth.gt(MIN_DEPTH_THRESHOLD)
    
    total_mining_mask = candidate_mask.And(verified_depth_mask)

    # --- D. CLASSIFICATION (Legal vs Illegal) ---
    lease_image = ee.Image.constant(0).byte().paint(roi, 1).unmask(0)
    
    outside_mask = lease_image.eq(0)
    illegal_mask = total_mining_mask.And(outside_mask)
    
    inside_mask = lease_image.eq(1)
    legal_mask = total_mining_mask.And(inside_mask)

    # --- E. QUANTIFICATION ---
    print("🚀 Step 3: Quantifying Volumes (Total vs Illegal)...")
    
    def calc_stats(mask):
        area = mask.multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=search_zone, scale=10, maxPixels=1e9).values().get(0).getInfo() or 0.0
        depth_masked = raw_depth.updateMask(mask)
        vol = depth_masked.multiply(ee.Image.pixelArea()).reduceRegion(reducer=ee.Reducer.sum(), geometry=search_zone, scale=30, maxPixels=1e9).values().get(0).getInfo() or 0.0
        return area, vol

    illegal_area_m2, illegal_vol_m3 = calc_stats(illegal_mask)
    legal_area_m2, legal_vol_m3 = calc_stats(legal_mask)
    
    total_area_m2 = illegal_area_m2 + legal_area_m2
    total_vol_m3 = illegal_vol_m3 + legal_vol_m3
    
    avg_depth_m = illegal_vol_m3 / illegal_area_m2 if illegal_area_m2 > 0 else 0.0

    # --- F. PREPARE 3D DATA ---
    status_map = ee.Image.constant(0).where(illegal_mask, 1).where(legal_mask, 2).rename('status')
    combined_3d_image = raw_depth.rename('depth').addBands(status_map).updateMask(total_mining_mask)

    # --- G. OUTPUT GENERATION ---
    
    # 1. 2D Map
    Map = geemap.Map()
    Map.centerObject(roi, 13)
    Map.addLayer(s2_image, {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}, 'Satellite Image')
    Map.addLayer(legal_mask.selfMask(), {'palette': ['00FF00']}, 'Legal Mining (Inside)')
    Map.addLayer(illegal_mask.selfMask(), {'palette': ['FF0000']}, 'ILLEGAL Encroachment (Outside)')
    Map.addLayer(roi, {'color': 'blue', 'width': 3}, 'Lease Boundary')
    
    map_filename = "map_2d.html"
    Map.to_html(os.path.join(output_dir, map_filename))

    # 2. 3D TIN
    tin_filename = "model_3d.html"
    tin_full_path = os.path.join(output_dir, tin_filename)
    if total_area_m2 > 0 and generate_tin_visualization:
        generate_tin_visualization(combined_3d_image, search_zone, total_area_m2, output_path=tin_full_path)

    # 3. PDF Report
    pdf_filename = "report.pdf"
    if generate_pdf_report:
        report_data = {
            "start_date": start_date, "end_date": end_date, "dem_source": DEM_SOURCE,
            "filename": os.path.basename(filename),
            "illegal_area": illegal_area_m2, "total_area": total_area_m2,
            "lid_elevation": lid_elevation, "avg_depth": avg_depth_m, 
            "volume": illegal_vol_m3, "total_volume": total_vol_m3,
            "trucks": int(illegal_vol_m3 / 15) if illegal_vol_m3 else 0
        }
        try:
            generate_pdf_report(report_data, output_path=os.path.join(output_dir, pdf_filename))
        except Exception as e:
            print(f"PDF Error: {e}")

    # --- H. RETURN METRICS ---
    return {
        "status": "success",
        "metrics": {
            "illegal_area_m2": round(illegal_area_m2, 2),
            "volume_m3": round(illegal_vol_m3, 2),
            "total_vol_m3": round(total_vol_m3, 2),
            "avg_depth_m": round(avg_depth_m, 2),
            "truckloads": int(illegal_vol_m3 / 15)
        },
        "artifacts": {
            "map_url": map_filename,
            "model_url": tin_filename if total_area_m2 > 0 else None,
            "report_url": pdf_filename
        }
    }