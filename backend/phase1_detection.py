# phase1_detection.py

import ee
import geemap
import os
import numpy as np
import cv2  # <--- NEW IMPORT

# Import Helpers
try:
    from phase2_tin_viz import generate_tin_visualization
    from report_generator import generate_pdf_report
    from ai_inference import MineSegmenter  # <--- NEW IMPORT
except ImportError:
    generate_tin_visualization = None
    generate_pdf_report = None
    MineSegmenter = None

# ... (Configuration & Auth code remains the same) ...

def run_unified_detection(lease_geojson=None, filename="Manual_Input", output_dir="output", start_date=DEFAULT_START, end_date=DEFAULT_END):
    os.makedirs(output_dir, exist_ok=True)
    lid_elevation = 0.0
    
    # ... (Step A: Input Geometry code remains the same) ...

    # ... (Step B: Sensor Detection code remains the same) ...
    
    # ... (Step C: Fusion code remains the same) ...

    # --- D. QUANTIFICATION ---
    # ... (Existing Quantification code) ...
    legal_area_m2, legal_vol_m3 = get_metrics(legal_mining, "Legal")
    illegal_area_m2, illegal_vol_m3 = get_metrics(illegal_mining, "Illegal")
    # ... (Rest of existing aggregation) ...

    # ==========================================
    # 🧠 NEW STEP: AI CROSS-VERIFICATION
    # ==========================================
    print("🤖 Step 3: Running Deep Learning Inference...")
    ai_mask_filename = "ai_prediction.png"
    ai_overlay_filename = "ai_overlay.jpg"
    
    try:
        # 1. Extract RGB pixels from Earth Engine (Visual Bands)
        # We need to visualize the specific ROI
        roi_bounds = search_zone.bounds()
        
        # Convert EE Image to Numpy (This downloads the pixels)
        # Scale=10 ensures high res for the AI
        rgb_image = geemap.ee_to_numpy(
            s2_image.select(['B4', 'B3', 'B2']).divide(3000).multiply(255).uint8(),
            region=roi_bounds,
            scale=10 
        )

        if rgb_image is not None and MineSegmenter is not None:
            # 2. Run Inference
            # Remove Alpha channel if exists (Keep only RGB)
            if rgb_image.shape[2] > 3:
                rgb_image = rgb_image[:, :, :3]
                
            ai_mask = MineSegmenter().predict(rgb_image)
            
            # 3. Create Visual Artifacts
            # Save the raw mask
            cv2.imwrite(os.path.join(output_dir, ai_mask_filename), ai_mask)
            
            # Create an overlay (Red contour on original image)
            # Convert RGB (EE output) to BGR (OpenCV format)
            img_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Draw contours
            contours, _ = cv2.findContours(ai_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, contours, -1, (0, 0, 255), 2) # Red line, thickness 2
            
            cv2.imwrite(os.path.join(output_dir, ai_overlay_filename), img_bgr)
            print("✅ AI Analysis Complete")
        else:
            print("⚠️ Skipping AI: Image extraction failed or Model not loaded.")

    except Exception as e:
        print(f"❌ AI Inference Failed: {e}")

    # ==========================================
    # END AI STEP
    # ==========================================

    # ... (Step E: Prepare 3D Data remains the same) ...

    # ... (Step F: Output Generation - Map, TIN, PDF remains the same) ...

    # --- G. RETURN METRICS ---
    return {
        "status": "success",
        "metrics": {
            "illegal_area_m2": round(illegal_area_m2, 2),
            "legal_area_m2": round(legal_area_m2, 2),
            "volume_m3": round(illegal_vol_m3, 2),
            "total_vol_m3": round(total_vol_m3, 2),
            "avg_depth_m": round(avg_depth_m, 2),
            "truckloads": int(illegal_vol_m3 / 15)
        },
        "artifacts": {
            "map_url": map_filename,
            "model_url": tin_filename if total_area_m2 > 0 else None,
            "report_url": pdf_filename,
            "ai_mask_url": ai_mask_filename,      # <--- ADDED
            "ai_overlay_url": ai_overlay_filename # <--- ADDED
        }
    }