import sys
import ee
import os
from file_processor import process_lease_file
from phase1_detection import run_unified_detection

def main():
    print("\n🛰️  INITIALIZING MINEGUARD SYSTEM v2.0 (Unified) 🛰️")
    print("---------------------------------------------------")
    
    target_file = input("👉 Enter path to Shapefile (.zip) or GeoJSON: ").strip()
    target_file = target_file.replace('"', '').replace("'", "")

    if not target_file:
        print("❌ Error: No file provided.")
        return

    print("⏳ Reading Lease Boundary...")
    lease_geojson = process_lease_file(target_file)
    
    if not lease_geojson:
        print("❌ Failed to read boundary. Exiting.")
        return

    print("✅ Boundary loaded successfully!")
    print("🚀 Initializing Unified Detection Pipeline...")
    
    # FIX: Pass the filename explicitly
    run_unified_detection(lease_geojson, filename=target_file)

if __name__ == "__main__":
    main()