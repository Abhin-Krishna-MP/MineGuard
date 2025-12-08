from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from ai_inference import MineSegmenter

import shutil
import os
import uuid

# Import Engines
from file_processor import process_lease_file
from phase1_detection import run_unified_detection

# Import Database
from database import get_db
from models import Inspection

app = FastAPI(title="MineGuard Enterprise API")

@app.on_event("startup")
async def startup_event():
    print("🧠 Initializing Neural Network...")
    MineSegmenter()

# --- CONFIGURATION ---
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", "http://localhost:8000")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Directories
UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return {"status": "MineGuard System v2.0 Online", "public_url": API_PUBLIC_URL}

@app.post("/api/analyze")
async def analyze_mining_site(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    start_date: str = Form("2024-01-01"),
    end_date: str = Form("2024-04-30")
):
    """
    Analyzes the file AND saves the result to the PostgreSQL Database.
    """
    job_id = str(uuid.uuid4())[:8]
    user_filename = file.filename
    # Sanitize filename
    safe_filename = f"{job_id}_{user_filename.replace(' ', '_')}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    print(f"📥 Processing Job: {job_id} | File: {user_filename}")

    # 1. Save & Process File
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        lease_geojson = process_lease_file(file_path)
    except Exception as e:
        print(f"❌ File Error: {e}")
        raise HTTPException(status_code=500, detail=f"File Error: {str(e)}")

    # 2. Run AI Engine
    try:
        job_output_dir = os.path.join(OUTPUT_DIR, job_id)
        
        # Now calls the updated Unified Detection Logic
        result = run_unified_detection(
            lease_geojson, 
            filename=user_filename,
            output_dir=job_output_dir,
            start_date=start_date,
            end_date=end_date
        )
        
        metrics = result["metrics"]
        artifacts = result.get("artifacts", {})
        
        # 3. SAVE TO DATABASE
        url_report = f"{API_PUBLIC_URL}/static/outputs/{job_id}/{artifacts.get('report_url')}"
        url_map = f"{API_PUBLIC_URL}/static/outputs/{job_id}/{artifacts.get('map_url')}"
        url_model = None
        if artifacts.get('model_url'):
            url_model = f"{API_PUBLIC_URL}/static/outputs/{job_id}/{artifacts.get('model_url')}"
        
        new_inspection = Inspection(
            job_id=job_id,
            filename=user_filename,
            illegal_area_m2=metrics["illegal_area_m2"],
            volume_m3=metrics["volume_m3"],
            avg_depth_m=metrics["avg_depth_m"],
            truckloads=metrics["truckloads"],
            status=result["status"],
            report_url=url_report,
            map_url=url_map,
            model_url=url_model
        )
        
        db.add(new_inspection)
        db.commit()
        db.refresh(new_inspection)
        print(f"💾 Saved to Database: Inspection ID {new_inspection.id}")

        result["urls"] = {
            "report": url_report,
            "map": url_map,
            "3d_model": url_model
        }

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")

@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    """Fetch past inspections for the Dashboard."""
    return db.query(Inspection).order_by(Inspection.created_at.desc()).all()