from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
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

# Setup CORS & Directories
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return {"status": "MineGuard System v2.0 Online"}

@app.post("/api/analyze")
async def analyze_mining_site(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Analyzes the file AND saves the result to the PostgreSQL Database.
    """
    job_id = str(uuid.uuid4())[:8]
    user_filename = file.filename
    safe_filename = f"{job_id}_{user_filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    print(f"📥 Processing Job: {job_id}")

    # 1. Save & Process File
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        lease_geojson = process_lease_file(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File Error: {e}")

    # 2. Run AI Engine
    try:
        job_output_dir = os.path.join(OUTPUT_DIR, job_id)
        
        result = run_unified_detection(
            lease_geojson, 
            filename=user_filename,
            output_dir=job_output_dir
        )
        
        metrics = result["metrics"]
        artifacts = result.get("artifacts", {})
        
        # 3. SAVE TO DATABASE
        base_url = "http://localhost:8000"
        
        new_inspection = Inspection(
            job_id=job_id,
            filename=user_filename,
            illegal_area_m2=metrics["illegal_area_m2"],
            volume_m3=metrics["volume_m3"],
            avg_depth_m=metrics["avg_depth_m"],
            truckloads=metrics["truckloads"],
            status=result["status"],
            report_url=f"{base_url}/static/outputs/{job_id}/{artifacts.get('report_url')}",
            map_url=f"{base_url}/static/outputs/{job_id}/{artifacts.get('map_url')}",
            model_url=f"{base_url}/static/outputs/{job_id}/{artifacts.get('model_url')}" if artifacts.get("model_url") else None
        )
        
        db.add(new_inspection)
        db.commit()
        db.refresh(new_inspection)
        print(f"💾 Saved to Database: Inspection ID {new_inspection.id}")

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")

@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    """Fetch past inspections for the Dashboard."""
    return db.query(Inspection).order_by(Inspection.created_at.desc()).all()