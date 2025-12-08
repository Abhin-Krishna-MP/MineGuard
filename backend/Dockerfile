# Use a lightweight Python image
FROM python:3.10-slim

# 1. Install System Dependencies 
# Added 'libgl1' just in case OpenCV needs graphics drivers, 
# and 'build-essential' for compiling complex python libs.
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    g++ \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install Python Dependencies
# We increase the timeout because PyTorch is a large download
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 1000

# 3. Copy Application Code
# This command copies 'mineguard_unet.pth' IF it is in the backend folder
COPY . .

# 4. Expose the Port
EXPOSE 8000

# 5. Initialize DB and Start Server
CMD sh -c "python init_db.py && uvicorn server:app --host 0.0.0.0 --port 8000"