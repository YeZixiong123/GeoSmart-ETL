from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import shutil
import os
import json
import glob
from data_loader_v3 import ForestDataProcessor
from s3_client_v2 import S3HybridClient
from ai_agent import DataInsightAgent

app = FastAPI(title="GeoSmart-ETL Platform (Pro)", version="6.0.0")

UPLOAD_DIR, PROCESSED_DIR = "uploads", "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True); os.makedirs(PROCESSED_DIR, exist_ok=True)

s3_uploader = S3HybridClient()
ai_agent = DataInsightAgent()

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def read_index(): return FileResponse("index.html")

@app.post("/analyze")
async def analyze_forest_data(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    try:
        with open(file_location, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        filename_no_ext = os.path.splitext(file.filename)[0]
        output_parquet = f"{PROCESSED_DIR}/{filename_no_ext}_cleaned.parquet"
        output_profile = f"{PROCESSED_DIR}/{filename_no_ext}_profile.json"
        processor = ForestDataProcessor(raw_path=file_location)
        processor.process(output_parquet_path=output_parquet, output_profile_path=output_profile)
        upload_result = s3_uploader.upload_file(output_parquet)
        with open(output_profile, "r") as f: profile_data = json.load(f)
        return {"status": "success", "storage_info": upload_result, "ai_insight_source": profile_data}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_data(request: ChatRequest):
    try:
        list_of_files = glob.glob(f"{PROCESSED_DIR}/*_profile.json")
        if not list_of_files: return JSONResponse(status_code=404, content={"message": "No data analyzed."})
        latest_profile = max(list_of_files, key=os.path.getctime)
        result = ai_agent.generate_insight(latest_profile, request.query)
        return result 
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)