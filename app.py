from llm_services.bot import tutor,quiz
from llm_services.outline import create_outline
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import tempfile
import os
import shutil
import asyncio
from loaders.multiple_file import load_directory
app = FastAPI()
MAX_TOTAL_SIZE = 50 * 1024 * 1024 

# query ="""topic: Algebra of Complex Numbers ,subtopic : Multiplication"""
# tutor(query)
# # create_outline()
# from loaders.multiple_file import save_directory
# save_directory('./documents')

import json

def clean_and_parse_json(ai_response_text):
    # 1. Remove the "```json" from the start
    clean_text = ai_response_text.replace("```json", "")
    
    # 2. Remove the "```" from the end
    clean_text = clean_text.replace("```", "")
    
    # 3. Strip leading/trailing whitespace
    clean_text = clean_text.strip()
    
    # 4. Parse
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        return None

# usage:
# raw_ai_response = client.chat.completions.create(...)
# lesson_data = clean_and_parse_json(raw_ai_response)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}
from pydantic import BaseModel
class Query(BaseModel):
    text: str
    adapt: str



@app.get("/quizes")

async def quizes():
    cards = await quiz() 
    return clean_and_parse_json(cards)


@app.post("/tutor")
async def tutor_endpoint(payload: Query):
    # query ="""topic: Algebra of Complex Numbers ,subtopic : Multiplication"""
    data = await tutor(payload.text,payload.adapt)
    return clean_and_parse_json(data)
    # return {"you_sent": payload.text}

@app.get('/outline')    
async def outline():
    data = await create_outline()
    return data


@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="uploaded_pdfs_")
    total_size = 0
    saved_files = []

    try:
        for file in files:
            if file.content_type != "application/pdf":
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' is not a PDF."
                )
            contents = await file.read()
            total_size += len(contents)

            if total_size > MAX_TOTAL_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail="Total upload size exceeds 50 MB."
                )

            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(contents)
            saved_files.append(file.filename)

        # Pass the temp_dir and saved_files to your async processing function
        result = await load(temp_dir)

        return {
            "temp_directory": temp_dir,
            "uploaded_files": saved_files,
            "total_size_bytes": total_size,
            "processing_result": result
        }

    finally:
        # Clean up temp directory after processing
        shutil.rmtree(temp_dir, ignore_errors=True)