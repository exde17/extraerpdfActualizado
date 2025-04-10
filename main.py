

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
from pdf_processor import process_pdf, process_pdf_with_tables

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)

app = FastAPI(title="Conversor de PDF a Texto con Soporte de Tablas")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

@app.post("/convert")
async def convert_pdf(file: UploadFile = File(...), extract_tables: bool = Form(False), background_tasks: BackgroundTasks = None):
    
    # Validate file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")
    
    try:
        # Create temporary files for the PDF and resulting text
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            # Save the uploaded PDF
            content = await file.read()
            temp_pdf.write(content)
            pdf_path = temp_pdf.name
        
        # Create temporary file for the text output
        txt_filename = os.path.splitext(file.filename)[0] + '.txt'
        txt_path = os.path.join(tempfile.gettempdir(), txt_filename)
        
        # Process the PDF based on whether tables should be extracted
        if extract_tables:
            # Process with table extraction
            tables_json_filename = os.path.splitext(file.filename)[0] + '_tables.json'
            tables_json_path = os.path.join(tempfile.gettempdir(), tables_json_filename)
            
            process_pdf_with_tables(pdf_path, txt_path, tables_json_path)
            
            # Add cleanup tasks
            background_tasks.add_task(cleanup_temp_files, [pdf_path, txt_path, tables_json_path])
            
            # Return both text and tables JSON paths
            return JSONResponse(content={
                "text_content": txt_path,
                "tables_content": tables_json_path,
                "filename_base": os.path.splitext(file.filename)[0]
            })
        else:
            # Process without table extraction
            process_pdf(pdf_path, txt_path)
            
            # Add cleanup tasks
            background_tasks.add_task(cleanup_temp_files, [pdf_path, txt_path])
            
            # Return the text file as response
            return FileResponse(
                path=txt_path,
                filename=txt_filename,
                media_type="text/plain"
            )
    
    except Exception as e:
        # Clean up in case of error
        cleanup_files = []
        if 'pdf_path' in locals():
            cleanup_files.append(pdf_path)
        if 'txt_path' in locals():
            cleanup_files.append(txt_path)
        if 'tables_json_path' in locals():
            cleanup_files.append(tables_json_path)
            
        cleanup_temp_files(cleanup_files)
        raise HTTPException(status_code=500, detail=f"Error al procesar el PDF: {str(e)}")

def cleanup_temp_files(file_paths):
    """
    Limpia los archivos temporales después de enviar la respuesta
    """
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
