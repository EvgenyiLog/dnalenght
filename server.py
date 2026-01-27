from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import io
import traceback

# Импортируем ваши существующие функции
from analyze_peaks import analyze_peaks  # ← ваш файл
from analyze_single_spectrum import analyze_spectrum  # ← ваш файл

app = FastAPI(title="nd-forez Signal Processor", version="0.1.0")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Server is running"}

@app.post("/analyze/peaks")
async def analyze_peaks_endpoint(file: UploadFile = File(...)):
    """Анализ пиков из загруженного файла"""
    try:
        content = await file.read()
        # Предполагаем, что ваша функция принимает bytes или numpy array
        # Адаптируйте под ваш формат входных данных
        data = np.loadtxt(io.BytesIO(content), delimiter=',')  # пример для CSV
        
        result = analyze_peaks(data)  # ← вызов вашей функции
        
        return JSONResponse(content={"success": True, "result": result})
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )

@app.post("/analyze/spectrum")
async def analyze_spectrum_endpoint(file: UploadFile = File(...)):
    """Анализ спектра"""
    try:
        content = await file.read()
        data = np.loadtxt(io.BytesIO(content), delimiter=',')
        
        result = analyze_spectrum(data)  # ← ваша функция
        
        return JSONResponse(content={"success": True, "result": result})
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)