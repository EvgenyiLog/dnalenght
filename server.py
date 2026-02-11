import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Any, Union, Dict, List
import tempfile
from score_peaks import score_peaks_genlib
from correct_baseline import correct_baseline

# Импорт версии
from version import _version_,_release_date_

# Импорт ваших функций (убедитесь, что файлы лежат в этой же папке)
from readerfrf import parse_frf_file
from subtract_reference_from_columns import subtract_reference_from_columns
from msbackadj import msbackadj
from categorize_frf_files import categorize_frf_files
from reveal_paths import reveal_paths, extract_paths_from_categorize
from convert_numpy_types import convert_numpy_types
from find_numpy_types import find_numpy_types
from score_peaks import score_peaks_genlib

app = FastAPI(
    title="DNA Length Signal Processor",
    version=_version_,
    release_date=_release_date_
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # Разрешить запросы отовсюду
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/scan-folder/", summary="Сканировать папку и вернуть список файлов")
async def scan_folder(folder_path: str = Form(...)):
    """
    Принимает путь к папке, сканирует её и возвращает списки файлов:
    - keyword_files: файлы с ключевыми словами (ref, reference)
    - other_files: остальные .frf файлы
    """
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail=f"Папка не найдена: {folder_path}")
    
    # Получаем все .frf файлы
    all_frf = [f for f in os.listdir(folder_path) if f.lower().endswith('.frf')]

    # Категоризация (ваша логика)
    keyword_files, other_files = categorize_frf_files(input_path=folder_path)
    keyword_paths, other_paths = extract_paths_from_categorize(keyword_files, other_files)
    path_keyword_files = reveal_paths(keyword_paths)
    path_other_files = reveal_paths(other_paths)
    
    print(path_other_files)

    result = {
        "folder": folder_path,
        "keyword_files": keyword_files,
        "other_files": other_files,
        "path_keyword_files": path_keyword_files,
        "path_other_files": path_other_files,
    }
    
    print("\n=== DEBUG: Searching for numpy types ===")
    find_numpy_types(result)
    print("=========================================\n")

    return JSONResponse(content=convert_numpy_types(result))

@app.post("/process-by-path/")
async def process_by_path(full_path: str = Form(...)):
    print("RAW full_path:", repr(full_path))
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Файл не найден")

    try:
        matrix_df, channels_df, metadata = parse_frf_file(full_path)
        df_processed = subtract_reference_from_columns(channels_df, 50)
        signal_raw = df_processed['dR110'].values
        time = np.arange(len(signal_raw))
        signal_corrected = msbackadj(time, signal_raw)
        


        result = {
            "title": metadata.get('Title', os.path.basename(full_path)),
            "labels": time.tolist(),
            "raw": signal_raw.tolist(),
            "corrected": signal_corrected.tolist()
        }
        print("✅ Ответ сервера (первые 5 точек):", {
            "title": result["title"],
            "labels_len": len(result["labels"]),
            "raw_len": len(result["raw"]),
            "corrected_len": len(result["corrected"])
        })
        return result
    except Exception as e:
        print("❌ Ошибка обработки:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-pair/", summary="Обработка двух файлов для анализа")
async def analyze_pair(
    top_path: str = Form(None),
    bottom_path: str = Form(None)
):
    """
    Берёт два пути к файлам, обрабатывает и возвращает:
    - top: график верхний (raw, corrected, fill, vlines, hlines)
    - bottom: график нижний (raw, corrected, fill, vlines, hlines)
    - extra_top / extra_bottom: график только линия и точки (для PDF)
    - table: DataFrame с колонками top_raw, top_corrected, bottom_raw, bottom_corrected
    """
    if not top_path and not bottom_path:
        raise HTTPException(status_code=400,detail='Нужно указать хотя бы один файл')
    def process_file(path, include_fill=True, include_lines=True, extra=False):
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"Файл не найден: {path}")
        try:
            matrix_df, channels_df, metadata = parse_frf_file(path)
            df_proc = subtract_reference_from_columns(channels_df, 50)
            signal_raw = df_proc['dR110'].values
            time = np.arange(len(signal_raw))
            signal_corrected = msbackadj(time, signal_raw)

            result = {
                "title": metadata.get("Title", os.path.basename(path)),
                "labels": time.tolist(),
                "raw": signal_raw.tolist(),
                "corrected": signal_corrected.tolist()
            }

            if include_fill:
                upper = signal_corrected + 0.05
                lower = signal_corrected - 0.05
                result["fill_upper"] = upper.tolist()
                result["fill_lower"] = lower.tolist()

            if include_lines:
                vlines = [len(time)//4, len(time)//2, 3*len(time)//4]
                hlines = [np.mean(signal_corrected)]
                result["vlines"] = vlines
                result["hlines"] = hlines

            if extra:
                result_extra = {
                    "title": metadata.get("Title", os.path.basename(path)) + " Extra",
                    "labels": time.tolist(),
                    "raw": signal_raw.tolist(),
                    "corrected": signal_corrected.tolist()
                }
                return result, result_extra

            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка обработки {path}: {str(e)}")
    if top_path:
        top_data, top_extra = process_file(top_path, include_fill=True, include_lines=True, extra=True)
    if bottom_path:
        bottom_data, bottom_extra = process_file(bottom_path, include_fill=True, include_lines=True, extra=True)
    if bottom_path:
        # Формируем DataFrame для таблицы (все колонки)
        matrix_df, channels_df, metadata = parse_frf_file(bottom_path)
        df_proc = subtract_reference_from_columns(channels_df, 50)
        signal_raw = df_proc['dR110'].values
        time = np.arange(len(signal_raw))
        signal_corrected = msbackadj(time, signal_raw)
        df_table = score_peaks_genlib(signal_corrected)
        signal_for_peaks=signal_corrected
        peaks=df_table['Index'].astype(int).tolist()
        points=df_table['Mark'].astype(float).tolist()
        selected_indices=df_table[df_table['Selected'] == '✓']['Index'].astype(int).tolist()
        botom_peaks=[]
        for _,row in df_table[df_table['Selected'] == '✓'].iterrows():
            idx=int(row['Index'])
            y_val=float(signal_corrected[idx])
            botom_peaks.append({"x":idx,"y":y_val})
        print(botom_peaks)
        print(peaks)
        print(selected_indices)
       
    
   

    return JSONResponse({
        "top": top_data,
        "bottom": bottom_data,
        "extra_top": top_extra,
        "extra_bottom": bottom_extra,
        "table": df_table.to_dict(orient="records"),
        "bottom_peaks":botom_peaks
           
       
    })





@app.post("/process-frf/", summary="Загрузить .frf и получить график")
async def process_frf(file: UploadFile = File(...)):
    # 1. Проверяем расширение
    if not file.filename.endswith('.frf'):
        raise HTTPException(status_code=400, detail="Файл должен быть формата .frf")
    
    # Формируем путь для временного файла
    temp_path = os.path.join(
        os.environ.get('TEMP', tempfile.gettempdir()),
        f"frf_{os.urandom(4).hex()}_{file.filename}"
    )

    try:
        # 2. Сохраняем загруженные байты в реальный файл
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 3. Вызываем ваш парсер
        matrix_df, channels_df, metadata = parse_frf_file(temp_path)
        
        # 4. Обработка сигнала
        df_processed = subtract_reference_from_columns(channels_df, 50)
        signal_raw = df_processed['dR110'].values
        time = np.arange(len(signal_raw))
        signal_corrected = msbackadj(time, signal_raw)
        corectbaseline,baseline=correct_baseline(signal_corrected, method='modpoly',poly_order=1)
        corectbaseline,baseline=correct_baseline(corectbaseline,   method='psalsa',lam=1e5,k=0.05 )
    
        corectbaseline,baseline=correct_baseline(corectbaseline,method='beads',freq_cutoff=0.002,asymmetry=3,lam_0=3,lam_1=0.05,lam_2=0.2)
        corectbaseline,baseline=correct_baseline(corectbaseline, method='iasls',lam=1e6)
        signal_corrected=corectbaseline
        
        df_processed['dR110_corr'] = signal_corrected

        time_labels = np.arange(len(signal_raw)).tolist()
        
        chart_data = {
            "title": metadata.get('Title', 'Без названия'),
            "labels": time_labels,
            "datasets": [
                {
                    "label": "После subtract_reference",
                    "data": signal_raw.tolist(),
                    "borderColor": "rgba(255, 0, 255, 0.4)",
                    "borderWidth": 1,
                    "fill": False
                },
                {
                    "label": "После msbackadj (Итог)",
                    "data": signal_corrected.tolist(),
                    "borderColor": "rgba(0, 128, 0, 1)",
                    "borderWidth": 2,
                    "fill": False
                }
            ]
        }

        return chart_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

    finally:
        # ГАРАНТИРОВАННО удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/info")
def info():
    return {
        "status": "Online",
        "version": _version_,
        "docs": "/docs"
    }

@app.get("/app/static")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/api/health", summary="Проверка работоспособности")
def health_check():
    return {"status": "healthy"}

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def redirect_to_app():
    return "/app/static"

# Монтируем папку static
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=True)