import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException, Form 
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Импорт версии
from version import _version_

# Импорт ваших функций (убедитесь, что файлы лежат в этой же папке)
from readerfrf import parse_frf_file
from subtract_reference_from_columns import subtract_reference_from_columns
from msbackadj import msbackadj
from categorize_frf_files import categorize_frf_files

from reveal_paths import reveal_paths,extract_paths_from_categorize

import tempfile

app = FastAPI(
    title="DNA Length Signal Processor",
    version=_version_
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # Разрешить запросы отовсюду
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-frf/", summary="Загрузить .frf и получить график")
async def process_frf(file: UploadFile = File(...)):
    # 1. Проверяем расширение
    if not file.filename.endswith('.frf'):
        raise HTTPException(status_code=400, detail="Файл должен быть формата .frf")

    # Формируем путь для временного файла
    # temp_path = f"temp_{file.filename}"
    temp_path = os.path.join(
        os.environ.get('TEMP', tempfile.gettempdir()),
        f"frf_{os.urandom(4).hex()}_{file.filename}"
    )

    try:
        # 2. Сохраняем загруженные байты в реальный файл
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 3. Вызываем ваш парсер (теперь он видит файл по пути temp_path)
        # Мы берем вторую попытку из вашего кода, где данные перезаписывались
        matrix_df, channels_df, metadata = parse_frf_file(temp_path)
        
        # 4. Обработка сигнала (ваша логика)
        # Вычитание референса
        df_processed = subtract_reference_from_columns(channels_df, 50)
        
        # Коррекция базовой линии
        time = np.arange(len(df_processed))
        signal_raw = df_processed['dR110'].values
        signal_corrected = msbackadj(time, signal_raw)
        
        # Добавляем коррекцию в DataFrame (как в вашем исходнике)
        df_processed['dR110_corr'] = signal_corrected

       
    
        time_labels = np.arange(len(signal_raw)).tolist()
        
        chart_data = {
            "title": metadata.get('Title', 'Без названия'),
            "labels": time_labels, # Ось X
            "datasets": [
                {
                    "label": "После subtract_reference",
                    "data": signal_raw.tolist(),
                    "borderColor": "rgba(255, 0, 255, 0.4)", # Цвет 'm' (magenta)
                    "borderWidth": 1,
                    "fill": False
                },
                {
                    "label": "После msbackadj (Итог)",
                    "data": signal_corrected.tolist(),
                    "borderColor": "rgba(0, 128, 0, 1)", # Цвет 'g' (green)
                    "borderWidth": 2,
                    "fill": False
                }
            ]
        }

        # 6. Отправляем словарь (FastAPI автоматически конвертирует его в JSON)
        return chart_data

    except Exception as e:
        # Если что-то пошло не так (например, нет колонки dR110)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    
    finally:
        # 7. ГАРАНТИРОВАННО удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)

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
    keyword_files, other_files=categorize_frf_files(input_path=folder_path)
    keyword_paths, other_paths = extract_paths_from_categorize(keyword_files, other_files)
    path_keyword_files = reveal_paths(keyword_paths)
    path_other_files = reveal_paths(other_paths)
   
    
    return {
        "folder": folder_path,
        "keyword_files": keyword_files,
        "other_files": other_files,
        'path_keyword_files':path_keyword_files,
        'path_other_files': path_other_files,
        "total_count": len(all_frf)
    }

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

@app.get("/home", response_class=RedirectResponse, include_in_schema=False)
async def redirect_to_app():
    """
    Редирект с корня на веб-приложение
    Не показывается в документации (include_in_schema=False)
    """
    return "/app/static"

# 4. Монтируем папку static для доступа к другим файлам (если будут стили или скрипты)
app.mount("/static", StaticFiles(directory="static"), name="static")
if __name__ == "__main__":
    import uvicorn
    # Запуск сервера на порту 8000
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=True)