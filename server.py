import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# Импорт версии
from version import _version_

# Импорт ваших функций (убедитесь, что файлы лежат в этой же папке)
from readerfrf import parse_frf_file
from subtract_reference_from_columns import subtract_reference_from_columns
from msbackadj import msbackadj

app = FastAPI(
    title="DNA Length Signal Processor",
    version=_version_
)

@app.post("/process-frf/", summary="Загрузить .frf и получить график")
async def process_frf(file: UploadFile = File(...)):
    # 1. Проверяем расширение
    if not file.filename.endswith('.frf'):
        raise HTTPException(status_code=400, detail="Файл должен быть формата .frf")

    # Формируем путь для временного файла
    temp_path = f"temp_{file.filename}"

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

        # 5. Отрисовка графика в память
        plt.figure(figsize=(10, 5))
        plt.plot(signal_raw, color='m', alpha=0.4, label='После subtract_reference')
        plt.plot(signal_corrected, color='g', label='После msbackadj (Итог)')
        plt.title(f"Сигнал: {metadata.get('Title', 'Без названия')}")
        plt.xlabel("Отсчеты")
        plt.ylabel("Амплитуда")
        plt.legend()
        plt.grid(True)

        # Сохраняем в байтовый буфер вместо файла
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close() # Важно закрыть график, чтобы не копился в памяти

        # 6. Отправляем картинку пользователю
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        # Если что-то пошло не так (например, нет колонки dR110)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    
    finally:
        # 7. ГАРАНТИРОВАННО удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def home():
    return {
        "status": "Online",
        "version": _version_,
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # Запуск сервера на порту 8000
    uvicorn.run(app, host="180.0.0.0", port=8000)