import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import glob

from readerfrf import parse_frf_file


# ======================================================================
# СБОР FRF ФАЙЛОВ
# ======================================================================

def _collect_file_paths(
    path_input: Union[str, Path, List[Union[str, Path]]]
) -> List[Path]:
    """
    Собирает все .frf файлы:
    - одиночный файл
    - директория (рекурсивно)
    - список путей
    """
    file_paths: List[Path] = []

    if isinstance(path_input, (str, Path)):
        path = Path(path_input)

        if not path.exists():
            raise ValueError(f"Путь не существует: {path}")

        if path.is_file() and path.suffix.lower() == ".frf":
            file_paths = [path]

        elif path.is_dir():
            file_paths = [
                Path(p) for p in glob.glob(str(path / "**" / "*.frf"), recursive=True)
            ]

        else:
            raise TypeError(f"Неподдерживаемый путь: {path}")

    elif isinstance(path_input, list):
        for item in path_input:
            p = Path(item)
            if p.exists() and p.suffix.lower() == ".frf":
                file_paths.append(p)
            else:
                print(f"⚠️ Пропущен: {p}")

    else:
        raise TypeError(f"Неподдерживаемый тип входа: {type(path_input)}")

    return file_paths


# ======================================================================
# СИНХРОННАЯ ОБРАБОТКА ОДНОГО ФАЙЛА (WORKER)
# ======================================================================

def _process_single_frf(
    file_path: Path,
    target_type: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Обработка одного FRF файла.
    Вызывается внутри ThreadPoolExecutor.
    """
    matrix_df, channels_df, metadata = parse_frf_file(str(file_path))

    title = str(metadata.get("Title", ""))
    file_type = str(metadata.get("Type", ""))

    file_data = {
        "path": file_path,
        "title": title,
        "type": file_type,
        "metadata": metadata,
        "matrix_df": matrix_df,
        "channels_df": channels_df,
        "genlib_title": f"GenLib_{title}" if file_type == target_type else title,
    }

    return file_type == target_type, file_data


# ======================================================================
# АСИНХРОННАЯ КАТЕГОРИЗАЦИЯ
# ======================================================================

async def categorize_frf_files_async(
    input_path: Union[str, Path, List[Union[str, Path]]],
    target_type: str = "Sample",
    max_workers: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Асинхронная версия категоризации FRF файлов
    (MATLAB-логика: Type == 'Sample' → GenLib)
    """

    frf_paths = _collect_file_paths(input_path)

    if not frf_paths:
        print("⚠️ FRF файлы не найдены")
        return [], []

    type_files: List[Dict[str, Any]] = []
    other_files: List[Dict[str, Any]] = []

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                _process_single_frf,
                file_path,
                target_type,
            )
            for file_path in frf_paths
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                is_target, file_data = await coro
                if is_target:
                    type_files.append(file_data)
                else:
                    other_files.append(file_data)
            except Exception as e:
                print(f"✗ Ошибка обработки файла: {e}")

    return type_files, other_files


# ======================================================================
# СИНХРОННЫЙ WRAPPER (ЧТОБЫ ВЫЗЫВАТЬ КАК ОБЫЧНО)
# ======================================================================

def categorize_frf_files_by_type_async(
    input_path: Union[str, Path, List[Union[str, Path]]],
    target_type: str = "Sample",
    max_workers: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Синхронный интерфейс к async-версии.
    """
    return asyncio.run(
        categorize_frf_files_async(
            input_path=input_path,
            target_type=target_type,
            max_workers=max_workers,
        )
    )


# ======================================================================
# АЛИАС ДЛЯ СОВМЕСТИМОСТИ С ВАШИМ КОДОМ
# ======================================================================

categorize_frf_files_by_type = categorize_frf_files_by_type_async


# ======================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ======================================================================

if __name__ == "__main__":

    genlib_files, other_files = categorize_frf_files_by_type(
        input_path="путь/к/вашей/директории",
        target_type="Sample",
        max_workers=8,   # подбирайте под CPU / диск
    )

    print(f"\nGenLib файлов (Type='Sample'): {len(genlib_files)}")
    for f in genlib_files[:5]:
        print(f"  ✓ {f['path'].name} → {f['genlib_title']}")

    print(f"\nДругих файлов: {len(other_files)}")
    for f in other_files[:5]:
        print(f"  • {f['path'].name} (Type='{f['type']}')")
