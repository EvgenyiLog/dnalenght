from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from readerfrf import parse_frf_file
import os
import glob

def categorize_frf_files(
    input_path: Union[str, Path, List[Union[str, Path]]],
    target_type: str = "Sample"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Категоризирует FRF файлы на две группы по типу файла (MATLAB-логика).
    
    MATLAB-логика: все файлы с Type == 'Sample' считаются GenLib файлами
    (в MATLAB к префиксу 'GenLib_' автоматически добавляется к заголовку)
    
    Parameters
    ----------
    input_path : Union[str, Path, List[Union[str, Path]]]
        Входной путь к:
        - одиночному FRF файлу
        - директории с FRF файлами (будут обработаны все .frf файлы рекурсивно)
        - списку путей к FRF файлам
    target_type : str, optional
        Тип файла для выделения в первую группу (по умолчанию "Sample")
    
    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        Кортеж из двух списков словарей:
        - Первый список: файлы с Type == target_type (GenLib/Sample файлы)
        - Второй список: файлы с другим типом (Ladder, Control, etc.)
        
        Каждый словарь содержит:
        - 'path': путь к исходному файлу (Path)
        - 'title': заголовок из метаданных (str)
        - 'type': тип файла из метаданных (str)
        - 'metadata': все метаданные файла (dict)
        - 'matrix_df': DataFrame с матрицей данных
        - 'channels_df': DataFrame с каналами
        - 'genlib_title': заголовок с добавленным префиксом 'GenLib_' (если type == 'Sample')
    
    Examples
    --------
    >>> # MATLAB-логика: разделяем по типу 'Sample'
    >>> genlib_files, other_files = categorize_frf_files_by_type("path/to/files")
    
    >>> # Можно искать другие типы
    >>> ladder_files, other_files = categorize_frf_files_by_type("path", target_type="Ladder")
    """
    
    def _collect_file_paths(path_input: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        """Внутренняя функция для сбора всех FRF файлов из различных входных типов"""
        file_paths = []
        
        if isinstance(path_input, (str, Path)):
            path = Path(path_input)
            
            if not path.exists():
                raise ValueError(f"Путь не существует: {path}")
            
            if path.is_file() and path.suffix.lower() == '.frf':
                file_paths = [path]
            elif path.is_dir():
                # Рекурсивный поиск всех .frf файлов
                file_paths = [Path(f) for f in glob.glob(str(path / "**" / "*.frf"), recursive=True)]
            else:
                raise TypeError(f"Неподдерживаемый тип пути: {path}")
                
        elif isinstance(path_input, list):
            for item in path_input:
                item_path = Path(item)
                if not item_path.exists():
                    print(f"Предупреждение: путь не существует, пропускаем: {item_path}")
                    continue
                if item_path.suffix.lower() == '.frf':
                    file_paths.append(item_path)
                else:
                    print(f"Предупреждение: файл не .frf, пропускаем: {item_path}")
        else:
            raise TypeError(f"Неподдерживаемый тип входных данных: {type(path_input)}")
        
        return file_paths
    
    # Собираем все пути к файлам
    frf_paths = _collect_file_paths(input_path)
    
    if not frf_paths:
        print("Предупреждение: FRF файлы не найдены")
        return [], []
    
    # Инициализируем группы
    type_files: List[Dict[str, Any]] = []  # Файлы target_type (Sample/GenLib)
    other_files: List[Dict[str, Any]] = []  # Все остальные файлы
    
    # Обрабатываем каждый файл
    for file_path in frf_paths:
        try:
            # Парсим FRF файл
            matrix_df, channels_df, metadata = parse_frf_file(str(file_path))
            title = str(metadata.get('Title', ''))
            file_type = str(metadata.get('Type', ''))  # Получаем тип файла
            
            # Создаем словарь с данными файла (как в MATLAB)
            file_data = {
                'path': file_path,
                'title': title,
                'type': file_type,
                'metadata': metadata,
                'matrix_df': matrix_df,
                'channels_df': channels_df,
                'genlib_title': f"GenLib_{title}" if file_type == "Sample" else title
            }
            
            # MATLAB-логика: если тип файла == 'Sample', то это GenLib файл
            if file_type == target_type:
                type_files.append(file_data)
                print(f"✓ {file_path.name} -> Группа '{target_type}' (Type: '{file_type}', Title: '{title}')")
            else:
                other_files.append(file_data)
                print(f"✓ {file_path.name} -> Другая группа (Type: '{file_type}', Title: '{title}')")
                
        except Exception as e:
            print(f"✗ Ошибка при обработке {file_path.name}: {e}")
            continue
    
    # Выводим итоговую статистику
    print(f"\n{'='*50}")
    print(f"ИТОГОВАЯ СТАТИСТИКА (MATLAB-логика):")
    print(f"Файлов с Type == '{target_type}': {len(type_files)}")
    print(f"Файлов с другим типом: {len(other_files)}")
    
    # Анализ всех встреченных типов
    all_types = {}
    for group in [type_files, other_files]:
        for file_data in group:
            file_type = file_data.get('type', 'Unknown')
            all_types[file_type] = all_types.get(file_type, 0) + 1
    
    if all_types:
        print(f"\nРаспределение по типам:")
        for t, count in sorted(all_types.items()):
            print(f"  {t}: {count} файлов")
    
    print(f"Всего обработано: {len(type_files) + len(other_files)} из {len(frf_paths)}")
    print(f"{'='*50}")
    
    return type_files, other_files


def categorize_frf_files_combined(
    input_path: Union[str, Path, List[Union[str, Path]]],
    method: str = "auto"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Универсальная функция категоризации с разными стратегиями.
    
    Parameters
    ----------
    input_path : Union[str, Path, List[Union[str, Path]]]
        Входной путь к файлам
    method : str, optional
        Метод категоризации:
        - 'type': по Type == 'Sample' (MATLAB-логика)
        - 'title': по наличию 'GenLib_' в заголовке (оригинальная логика)
        - 'auto': автоматически выбирает метод на основе данных
    
    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        Группа GenLib и группа других файлов
    """
    
    # Сначала собираем информацию о файлах
    def _collect_file_info(path_input):
        """Собирает базовую информацию о файлах"""
        from pathlib import Path
        import glob
        
        file_paths = []
        if isinstance(path_input, (str, Path)):
            path = Path(path_input)
            if path.is_file():
                file_paths = [path]
            elif path.is_dir():
                file_paths = [Path(f) for f in glob.glob(str(path / "**" / "*.frf"), recursive=True)]
        elif isinstance(path_input, list):
            file_paths = [Path(p) for p in path_input if Path(p).exists()]
        
        return file_paths
    
    file_paths = _collect_file_info(input_path)
    
    if not file_paths:
        return [], []
    
    # Пробуем проанализировать первые несколько файлов для выбора метода
    sample_files = min(3, len(file_paths))
    has_genlib_in_title = False
    has_sample_type = False
    
    for i in range(sample_files):
        try:
            _, _, metadata = parse_frf_file(str(file_paths[i]))
            title = str(metadata.get('Title', ''))
            file_type = str(metadata.get('Type', ''))
            
            if 'GenLib_' in title:
                has_genlib_in_title = True
            if file_type == 'Sample':
                has_sample_type = True
                
            print(f"Тестовый файл {file_paths[i].name}:")
            print(f"  Title: '{title}'")
            print(f"  Type: '{file_type}'")
            
        except:
            continue
    
    # Автоматический выбор метода
    if method == 'auto':
        if has_sample_type:
            print("\nОбнаружен Type='Sample' в файлах -> используем MATLAB-логику (по типу)")
            method = 'type'
        elif has_genlib_in_title:
            print("\nОбнаружен 'GenLib_' в заголовках -> используем логику по заголовку")
            method = 'title'
        else:
            print("\nНе удалось определить метод, используем MATLAB-логику по умолчанию")
            method = 'type'
    
    # Выполняем категоризацию выбранным методом
    if method == 'type':
        return categorize_frf_files_by_type(input_path, target_type="Sample")
    elif method == 'title':
        # Используем оригинальную функцию с поиском по заголовку
        return categorize_frf_files_by_title(input_path, title_keyword="GenLib_")
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def categorize_frf_files_by_title(
    input_path: Union[str, Path, List[Union[str, Path]]],
    title_keyword: str = "GenLib_"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Оригинальная версия функции для поиска по заголовку"""
    # (используйте ваш исходный код с поиском по title_keyword)
    pass


# Пример использования с MATLAB-логикой:
if __name__ == "__main__":
    # Способ 1: Чистая MATLAB-логика (по типу)
    print("=" * 60)
    print("MATLAB-ЛОГИКА: Разделение по Type == 'Sample'")
    print("=" * 60)
    
    genlib_files, other_files = categorize_frf_files_by_type(
        input_path="путь/к/вашей/директории",
        target_type="Sample"  # Это значение из MATLAB кода
    )
    
    # Пример: что делать с результатами
    print(f"\nНайдено GenLib файлов (Type='Sample'): {len(genlib_files)}")
    for file_data in genlib_files[:5]:  # Показать первые 5
        print(f"  - {file_data['path'].name}: '{file_data['title']}' -> '{file_data['genlib_title']}'")
    
    print(f"\nДругих файлов: {len(other_files)}")
    for file_data in other_files[:5]:
        print(f"  - {file_data['path'].name}: Type='{file_data['type']}'")
    
    # Способ 2: Универсальный подход с автоопределением
    print("\n" + "=" * 60)
    print("УНИВЕРСАЛЬНЫЙ ПОДХОД с автоопределением")
    print("=" * 60)
    
    genlib, other = categorize_frf_files_combined(
        input_path="путь/к/вашей/директории",
        method="auto"
    )


