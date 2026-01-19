from pathlib import Path
from typing import List
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

def extract_paths_from_categorize(frf_type_files: List[Dict[str, Any]], 
                                 other_files: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Извлекает списки путей из результата categorize_frf_files().
    
    Args:
        frf_type_files: первый результат categorize_frf_files() (Sample/GenLib)
        other_files: второй результат categorize_frf_files()
    
    Returns:
        Tuple[List[str], List[str]]: списки строковых путей
    """
    # Извлекаем пути из словарей
    keyword_paths = [str(item['path']) for item in frf_type_files]
    other_paths = [str(item['path']) for item in other_files]
    
    print(f"🔑 Keyword файлов: {len(keyword_paths)}")
    print(f"📋 Other файлов: {len(other_paths)}")
    
    return keyword_paths, other_paths

def reveal_paths(raw_paths: List[Union[str, Path, Dict]]) -> List[str]:
    """
    Универсальная версия: работает с list[str], list[Path], list[dict].
    """
    all_paths = []
    
    for item in raw_paths:
        if isinstance(item, dict):
            # Извлекаем 'path' из словаря (categorize_frf_files)
            path_str = str(item.get('path', item.get('file_path', '')))
        elif isinstance(item, (str, Path)):
            path_str = str(item)
        else:
            path_str = str(item)
        
        # Проверяем FRF и существование
        path = Path(path_str.strip())
        if path.suffix.lower() == '.frf' and path.exists():
            full_path = path.absolute()
            all_paths.append(str(full_path))
            print(f"✅ {full_path}")
    
    return sorted(list(set(all_paths)))




