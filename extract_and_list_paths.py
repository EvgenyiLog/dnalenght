from pathlib import Path
import pandas as pd
import zipfile
import tarfile

def extract_and_list_paths(file_path: str) -> list[str]:
    """
    Распаковывает архив и выводит все пути файлов внутри.
    
    Args:
        file_path: путь к архиву (.zip, .tar.gz, .txt с путями)
    
    Returns:
        список всех путей внутри архива
    """
    all_paths = []
    
    # Если это ZIP архив
    if zipfile.is_zipfile(file_path):
        print(f"🔓 Распаковка ZIP: {file_path}")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for path in file_list:
                all_paths.append(path)
                print(f"  📄 {path}")
            zip_ref.extractall("extracted_files")
    
    # Если это TAR/TGZ архив
    elif tarfile.is_tarfile(file_path):
        print(f"🔓 Распаковка TAR: {file_path}")
        with tarfile.open(file_path, 'r:auto') as tar_ref:
            for member in tar_ref.getmembers():
                all_paths.append(member.name)
                print(f"  📄 {member.name}")
            tar_ref.extractall("extracted_files")
    
    # Если это текстовый файл с путями (paste.txt)
    else:
        print(f"📝 Читаем пути из TXT: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Ищем все пути Windows/Linux формата
        import re
        paths = re.findall(r'[A-Za-z]:[\\\/][^"\n\r]+|/[^\s"\n\r]+', content)
        
        for path in paths:
            all_paths.append(path)
            print(f"  📄 {path}")
    
    print(f"\n✅ Всего найдено путей: {len(all_paths)}")
    return all_paths


