#!/usr/bin/env python3
"""
Скрипт для загрузки кода на GitHub через API
"""

import os
import base64
import requests
import json
from pathlib import Path

# Настройки
REPO_OWNER = "CherryProgger"
REPO_NAME = "medical-rag-system"
BRANCH = "main"

# Файлы для загрузки (исключаем служебные)
EXCLUDE_FILES = {
    '.git', 'venv', '__pycache__', '.DS_Store', 
    'medical-rag-system.tar.gz', 'get_ngrok_url.py',
    'deploy_to_github.sh', 'upload_to_github.py'
}

def upload_file(file_path, content):
    """Загружает файл на GitHub"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}"
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Кодируем содержимое в base64
    encoded_content = base64.b64encode(content).decode('utf-8')
    
    data = {
        "message": f"Add {file_path}",
        "content": encoded_content,
        "branch": BRANCH
    }
    
    try:
        response = requests.put(url, headers=headers, json=data)
        if response.status_code in [200, 201]:
            print(f"✅ {file_path} - загружен")
            return True
        else:
            print(f"❌ {file_path} - ошибка: {response.status_code}")
            print(f"   {response.text}")
            return False
    except Exception as e:
        print(f"❌ {file_path} - исключение: {e}")
        return False

def main():
    """Основная функция"""
    print("🚀 Загружаем файлы на GitHub...")
    
    project_root = Path(".")
    uploaded_count = 0
    total_files = 0
    
    # Собираем все файлы для загрузки
    files_to_upload = []
    for file_path in project_root.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(project_root)
            if str(relative_path) not in EXCLUDE_FILES:
                files_to_upload.append(relative_path)
    
    total_files = len(files_to_upload)
    print(f"📁 Найдено {total_files} файлов для загрузки")
    
    # Загружаем файлы
    for file_path in files_to_upload:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            if upload_file(str(file_path), content):
                uploaded_count += 1
                
        except Exception as e:
            print(f"❌ Ошибка чтения {file_path}: {e}")
    
    print(f"\n🎉 Загружено {uploaded_count} из {total_files} файлов")
    
    if uploaded_count > 0:
        print(f"🌐 Репозиторий: https://github.com/{REPO_OWNER}/{REPO_NAME}")
        print("🚀 Теперь можно развернуть на Streamlit Cloud!")

if __name__ == "__main__":
    main()
