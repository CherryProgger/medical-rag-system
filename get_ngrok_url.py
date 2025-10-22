#!/usr/bin/env python3
"""
Скрипт для получения публичного URL от ngrok
"""

import requests
import time
import json

def get_ngrok_url():
    """Получает публичный URL от ngrok"""
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('tunnels'):
                    tunnel = data['tunnels'][0]
                    return tunnel['public_url']
        except Exception as e:
            print(f"Попытка {attempt + 1}: {e}")
            time.sleep(2)
    
    return None

if __name__ == "__main__":
    print("🔍 Ищем публичный URL ngrok...")
    url = get_ngrok_url()
    
    if url:
        print(f"🌐 Публичный URL: {url}")
        print(f"📱 Поделитесь этой ссылкой с другими людьми!")
    else:
        print("❌ Не удалось получить URL. Убедитесь, что ngrok запущен.")
