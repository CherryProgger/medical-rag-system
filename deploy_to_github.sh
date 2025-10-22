#!/bin/bash

# Скрипт для загрузки кода на GitHub
echo "🚀 Загружаем код на GitHub..."

# Проверяем, что мы в правильной директории
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Ошибка: файл streamlit_app.py не найден. Запустите скрипт из корневой директории проекта."
    exit 1
fi

# Проверяем, что git инициализирован
if [ ! -d ".git" ]; then
    echo "❌ Ошибка: git не инициализирован. Запустите 'git init' сначала."
    exit 1
fi

# Добавляем все файлы
echo "📁 Добавляем файлы в git..."
git add .

# Коммитим изменения
echo "💾 Создаем коммит..."
git commit -m "Medical RAG System - Ready for deployment"

# Проверяем, есть ли remote origin
if git remote get-url origin >/dev/null 2>&1; then
    echo "📤 Загружаем на GitHub..."
    git push -u origin main
    echo "✅ Код успешно загружен на GitHub!"
    echo "🌐 Теперь перейдите на https://share.streamlit.io для развертывания"
else
    echo "⚠️  Remote origin не настроен."
    echo "📋 Выполните следующие команды:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/medical-rag-system.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
fi
