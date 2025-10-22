# 🚀 Руководство по развертыванию Medical RAG System

## 📋 Пошаговая инструкция

### 1. Создайте репозиторий на GitHub

1. Перейдите на [github.com](https://github.com)
2. Нажмите **"New repository"** (зеленая кнопка)
3. Заполните:
   - **Repository name:** `medical-rag-system`
   - **Description:** `Medical RAG System for answering questions about medical documentation`
   - **Visibility:** ✅ **Public** (обязательно!)
   - **НЕ** добавляйте README, .gitignore, лицензию
4. Нажмите **"Create repository"**

### 2. Загрузите код на GitHub

**Вариант A: Автоматически (рекомендуется)**
```bash
# Замените YOUR_USERNAME на ваш username GitHub
git remote add origin https://github.com/YOUR_USERNAME/medical-rag-system.git
git branch -M main
git push -u origin main
```

**Вариант B: Используйте скрипт**
```bash
# Сначала настройте remote origin
git remote add origin https://github.com/YOUR_USERNAME/medical-rag-system.git
git branch -M main

# Затем запустите скрипт
./deploy_to_github.sh
```

### 3. Разверните на Streamlit Cloud

1. Перейдите на [share.streamlit.io](https://share.streamlit.io)
2. Нажмите **"New app"**
3. Заполните:
   - **Repository:** `YOUR_USERNAME/medical-rag-system`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Нажмите **"Deploy!"**

### 4. Настройте переменные окружения (если понадобится)

В настройках приложения добавьте:
- `PYTHONPATH`: `/app/src`

## 🎯 Результат

После развертывания вы получите публичный URL вида:
```
https://your-app-name.streamlit.app
```

## 🔧 Устранение неполадок

### Ошибка: "Module not found"
- Убедитесь, что `streamlit_app.py` находится в корне репозитория
- Проверьте, что все файлы загружены на GitHub

### Ошибка: "Import error"
- Добавьте переменную окружения `PYTHONPATH`: `/app/src`

### Медленная загрузка
- Это нормально для первого запуска (загружаются модели)
- Последующие запросы будут быстрее

## 📱 Тестирование

После развертывания протестируйте с вопросами:
- "Что такое варикозное расширение вен?"
- "Как лечить флебиты?"
- "Что такое тромбоэмболия?"

## 🎉 Готово!

Ваша медицинская RAG система теперь доступна публично!
