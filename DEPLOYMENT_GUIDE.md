# Руководство по развертыванию Medical RAG System на «чистом» macOS

Ниже приведён полный сценарий: от установки инструментов до публикации приложения на Streamlit Cloud. Подходит для Mac, где ещё не настроены Git, Python и прочие утилиты.

## 1. Что понадобится
- доступ администратора на macOS
- учётная запись GitHub и Streamlit Cloud
- устойчивое интернет‑соединение
- не менее 5 ГБ свободного места на диске

## 2. Установка базовых инструментов
1. Откройте Spotlight (⌘+Space), введите `Terminal` и запустите терминал.
2. Установите инструменты командной строки Xcode:
   ```bash
   xcode-select --install
   ```
   После запуска появится окно установки. Подтвердите и дождитесь окончания процесса.
3. Если у вас процессор Apple Silicon и система предлагается установить Rosetta, выполните:
   ```bash
   softwareupdate --install-rosetta --agree-to-license
   ```

## 3. Установка Homebrew, Git и Python
1. Установите Homebrew — пакетный менеджер для macOS:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Добавьте Homebrew в PATH (команды отличаются в зависимости от архитектуры, установщик подскажет точные строки). На Apple Silicon обычно нужно выполнить:
   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```
3. Установите Git и Python 3.11:
   ```bash
   brew install git python@3.11
   ```
4. Проверьте версии:
   ```bash
   git --version
   python3 --version
   ```

## 4. Получение исходников проекта
Перейдите в каталог, где хотите разместить проект (например, `~/Documents`):
```bash
cd ~/Documents
```

### Вариант A. Клонирование через Git (предпочтительный)
```bash
git clone https://github.com/YOUR_USERNAME/medical-rag-system.git
cd medical-rag-system
```

### Вариант B. Загрузка архива
1. На странице репозитория GitHub нажмите **Code → Download ZIP**.
2. Распакуйте архив (двойной клик) и перейдите в папку проекта:
   ```bash
   cd ~/Downloads/medical-rag-system-main
   ```
3. При необходимости инициализируйте Git позднее (см. шаг 8).

## 5. Создание виртуального окружения и установка зависимостей
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Проверка локальной работы (по желанию)
- CLI-режим:
  ```bash
  PYTHONPATH=src python main.py --cli
  ```
- Простой локальный веб-интерфейс (Gradio):
  ```bash
  PYTHONPATH=src python scripts/simple_local_ui.py
  ```
  Откройте в браузере `http://127.0.0.1:7860`.

## 7. Подготовка репозитория к публикации
Если вы скачали архив и в каталоге нет `.git`, выполните:
```bash
git init
git add .
git commit -m "Initial commit"
```

## 8. Создание удалённого репозитория
1. Перейдите на [github.com](https://github.com) → **New repository**.
2. Задайте параметры:
   - Repository name: `medical-rag-system`
   - Visibility: Public
   - Не добавляйте README, .gitignore и лицензию
3. Нажмите **Create repository** и скопируйте URL вида `https://github.com/YOUR_USERNAME/medical-rag-system.git`.

## 9. Загрузка кода в GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/medical-rag-system.git
git branch -M main
git push -u origin main
```
При необходимости введите данные учётной записи или персональный токен GitHub.

## 10. Развёртывание на Streamlit Cloud
1. Откройте https://share.streamlit.io и нажмите **New App**.
2. Укажите:
   - Repository: `YOUR_USERNAME/medical-rag-system`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
3. В разделе *Advanced settings → Environment variables* добавьте переменную:
   - `PYTHONPATH` = `/app/src`
4. Нажмите **Deploy** и дождитесь завершения сборки (обычно 2–5 минут).

После деплоя приложение будет доступно по адресу вида `https://your-app-name.streamlit.app`.

## 11. Проверка работоспособности
Зайдите на опубликованный URL и попробуйте несколько запросов:
- «Что такое варикозное расширение вен?»
- «Как диагностировать тромбофлебит поверхностных вен?»

Если ответы корректны, публикация завершена.

## 12. Частые проблемы
- **Module not found** — убедитесь, что `streamlit_app.py` лежит в корне репозитория и зависимости установлены в `requirements.txt`.
- **Import error** — проверьте, что переменная окружения `PYTHONPATH` выставлена в `/app/src` (на Streamlit Cloud → Settings → Advanced settings).
- **Первая загрузка выполняется долго** — это штатно, модели и индекс считываются в память. Повторные запросы работают быстрее.

## 13. Обновление приложения
1. Внесите изменения локально.
2. Выполните:
   ```bash
   git add .
   git commit -m "Описание изменений"
   git push
   ```
3. Streamlit Cloud автоматически подтянет свежий код и перезапустит приложение.

На этом всё: проект готов к демонстрации и дальнейшему развитию.
