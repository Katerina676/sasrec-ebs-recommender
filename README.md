# Рекомендательная система для ЭБС на основе SASRec
Прототип рекомендательной системы для Электронной Библиотечной Системы (ЭБС) университета.
Модель SASRec (Self-Attentive Sequential Recommendation) предсказывает следующую книгу
на основе истории чтения пользователя.

## Данные

- Датасет Yambda (Яндекс.Музыка)

## Установка

### 1. Клонировать репозиторий
- git clone https://github.com/Katerina676/sasrec-ebs-recommender.git
- cd sasrec-ebs-recommender

### 2. Создать виртуальное окружение
- python -m venv venv

### 3. Активировать окружение

### Windows:
- venv\Scripts\activate
### Linux/Mac:
- source venv/bin/activate

### 4. Установить зависимости

- pip install -r requirements.txt

### Если у вас NVIDIA GPU:
- сначала удалить torch потом установить верный
- pip uninstall torch torchvision torchaudio -y

- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
### в colab
- будет по умолчанию cuda если выбрать tesla t4

## Запуск GUI
- python gui_demo.py