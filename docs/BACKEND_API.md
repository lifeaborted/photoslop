# PepeShop Backend API Specification & Integration Guide

> **Документация для фронтенд-разработчика (напарника по проекту)**  
> **Версия API:** 0.1.0  
> **Хост по умолчанию:** `http://localhost:8000`  
> **Интерактивная документация (Swagger UI):** `http://localhost:8000/docs`  
> **Альтернативная схема (ReDoc):** `http://localhost:8000/redoc`

---

## 1. Архитектура и структура проекта

Бэкенд представляет собой высокопроизводительный сервис на **FastAPI**, спроектированный специально для локального графического редактора с нейросетевыми возможностями на видеокартах с 6 ГБ VRAM (NVIDIA GeForce GTX 1660 Ti).

- **`/models/models.json`** — конфигурация доступных моделей, сгруппированных по категориям (`txt2img`, `inpaint`, `edit`, `background_removal`).
- **`/models/<folder_or_file>`** — веса локальных моделей (например, `sdxl-turbo-torch-int8`).
- **`/output/`** — папка сохранения готовых изображений (PNG/JPG).
- **`/logs/log.txt`** — структурированный лог работы через **Loguru** (замер времени, пиковый VRAM, обращение к моделям, ошибки).
- **`/frontend/`** — папка для статики твоего фронтенда (HTML, CSS, JS). Любые файлы из этой папки автоматически отдаются бэкендом по адресу `http://localhost:8000/`.

---

## 2. Формат обмена изображениями

Все графические данные между фронтендом и бэкендом передаются в виде стандартных **Base64 Data URI**:
```text
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
```
или **Multipart Form-Data** при загрузке файлов пачками с диска (`POST /api/editor/upload`).

---

## 3. Справочник REST API эндпоинтов

### 3.1. Управление моделями (`/api/models`)

#### `GET /api/models`
Возвращает список всех зарегистрированных моделей и текущие активные модели.

- **Ответ (200 OK):**
```json
{
  "status": "success",
  "total_count": 7,
  "models": [
    {
      "id": "sdxl-turbo-int8",
      "name": "SDXL Turbo INT8",
      "filename": "sdxl-turbo-torch-int8",
      "category": "txt2img",
      "is_downloaded": true,
      "file_size_mb": 4280.5
    },
    {
      "id": "flux1-fill-dev-q2",
      "name": "FLUX.1 Fill Dev (GGUF Q2_K)",
      "filename": "flux1-fill-dev-q2_k.gguf",
      "category": "inpaint",
      "is_downloaded": false,
      "file_size_mb": 0
    }
  ],
  "active_models": {
    "txt2img": { "id": "sdxl-turbo-int8", "name": "SDXL Turbo INT8" },
    "inpaint": { "id": "sd15-inpainting-fp16", "name": "Stable Diffusion 1.5 Inpainting" },
    "background_removal": { "id": "bria-rmbg-1.4", "name": "BRIA RMBG-1.4" }
  }
}
```

#### `POST /api/models/switch`
Динамическое переключение активной модели для заданной категории без перезапуска сервера.

- **Тело запроса (JSON):**
```json
{
  "category": "txt2img",
  "model_id": "sdxl-turbo-int8"
}
```
- **Ответ (200 OK):**
```json
{
  "status": "success",
  "category": "txt2img",
  "active_model": "sdxl-turbo-int8",
  "model_info": { ... }
}
```

#### `GET /api/models/hardware`
Текущая телеметрия видеокарты и оперативной памяти в реальном времени. Фронтенд может опрашивать этот эндпоинт раз в несколько секунд для отрисовки плашки статуса в шапке.

- **Ответ (200 OK):**
```json
{
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce GTX 1660 Ti",
  "driver_version": "621.01",
  "vram_total_mb": 6144.0,
  "vram_used_mb": 1010.5,
  "vram_free_mb": 5133.5,
  "vram_percent": 16.4
}
```

---

### 3.2. Редактор и операции со слоями (`/api/editor`)

#### `POST /api/editor/upload`
Загрузка изображений пачкой (через файловый диалог или Drag & Drop). Бэкенд возвращает готовый массив объектов слоев для автоматического распределения на холсте фронтенда.

- **Тело запроса:** `multipart/form-data` с полем `files` (один или несколько файлов).
- **Ответ (200 OK):**
```json
{
  "status": "success",
  "count": 2,
  "layers": [
    {
      "id": "layer_1730310000000_0",
      "name": "photo_1.png",
      "width": 1920,
      "height": 1080,
      "data_uri": "data:image/png;base64,...",
      "format": "PNG",
      "size_bytes": 1048576
    },
    {
      "id": "layer_1730310000000_1",
      "name": "photo_2.jpg",
      "width": 800,
      "height": 600,
      "data_uri": "data:image/png;base64,...",
      "format": "JPEG",
      "size_bytes": 245100
    }
  ]
}
```

#### `POST /api/editor/paste`
Обработка вставки из буфера обмена (`Ctrl+V`).

- **Тело запроса (JSON):**
```json
{
  "image_data": "data:image/png;base64,...",
  "name": "Clipboard_Layer"
}
```
- **Ответ (200 OK):**
```json
{
  "status": "success",
  "layer": {
    "id": "layer_paste_1730310000000",
    "name": "Clipboard_Layer",
    "width": 512,
    "height": 512,
    "data_uri": "data:image/png;base64,..."
  }
}
```

#### `POST /api/editor/adjustments`
Базовая коррекция изображения активного слоя (яркость, контраст, насыщенность, поворот, масштабирование).

- **Тело запроса (JSON):**
```json
{
  "image_data": "data:image/png;base64,...",
  "brightness": 1.1,
  "contrast": 1.05,
  "saturation": 1.2,
  "rotate_angle": 0.0,
  "resize_width": null,
  "resize_height": null
}
```
- **Ответ (200 OK):**
```json
{
  "status": "success",
  "width": 512,
  "height": 512,
  "data_uri": "data:image/png;base64,..."
}
```

#### `POST /api/editor/save`
Экспорт объединенного холста (merged canvas) или выбранного слоя в папку `/output/`.

- **Тело запроса (JSON):**
```json
{
  "image_data": "data:image/png;base64,...",
  "filename": "my_artwork",
  "format": "PNG"
}
```
*(Формат может быть `"PNG"` или `"JPEG"`).*
- **Ответ (200 OK):**
```json
{
  "status": "success",
  "filename": "my_artwork_20261030_212500.png",
  "path": "C:\\others\\Codes\\pepeshop\\output\\my_artwork_20261030_212500.png",
  "size_kb": 420.5,
  "width": 1920,
  "height": 1080
}
```

#### `GET /api/editor/outputs`
Список всех экспортированных изображений в директории `/output/`.

- **Ответ (200 OK):**
```json
{
  "status": "success",
  "output_dir": "C:\\others\\Codes\\pepeshop\\output",
  "files": [
    {
      "name": "my_artwork_20261030_212500.png",
      "size_kb": 420.5,
      "modified": 1730310300.0
    }
  ]
}
```

---

### 3.3. Нейросетевая обработка (`/api/generation`)

#### `POST /api/generation/txt2img`
Генерация нового изображения по текстовому запросу с помощью **SDXL Turbo INT8** (время генерации на GTX 1660 Ti — около 2 секунд).

- **Тело запроса (JSON):**
```json
{
  "prompt": "futuristic flying car over neon city, 8k resolution",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 512,
  "height": 512,
  "steps": 1,
  "seed": 42,
  "model_id": "sdxl-turbo-int8"
}
```
*(Параметры `steps`, `seed`, `width`, `height`, `model_id` опциональны; по умолчанию `steps=1`, `width=512`, `height=512`, `seed=null` — случайный).*

- **Ответ (200 OK):**
```json
{
  "status": "success",
  "model_id": "sdxl-turbo-int8",
  "model_name": "SDXL Turbo INT8",
  "elapsed_seconds": 2.45,
  "width": 512,
  "height": 512,
  "data_uri": "data:image/png;base64,..."
}
```

#### `POST /api/generation/inpaint`
Инпейнтинг (перерисовка выделенной области по маске и промпту).

- **Тело запроса (JSON):**
```json
{
  "image_data": "data:image/png;base64,...",
  "mask_data": "data:image/png;base64,...",
  "prompt": "vintage sunglasses",
  "negative_prompt": "blurry, artifacts",
  "steps": 20,
  "guidance_scale": 7.5,
  "seed": null,
  "model_id": "flux1-fill-dev-q2"
}
```
*(Маска `mask_data` — черно-белое изображение, где белый цвет (255) указывает область для замены).*

- **Ответ (200 OK):**
```json
{
  "status": "success",
  "model_id": "flux1-fill-dev-q2",
  "model_name": "FLUX.1 Fill Dev (GGUF Q2_K)",
  "elapsed_seconds": 3.12,
  "width": 512,
  "height": 512,
  "data_uri": "data:image/png;base64,..."
}
```

#### `POST /api/generation/text-edit`
Редактирование всего изображения по текстовой инструкции без явной маски (стилизация, изменение погоды/времени суток).

- **Тело запроса (JSON):**
```json
{
  "image_data": "data:image/png;base64,...",
  "prompt": "convert to vibrant cyberpunk neon at night",
  "strength": 0.65,
  "seed": null,
  "model_id": "flux1-kontext-q2"
}
```
- **Ответ (200 OK):**
```json
{
  "status": "success",
  "model_id": "flux1-kontext-q2",
  "model_name": "FLUX.1 Kontext (GGUF Q2_K)",
  "elapsed_seconds": 2.85,
  "width": 512,
  "height": 512,
  "data_uri": "data:image/png;base64,..."
}
```

#### `POST /api/generation/remove-background`
Удаление фона и выделение объекта (возвращает RGBA с прозрачным альфа-каналом).

- **Тело запроса (JSON):**
```json
{
  "image_data": "data:image/png;base64,...",
  "model_id": "bria-rmbg-1.4"
}
```
- **Ответ (200 OK):**
```json
{
  "status": "success",
  "model_id": "bria-rmbg-1.4",
  "model_name": "BRIA RMBG-1.4",
  "elapsed_seconds": 0.82,
  "width": 512,
  "height": 512,
  "data_uri": "data:image/png;base64,..."
}
```

---

## 4. Обработка ошибок

Если что-то пошло не так (например, неверный формат картинки, неизвестный ID модели или ошибка инференса), бэкенд возвращает стандартную ошибку HTTP:

- **HTTP 400 Bad Request:** Ошибка входных данных (невалидный Base64, некорректные размеры).
```json
{
  "detail": "Failed to parse pasted image: cannot identify image file"
}
```
- **HTTP 500 Internal Server Error:** Ошибка на стороне модели или нехватка памяти.
```json
{
  "detail": "CUDA out of memory. Tried to allocate 1.50 GiB..."
}
```
Все ошибки автоматически фиксируются в файле `/logs/log.txt` с полным стек-трейсом и состоянием памяти.

---

## 5. Как запускать бэкенд

Для разработки и тестирования бэкенд запускается одной командой из корневой директории проекта:

```powershell
# 1. Активация виртуального окружения
.venv\Scripts\Activate.ps1

# 2. Запуск сервера с авто-перезагрузкой при изменении кода
python backend/run.py
```
Сервер будет доступен по адресу `http://localhost:8000`. Любые изменения в файлах фронтенда внутри `/frontend/` сразу видны в браузере при обновлении страницы (`F5`).
