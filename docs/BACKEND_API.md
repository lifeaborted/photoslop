# PepeShop Backend API Specification & Integration Guide

> **Документация для фронтенд-разработчика (напарника по проекту)**  
> **Версия API:** 0.1.0  
> **Хост по умолчанию:** `http://localhost:8000`  
> **Интерактивная документация (Swagger UI):** `http://localhost:8000/docs`  
> **Альтернативная схема (ReDoc):** `http://localhost:8000/redoc`

---

## 1. Архитектура и утверждённые модели

Бэкенд работает на **FastAPI** и оптимизирован под локальный ПК с **6 ГБ VRAM (NVIDIA GTX 1660 Ti)** + **CPU Fallback**.

- **`/models/models.json`** — продакшн-конфиг 5 утверждённых моделей, сгруппированных по 5 категориям:
  - `txt2img`: `sdxl-turbo-int8` (TorchAO INT8, ~`1.66 с` тёплая)
  - `background_removal`: `birefnet-general-lite` (ONNX CPU, `IoU = 0.9402`, ~`10.25 с`)
  - `object_removal`: `lama-cleaner-onnx` (ONNX CPU, без промпта, ~`4.33 с`)
  - `inpaint`: `dreamshaper-8-inpainting` (GPU FP32 + `cpu_offload`, ~`13.36 с`)
  - `edit`: `instruct-pix2pix-fp16` (GPU FP32 + `cpu_offload`, без маски, ~`10.15 с`)
- **`/output/`** — папка сохранения экспортированных PNG/JPEG.
- **`/logs/log.txt`** — структурированный лог работы **Loguru** (время, пик VRAM/RAM, переключения моделей, ошибки).
- **`/frontend/`** — статика UI, отдаётся по `http://localhost:8000/`.

> **Формат изображений:** все картинки и маски передаются в Base64 Data URI (`data:image/png;base64,...`), кроме пакетного `POST /api/editor/upload` (`multipart/form-data`).  
> Для `seed`: используйте `null` (или `-1`, бэкенд сам приведёт отрицательное значение к случайному `seed`).

---

## 2. Справочник REST API эндпоинтов

### 2.1. Управление моделями (`/api/models`)

#### `GET /api/models`
Возвращает список 5 моделей и словарь активных моделей по категориям.

#### `GET /api/models/active`
Возвращает текущие активные модели (`txt2img`, `background_removal`, `object_removal`, `inpaint`, `edit`).

#### `POST /api/models/switch`
Динамическое переключение активной модели **внутри своей категории**:
```json
{
  "category": "inpaint",
  "model_id": "dreamshaper-8-inpainting"
}
```
*(Если передать `model_id` от другой категории, бэкенд вернёт `400 Bad Request`.)*

#### `GET /api/models/hardware`
Телеметрия видеокарты (`gpu_available`, `gpu_name`, `driver_version`, `vram_total_mb`, `vram_used_mb`, `vram_free_mb`, `vram_percent`).

---

### 2.2. Редактор, слои и базовые правки (`/api/editor`)

#### `POST /api/editor/upload`
- `multipart/form-data`, поле `files` (один или несколько файлов).
- Возвращает массив объектов `layers` (`id`, `name`, `width`, `height`, `data_uri`, `format`, `size_bytes`) для распределения по слоям на Canvas.

#### `POST /api/editor/paste`
- JSON: `{"image_data": "data:image/png;base64,...", "name": "Clipboard_Layer"}`.

#### `POST /api/editor/adjustments`
Применяет к слою яркость, контраст, насыщенность, **кадрирование (crop)**, поворот и resize:
```json
{
  "image_data": "data:image/png;base64,...",
  "brightness": 1.1,
  "contrast": 1.05,
  "saturation": 1.2,
  "crop_x": 50,
  "crop_y": 40,
  "crop_width": 400,
  "crop_height": 300,
  "rotate_angle": 0.0,
  "resize_width": null,
  "resize_height": null
}
```
*(Все `crop_*`, `resize_*` опциональны; если `crop_*` переданы — кадрирование выполняется до поворота и масштабирования.)*

#### `POST /api/editor/save` и `GET /api/editor/outputs`
Сохранение в `/output/` (`format`: `"PNG"` или `"JPEG"`) и получение списка сохранённых файлов.

---

### 2.3. Нейросетевая обработка (`/api/generation`)

#### `POST /api/generation/remove-background`
Удаление фона в 1 клик (автоматическое определение главного объекта, возвращает RGBA с прозрачностью):
```json
{
  "image_data": "data:image/png;base64,...",
  "model_id": "birefnet-general-lite"
}
```

#### `POST /api/generation/replace-background`
Удаление фона + композитинг объекта поверх цвета, пользовательского изображения или AI-сгенерированного фона:
```json
{
  "image_data": "data:image/png;base64,...",
  "bg_type": "color",
  "bg_color": "#ffffff",
  "bg_image_data": null,
  "bg_prompt": null,
  "bg_model_id": null,
  "txt2img_model_id": null,
  "seed": null
}
```
Режимы `bg_type`:
- `"color"` — заливка цветом `bg_color` (hex или CSS-имя).
- `"image"` — подстановка своей картинки из `bg_image_data`.
- `"prompt"` — генерация нового фона через `SDXL Turbo INT8` по `bg_prompt`.

#### `POST /api/generation/remove-object`
«Умный ластик» без текстового промпта (`lama-cleaner-onnx`) — удаляет закрашенный объект и достраивает фон:
```json
{
  "image_data": "data:image/png;base64,...",
  "mask_data": "data:image/png;base64,...",
  "model_id": "lama-cleaner-onnx"
}
```
*(Белые пиксели в `mask_data` = зона удаления.)*

#### `POST /api/generation/inpaint`
Генеративный инпейнтинг внутри белой области маски (`dreamshaper-8-inpainting`):
```json
{
  "image_data": "data:image/png;base64,...",
  "mask_data": "data:image/png;base64,...",
  "prompt": "vintage brown leather backpack",
  "negative_prompt": "blurry, artifacts",
  "steps": 20,
  "guidance_scale": 7.5,
  "seed": null,
  "model_id": "dreamshaper-8-inpainting"
}
```

#### `POST /api/generation/text-edit`
Редактирование всего кадра целиком по текстовой инструкции без маски (`instruct-pix2pix-fp16`):
```json
{
  "image_data": "data:image/png;base64,...",
  "prompt": "turn the scene into winter with snow",
  "strength": 0.6,
  "seed": null,
  "model_id": "instruct-pix2pix-fp16"
}
```

#### `POST /api/generation/txt2img`
Генерация изображения с нуля (`sdxl-turbo-int8`, оптимально `steps: 1`):
```json
{
  "prompt": "ripe red apple on a wooden table, studio lighting",
  "negative_prompt": "",
  "width": 512,
  "height": 512,
  "steps": 1,
  "seed": 42,
  "model_id": "sdxl-turbo-int8"
}
```

---

## 3. Заметки для обновления UI фронтенда

В текущем `frontend/` бэкенд уже полностью поддерживает, но UI ещё можно дополнить следующими действиями:
1. **Разделить селектор `#select-inpaint-model`** на отдельные выборы для `inpaint` (`dreamshaper-8-inpainting`) и `edit` (`instruct-pix2pix-fp16`) *(бэкенд защищён и при несовпадении автоматически переключит на активную модель нужной категории, но раздельный UI нагляднее)*.
2. Добавить кнопку **«Стереть объект (Ластик AI)»** -> вызов `POST /api/generation/remove-object` с маской без требования вводить промпт.
3. Добавить панель **«Замена фона»** -> вызов `POST /api/generation/replace-background` (`color` / `image` / `prompt`).
4. Вывести инструмент выделения области для передачи `crop_x / crop_y / crop_width / crop_height` в `POST /api/editor/adjustments`.

---

## 4. Запуск бэкенда для разработки UI

```powershell
uv run python backend/run.py
```
После запуска открывай `http://localhost:8000` (и `http://localhost:8000/docs` для интерактивной проверки любых запросов).
