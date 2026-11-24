# PepeShop: Локальный AI-редактор изображений (ТЗ 4)

Локальный веб-редактор с поддержкой многослойной композиции, классическими инструментами обработки и 5 открытыми нейросетевыми моделями (удаление фона на CPU, генерация `txt2img`, стирание объектов, инпейнтинг по маске и текстовое редактирование всего кадра), оптимизированными под видеокарты уровня **6 ГБ VRAM** (NVIDIA GeForce GTX 1660 Ti, архитектура Turing).

---

## Содержание
- [Архитектура и стек технологий](#архитектура-и-стек-технологий)
- [Структура репозитория](#структура-репозитория)
- [Утверждённый стек моделей (6 ГБ VRAM / CPU)](#утверждённый-стек-моделей-6-гб-vram--cpu)
- [Конфигурация моделей (`models/models.json`)](#конфигурация-моделей-modelsmodelsjson)
- [Система логирования (Loguru)](#система-логирования-loguru)
- [Развёртывание и запуск](#развёртывание-и-запуск)
- [API и функционал редактора](#api-и-функционал-редактора)
- [Соответствие критериям приёмки ТЗ 4](#соответствие-критериям-приёмки-тз-4)

---

## Архитектура и стек технологий

```text
               +-------------------------------------------------------+
               |              Браузер (Frontend UI)                    |
               |  - HTML5 Canvas Engine (Многослойный рендеринг)       |
               |  - Вставка Ctrl+V, Drag-and-Drop и пакетная загрузка  |
               |  - Переключатель моделей и панель цветокоррекции      |
               |  - Кисть, ластик, маска инпейнтинга, Undo/Redo        |
               +---------------------------+---------------------------+
                                           | HTTP REST API
                                           v
               +-------------------------------------------------------+
               |            FastAPI Backend (Python 3.12)              |
               |  - ModelsManager (динамический парсер models.json)    |
               |  - PipelineManager (кэш и выгрузка конкурирующих GPU) |
               |  - Pipelines (Txt2Img, Rembg, LaMa, Inpaint, Edit)    |
               |  - Асинхронный логер Loguru -> /logs/log.txt          |
               +---------------------------+---------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |                       Хранилище                       |
               |  - /models  : models.json и веса нейросетей           |
               |  - /output  : экспортированные изображения (PNG/JPG)  |
               |  - /logs    : ротируемый лог log.txt                  |
               +-------------------------------------------------------+
```

- **Backend:** Python 3.11+, FastAPI, Uvicorn, PyTorch (CUDA 12.4), TorchAO, Diffusers, ONNX Runtime, rembg, Pillow, NumPy, Loguru.
- **Frontend:** Vanilla ES6+ Modules, HTML5 Canvas API, CSS3.
- **Контейнеризация и установка:** Docker, Docker Compose (NVIDIA Container Runtime), Python/PyInstaller-установщик и PowerShell-скрипт.

---

## Структура репозитория

```text
pepeshop/
├── models/
│   ├── models.json                 # Продакшн-конфиг 5 утверждённых моделей
│   ├── models_benchmark.json       # Конфиг кандидатов для воспроизведения тестов
│   └── README.md                   # Инструкция по весам
├── output/                         # Папка экспорта готовых изображений
├── logs/                           # Папка логов (log.txt)
├── backend/
│   ├── app/
│   │   ├── config.py               # Пути и настройки сервера
│   │   ├── logger.py               # Loguru + трекер VRAM/RAM и скорости
│   │   ├── models_manager.py       # Парсинг models.json и переключение моделей
│   │   ├── pipeline_manager.py     # Кэш активного пайплайна и очистка VRAM
│   │   ├── main.py                 # FastAPI, CORS, статика фронтенда
│   │   ├── pipelines/
│   │   │   ├── base.py
│   │   │   ├── rembg_pipeline.py   # BiRefNet General Lite (ONNX, CPU)
│   │   │   ├── txt2img_pipeline.py # SDXL Turbo INT8 (TorchAO + FP32 VAE)
│   │   │   └── inpaint_pipeline.py # LaMa ONNX + DreamShaper 8 + InstructPix2Pix
│   │   └── api/
│   │       ├── models.py           # /api/models/*
│   │       ├── editor.py           # /api/editor/* (upload, paste, adjustments+crop, save)
│   │       └── generation.py       # /api/generation/* (все 6 AI-операций)
│   ├── requirements.txt
│   └── run.py
├── frontend/                       # Клиентский интерфейс (HTML/CSS/JS)
├── scripts/
│   ├── download_model.py           # Поэтапное скачивание моделей (--stage champions)
│   ├── cli_test_api.py             # CLI-клиент для проверки API из терминала
│   ├── prepare_dataset.py          # Подготовка бенчмарк-датасета
│   ├── benchmark_collector.py      # Сборщик замеров скорости и памяти
│   └── compute_metrics.py          # Расчёт IoU, Boundary-IoU, CLIP, PSNR, SSIM
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── installer/
    ├── installer.py
    ├── build_installer.py
    └── install.ps1
```

---

## Утверждённый стек моделей (6 ГБ VRAM / CPU)

На этапе выбора моделей были отсеяны тяжёлые 12B архитектуры (`FLUX.1` в связке с `T5-XXL` не помещается в 6 ГБ VRAM и нарушает лимит 60 с из-за PCIe-оффлоада) и нестабильные FP16-конфигурации (`Segmind SSD-1B` давал `NaN` на GPU Turing и 59.5 с/кадр). Итоговый проверенный стек на GTX 1660 Ti (6 ГБ VRAM) и Ryzen CPU:

| Задача | Утверждённая модель | Режим | Фактические замеры (`512×512`) |
| :--- | :--- | :--- | :--- |
| **Удаление фона** (`background_removal`) | **BiRefNet General Lite** (`ONNX`) | **CPU** (`0 МБ` VRAM) | **IoU `0.9402`** ($\ge 0.90$), B-IoU `0.5770`, **`10.25 с`** *(тёплое `9.69 с`)* |
| **Генерация** (`txt2img`) | **SDXL Turbo INT8** (`TorchAO`, 1 шаг) | **GPU** (`CUDA`) | **CLIP `0.3572`**, **`3.85 с`** *(прогретое **`1.66 с`**)* |
| **Удаление объектов** (`object_removal`) | **LaMa FP32** (`ONNX`) | **CPU** (`0 МБ` VRAM) | **PSNR `33.36 dB`**, **SSIM `0.9798`**, **`4.33 с`** |
| **Инпейнтинг по маске** (`inpaint`) | **DreamShaper 8 Inpainting** (20 шагов) | **GPU `FP32` + `cpu_offload`** | `4.17 ГБ` пик VRAM, **`13.36 с`** *(тёплое `12.90 с`)* |
| **Правка целиком по тексту** (`edit`) | **InstructPix2Pix FP16** (10 шагов) | **GPU `FP32` + `cpu_offload`** | **CLIP `0.2317`**, `4.23 ГБ` пик VRAM, **`10.15 с`** *(тёплое `9.99 с`)* |

> **Инженерная адаптация под NVIDIA Turing (GTX 1660 Ti):** для `SDXL Turbo INT8` применён патч TorchAO с FP32-аккумуляцией `torch.mm` и апкастом VAE до `float32`; для SD 1.5 / InstructPix2Pix используется нативный runtime `torch.float32` с `enable_model_cpu_offload()`, что полностью устранило `NaN` и чёрные кадры.

---

## Конфигурация моделей (`models/models.json`)

Конфигурация разбита по категориям (`txt2img`, `background_removal`, `object_removal`, `inpaint`, `edit`):

```json
{
  "categories": {
    "txt2img": [
      {
        "id": "sdxl-turbo-int8",
        "name": "SDXL Turbo INT8 (TorchAO)",
        "filename": "sdxl-turbo-torch-int8",
        "repo_id": "imgailab/sdxl-turbo-torch-int8",
        "default_steps": 1,
        "default_guidance": 0.0
      }
    ],
    "background_removal": [
      {
        "id": "birefnet-general-lite",
        "name": "BiRefNet General Lite (ONNX SOTA)",
        "filename": "rembg/models/birefnet-general-lite/birefnet-general-lite.onnx",
        "rembg_model": "birefnet-general-lite"
      }
    ],
    "object_removal": [
      {
        "id": "lama-cleaner-onnx",
        "name": "LaMa (Fast Fourier Object Eraser)",
        "filename": "lama_fp32.onnx",
        "repo_id": "Carve/LaMa-ONNX"
      }
    ],
    "inpaint": [
      {
        "id": "dreamshaper-8-inpainting",
        "name": "DreamShaper 8 Inpainting (FP16)",
        "filename": "dreamshaper-8-inpainting",
        "repo_id": "Lykon/dreamshaper-8-inpainting",
        "variant": "fp16",
        "default_steps": 20,
        "default_guidance": 7.5
      }
    ],
    "edit": [
      {
        "id": "instruct-pix2pix-fp16",
        "name": "InstructPix2Pix (FP16 Instruction Editing)",
        "filename": "instruct-pix2pix-fp16",
        "repo_id": "timbrooks/instruct-pix2pix",
        "variant": "fp16",
        "default_steps": 10,
        "default_guidance": 7.5
      }
    ]
  }
}
```

---

## Система логирования (Loguru)

Логирование ведётся через `loguru` в [`logs/log.txt`](file:///c:/others/Codes/pepeshop/logs/log.txt) (ротация 20 МБ, хранение 14 дней, `enqueue=True`), включая перехват стандартных логов FastAPI/Uvicorn и детальный трекер `track_model_execution()` (время, пик VRAM, RAM и результат каждого инференса).

---

## Развёртывание и запуск

### 1. Локальный запуск с `uv` (рекомендуемый способ разработки)

```powershell
# Скачать только 5 утверждённых продакшн-моделей
uv run python scripts/download_model.py --stage champions

# Проверить статус моделей на диске
uv run python scripts/download_model.py --list

# Запустить FastAPI-сервер (http://localhost:8000, Swagger: /docs)
uv run python backend/run.py
```

После запуска сервера его можно протестировать из терминала:

```powershell
uv run python scripts/cli_test_api.py --status
uv run python scripts/cli_test_api.py --models
uv run python scripts/cli_test_api.py --task txt2img --prompt "ripe red apple on wood table" --steps 1 --seed 42
```

### 2. Docker Compose / Установщики

```powershell
# Docker Compose с пробросом GPU и томов (/models, /output, /logs)
docker compose -f docker/docker-compose.yml up --build -d

# Либо 1-Click PowerShell / интерактивный Python-установщик
powershell -ExecutionPolicy Bypass -File installer/install.ps1
uv run python installer/installer.py
```

---

## API и функционал редактора

- **Модели (`/api/models`):** `GET /api/models`, `GET /api/models/active`, `POST /api/models/switch`, `GET /api/models/hardware`.
- **Редактор (`/api/editor`):**
  - `POST /api/editor/upload` — пакетная загрузка изображений по отдельным слоям;
  - `POST /api/editor/paste` — вставка из буфера обмена (`Ctrl+V`);
  - `POST /api/editor/adjustments` — яркость, контраст, насыщенность, **crop** (`crop_x`, `crop_y`, `crop_width`, `crop_height`), поворот, resize;
  - `POST /api/editor/save` / `GET /api/editor/outputs` — экспорт результата в PNG / JPEG в `/output/`.
- **AI-операции (`/api/generation`):**
  - `POST /api/generation/remove-background` — 1-клик удаление фона на CPU (`BiRefNet General Lite`);
  - `POST /api/generation/replace-background` — вырезание объекта и композитинг на цвет (`color`), пользовательский фон (`image`) или сгенерированный фон (`prompt`);
  - `POST /api/generation/remove-object` — бесшовное удаление выделенного кистью объекта без промпта (`LaMa FP32 ONNX`);
  - `POST /api/generation/inpaint` — генерация/замена внутри маски по промпту (`DreamShaper 8 Inpainting`);
  - `POST /api/generation/text-edit` — преобразование всего кадра по текстовой инструкции (`InstructPix2Pix`);
  - `POST /api/generation/txt2img` — генерация нового слоя по тексту (`SDXL Turbo INT8`).

Полная интерактивная схема всех запросов доступна после запуска сервера по адресу `http://localhost:8000/docs`.
