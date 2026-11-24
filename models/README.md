# Хранилище моделей PepeShop

В этой директории располагаются конфиги моделей и локальные веса (сами бинарные веса исключены из Git через `.gitignore`).

## Загрузка утверждённых весов в один клик

Для скачивания всех 5 продакшн-моделей с Hugging Face / rembg выполните:

```powershell
uv run python scripts/download_model.py --stage champions
uv run python scripts/download_model.py --list
```

## Конфигурация `models/models.json`

Бэкенд динамически считывает [`models.json`](file:///c:/others/Codes/pepeshop/models/models.json) и отслеживает наличие весов на диске (`is_downloaded`):
- `categories`: `txt2img`, `background_removal`, `object_removal`, `inpaint`, `edit`.
- `filename`: имя `.onnx` файла или папки Diffusers относительно `models/`.
- `repo_id` / `rembg_model`: источник для автоматической загрузки через [`scripts/download_model.py`](file:///c:/others/Codes/pepeshop/scripts/download_model.py).

## Продакшн-состав директории (`<= 6 ГБ VRAM`)

```text
models/
├── models.json                            # Активный продакшн-конфиг (5 моделей)
├── models_benchmark.json                  # Расширенный конфиг кандидатов для бенчмарков
├── sdxl-turbo-torch-int8/                 # txt2img: SDXL Turbo INT8 (TorchAO, ~4.1 ГБ)
├── rembg/models/birefnet-general-lite/    # background_removal: BiRefNet General Lite ONNX (~214 МБ, CPU)
├── lama_fp32.onnx                         # object_removal: LaMa FP32 ONNX (~200 МБ, CPU)
├── dreamshaper-8-inpainting/              # inpaint: DreamShaper 8 Inpainting FP16 (~2.0 ГБ)
└── instruct-pix2pix-fp16/                 # edit: InstructPix2Pix FP16 (~2.0 ГБ)
```
