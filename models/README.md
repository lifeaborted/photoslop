# Хранилище моделей PepeShop

В этой директории располагаются файлы весов нейросетевых моделей и конфигурационный файл `models.json`.

## Конфигурация models.json
Бэкенд динамически считывает `models.json` при каждом обращении и при переключении моделей через веб-интерфейс:
- Поле `category`: `inpaint`, `edit`, `txt2img`, `background_removal`.
- Поле `filename`: имя файла в директории `models/`.
- Поле `download_url`: ссылка на Hugging Face / GitHub Releases для автоматического скачивания установщиком.
- Поле `is_downloaded`: проверяется бэкендом автоматически по наличию файла на диске.

## Структура директории
```
models/
├── models.json
├── sd-v1-5-inpainting.safetensors  (inpaint)
├── flux1-fill-dev-q2_k.gguf        (inpaint)
├── flux1-kontext-q2_k.gguf         (edit)
├── sd_xl_turbo_1.0_fp16.safetensors(txt2img)
├── model.onnx                      (bria rmbg)
└── u2net.onnx                      (rembg cpu fallback)
```
