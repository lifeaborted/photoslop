"""
Тест проверки и запуска SDXL-Turbo INT8 (imgailab/sdxl-turbo-torch-int8)
Проверяет структуру скачанных файлов, потребление VRAM и время генерации.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sdxl-turbo-torch-int8"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COMPONENTS = [
    "model_index.json",
    "quantization_info.json",
    "unet",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "scheduler"
]

def verify_structure() -> bool:
    print(f"=== [1/3] Проверка структуры файлов в {MODEL_PATH} ===")
    if not MODEL_PATH.exists():
        print(f"[-] Папка {MODEL_PATH} не найдена!")
        print("    Скачайте модель командой: python scripts/download_model.py")
        return False

    missing = []
    for comp in REQUIRED_COMPONENTS:
        p = MODEL_PATH / comp
        if not p.exists():
            missing.append(comp)

    if missing:
        print(f"[-] Отсутствуют обязательные компоненты модели: {missing}")
        return False

    print("  [+] Все обязательные компоненты найдены:")
    for comp in REQUIRED_COMPONENTS:
        p = MODEL_PATH / comp
        type_str = "DIR " if p.is_dir() else "FILE"
        print(f"      [✓] {type_str}: {comp}")
    return True

def test_inference():
    print("\n=== [2/3] Проверка инференса Diffusers & PyTorch ===")
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        print("[-] Библиотеки torch/diffusers не установлены в текущем окружении.")
        print("    Для полноценного запуска установите: pip install torch diffusers transformers accelerate")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Устройство для инференса: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print(f"  Загрузка пайплайна из {MODEL_PATH}...")
    start_load = time.time()
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant=None
        )
        pipe.to(device)
        print(f"  [+] Модель успешно загружена за {time.time() - start_load:.2f}с")
    except Exception as e:
        print(f"[-] Ошибка загрузки модели: {e}")
        return

    print("\n=== [3/3] Генерация тестового изображения (1 шаг) ===")
    prompt = "a cute frog in cyberpunk style, glowing neon lights, 8k resolution"
    print(f"  Промпт: '{prompt}'")
    
    start_gen = time.time()
    # SDXL Turbo оптимизирован под guidance_scale=0.0 и 1-4 шага
    image = pipe(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
        width=512,
        height=512
    ).images[0]
    gen_time = time.time() - start_gen

    out_file = OUTPUT_DIR / "test_sdxl_turbo_int8.png"
    image.save(out_file)
    print(f"  [+] Генерация завершена за {gen_time:.2f}с!")
    print(f"  [+] Результат сохранен в: {out_file}")

    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        print(f"  [i] Пиковое потребление VRAM: {peak_vram:.2f} ГБ")
        assert peak_vram <= 6.0, "Потребление VRAM превышает 6 ГБ!"
        print("  [✓] Тест VRAM успешно пройден (< 6 ГБ)!")

if __name__ == "__main__":
    if verify_structure():
        test_inference()
