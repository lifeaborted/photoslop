"""
Тест проверки и запуска SDXL-Turbo INT8 (imgailab/sdxl-turbo-torch-int8)
Проверяет структуру скачанных файлов, зависимости TorchAO/CUDA, потребление VRAM и время генерации.
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
        print(f"      [+] {type_str}: {comp}")
    return True


def apply_torchao_compatibility_patch(torch):
    """
    Разрешает загрузку сериализованных весов TorchAO в PyTorch 2.6+.
    По умолчанию в PyTorch 2.6+ включен weights_only=True, блокирующий кастомные классы pickle.
    """
    orig_load = torch.load

    def safe_load(*args, **kwargs):
        # Отключаем weights_only=True для доверенной локальной модели
        kwargs["weights_only"] = False
        if not torch.cuda.is_available() and "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return orig_load(*args, **kwargs)

    torch.load = safe_load

    # Пробуем зарегистрировать safe_globals
    try:
        import torchao
        from torchao.dtypes import AffineQuantizedTensor
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([AffineQuantizedTensor])
    except Exception:
        pass


def test_inference():
    print("\n=== [2/3] Проверка зависимостей и инференса Diffusers ===")

    # 1. Проверка PyTorch
    try:
        import torch
    except ImportError:
        print("[-] Библиотека torch не установлена!")
        print("    Установите: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return

    # 2. Проверка TorchAO
    try:
        import torchao
        print(f"  [+] torchao обнаружен (версия: {torchao.__version__})")
    except ImportError:
        print("[-] ОШИБКА: Библиотека torchao не установлена!")
        print("    Модель imgailab/sdxl-turbo-torch-int8 квантизована через TorchAO.")
        print("    Установите совместимую версию командой:")
        print("    pip install torchao==0.7.0")
        return

    # 3. Проверка CUDA
    if not torch.cuda.is_available():
        print("[-] ВНИМАНИЕ: PyTorch в текущем окружении установлен БЕЗ поддержки CUDA (CPU-only)!")
        print("    Модель imgailab/sdxl-turbo-torch-int8 была квантизована на GPU ('device': 'cuda').")
        print("    Квантизованные тензоры TorchAO INT8 не поддерживают запуск на CPU.")
        print("    Для запуска на вашей видеокарте (NVIDIA GeForce GTX 1660 Ti 6GB) выполните:")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return

    print(f"  [+] CUDA активна! GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

    # 4. Проверка diffusers
    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        print("[-] diffusers не установлен. Установите: pip install diffusers transformers accelerate")
        return

    # Применяем патч PyTorch 2.6
    apply_torchao_compatibility_patch(torch)

    print(f"  Загрузка пайплайна из {MODEL_PATH}...")
    start_load = time.time()
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.float16,
            variant=None
        )
        pipe.to("cuda")
        print(f"  [+] Модель успешно загружена за {time.time() - start_load:.2f}с")
    except Exception as e:
        print(f"[-] Ошибка загрузки модели: {e}")
        return

    print("\n=== [3/3] Генерация тестового изображения (1 шаг) ===")
    prompt = "a cute frog in cyberpunk style, glowing neon lights, 8k resolution"
    print(f"  Промпт: '{prompt}'")

    start_gen = time.time()
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

    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
    print(f"  [i] Пиковое потребление VRAM: {peak_vram:.2f} ГБ")
    if peak_vram <= 6.0:
        print("  [OK] Тест VRAM успешно пройден (< 6 ГБ)!")
    else:
        print("  [!] Внимание: потребление памяти превысило 6 ГБ")


if __name__ == "__main__":
    if verify_structure():
        test_inference()
