"""
Тест проверки и запуска SDXL-Turbo INT8 (imgailab/sdxl-turbo-torch-int8)
Проверяет структуру скачанных файлов, совместимость TorchAO/CUDA,
устраняет несовместимость с PyTorch 2.6+ (aten.alias, weights_only)
и предотвращает переполнение FP16 на GTX 1660 Ti (6GB VRAM).
"""
import sys
import time
from pathlib import Path
import numpy as np

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
    1. PyTorch 2.6 weights_only=False patch
    2. aten.alias.default dispatch patch for AffineQuantizedTensor
    3. FP32 accumulation patch for TorchAO INT8 linear layers to avoid
       FP16 integer overflow on NVIDIA Turing TU116 (GTX 1660 Ti / GTX 1650).
    """
    orig_load = torch.load

    def safe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        if not torch.cuda.is_available() and "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return orig_load(*args, **kwargs)

    torch.load = safe_load

    try:
        import torchao
        from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
        from torch.utils._python_dispatch import return_and_correct_aliasing

        # Регистрация безопасного глобала для сериализатора
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([AffineQuantizedTensor])

        # Патч для aten.alias.default (требуется для transformers/diffusers при загрузке)
        aten = torch.ops.aten

        @AffineQuantizedTensor.implements(aten.alias.default)
        def _alias(func, types, args, kwargs):
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        # Патч для предотвращения переполнения FP16 при умножении INT8 весов:
        # В исходном коде torchao умножение матрицы на целочисленные веса (int_data: -128..127)
        # происходит в FP16 ДО масштабирования (scale ~0.0005). При размерности 2048
        # сумма превышает предел FP16 (65504), вызывая переполнение в +inf и NaN.
        # Вычисление в float32 полностью устраняет переполнение без падения скорости.
        import torchao.dtypes.uintx.plain_layout as pl
        import torchao.dtypes.affine_quantized_tensor_ops as aqt_ops

        def safe_linear_fp_act_int8_weight_impl(input_tensor, weight_tensor, bias):
            w_vals_int8_t = weight_tensor.tensor_impl.int_data.t()
            scale = weight_tensor.tensor_impl.scale
            inp_flat = input_tensor.reshape(-1, input_tensor.shape[-1])
            m = torch.mm(inp_flat.float(), w_vals_int8_t.float())
            y = (m * scale.reshape(1, -1).float()).to(input_tensor.dtype)
            y = y.reshape(*input_tensor.shape[:-1], y.shape[-1])
            if bias is not None:
                y += bias.to(input_tensor.dtype)
            return y

        pl._linear_fp_act_int8_weight_impl = safe_linear_fp_act_int8_weight_impl
        for k in list(aqt_ops._AQT_QLINEAR_DISPATCH_TABLE.keys()):
            if "fp_act_int8" in k.__name__:
                aqt_ops._AQT_QLINEAR_DISPATCH_TABLE[k] = safe_linear_fp_act_int8_weight_impl

        print("  [+] Патчи совместимости TorchAO и FP32-аккумуляции успешно применены")
    except Exception as e:
        print(f"  [!] Предупреждение при применении патчей TorchAO: {e}")


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
        print("    Установите совместимую версию: pip install torchao==0.7.0")
        return

    # 3. Проверка CUDA
    if not torch.cuda.is_available():
        print("[-] ВНИМАНИЕ: PyTorch в текущем окружении установлен БЕЗ поддержки CUDA (CPU-only)!")
        return

    print(f"  [+] CUDA активна! GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

    # 4. Проверка diffusers
    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        print("[-] diffusers не установлен. Установите: pip install diffusers transformers accelerate")
        return

    # Применяем патчи совместимости
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

        # Защита VAE от переполнения FP16 на видеокартах Turing (GTX 1660 Ti)
        pipe.vae.to(torch.float32)
        orig_decode = pipe.vae.decode
        pipe.vae.decode = lambda z, *a, **kw: orig_decode(z.to(torch.float32), *a, **kw)

        load_time = time.time() - start_load
        print(f"  [+] Модель успешно загружена за {load_time:.2f}с")
    except Exception as e:
        print(f"[-] Ошибка загрузки модели: {e}")
        return

    print("\n=== [3/3] Генерация тестового изображения (1 шаг) ===")
    prompt = "a watermelon lying on table, 8k resolution"
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

    arr = np.array(image)
    print(f"  [+] Генерация завершена за {gen_time:.2f}с!")
    print(f"  [+] Статистика пикселей: min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")
    print(f"  [+] Результат сохранен в: {out_file}")

    if arr.max() > 0 and not np.isnan(arr.mean()):
        print("  [OK] Proverka validnosti izobrazheniya: USPESHNO (kartinka ne chernaya, bez NaN)")
    else:
        print("  [-] Vnimanie: izobrazhenie pustoe ili soderzhit artefakty!")

    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
    print(f"  [i] Пиковое потребление VRAM: {peak_vram:.2f} ГБ")
    if peak_vram <= 6.0:
        print("  [OK] Тест VRAM успешно пройден (< 6 ГБ)!")
    else:
        print("  [!] Внимание: потребление памяти превысило 6 ГБ")


if __name__ == "__main__":
    if verify_structure():
        test_inference()
