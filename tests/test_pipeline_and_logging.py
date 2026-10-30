import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import LOG_FILE_PATH, OUTPUT_DIR
from backend.app.logger import logger, get_memory_telemetry, track_model_execution
from backend.app.pipeline_manager import pipeline_manager
from backend.app.models_manager import models_manager


def test_telemetry():
    print("\n--- [1/4] Testing memory telemetry ---")
    telemetry = get_memory_telemetry()
    print(f"  GPU available: {telemetry['gpu_available']}")
    if telemetry['gpu_available']:
        print(f"  GPU Name     : {telemetry.get('gpu_name')}")
        print(f"  VRAM String  : {telemetry['vram_str']}")
    print(f"  RAM String   : {telemetry['ram_str']}")
    assert telemetry["ram_used_gb"] > 0, "RAM telemetry should be positive"


def test_track_model_execution_success():
    print("\n--- [2/4] Testing track_model_execution logging ---")
    with track_model_execution(
        category="txt2img",
        model_id="test-model",
        model_name="Test Model Unit",
        params={"prompt": "test prompt", "steps": 1, "size": "512x512"}
    ) as tracker:
        time.sleep(0.05)
        tracker.set_result("512x512 PNG Generated")

    # Flush loguru enqueue queue
    logger.complete()
    time.sleep(0.05)
    # Verify log file contains structured tags
    assert LOG_FILE_PATH.exists(), f"Log file must exist at {LOG_FILE_PATH}"
    log_content = LOG_FILE_PATH.read_text(encoding="utf-8")
    assert ">>> [MODEL_CALL]" in log_content
    assert "<<< [MODEL_SUCCESS]" in log_content
    print("  [OK] Telemetry SUCCESS block recorded in log.txt")


def test_track_model_execution_error():
    print("\n--- [3/4] Testing track_model_execution error recording ---")
    try:
        with track_model_execution(
            category="txt2img",
            model_id="error-model",
            model_name="Failing Model",
            params={"prompt": "fail"}
        ):
            raise RuntimeError("Simulated test inference failure")
    except RuntimeError:
        pass

    logger.complete()
    time.sleep(0.05)
    log_content = LOG_FILE_PATH.read_text(encoding="utf-8")
    assert "<<< [MODEL_ERROR]" in log_content
    assert "Simulated test inference failure" in log_content
    print("  [OK] Telemetry ERROR block recorded in log.txt")


def test_pipeline_manager_and_real_sdxl():
    print("\n--- [4/4] Testing PipelineManager warm cache & SDXL Turbo inference ---")
    
    # 1. First get_pipeline -> triggers warm load
    start_t = time.time()
    pipe1 = pipeline_manager.get_pipeline("txt2img", "sdxl-turbo-int8")
    load_time = time.time() - start_t
    print(f"  [+] Pipeline load time: {load_time:.2f}s")
    assert pipe1.is_loaded is True

    # 2. Second get_pipeline -> instant cache hit (0s)
    start_cached = time.time()
    pipe2 = pipeline_manager.get_pipeline("txt2img", "sdxl-turbo-int8")
    cache_hit_time = time.time() - start_cached
    print(f"  [+] Cached pipeline retrieval time: {cache_hit_time * 1000:.2f}ms")
    assert pipe1 is pipe2, "Cached pipeline instance should be identical"

    # 3. Generate image using the warm pipeline
    print("  [+] Running 1-step inference with warm pipeline...")
    with track_model_execution(
        category="txt2img",
        model_id="sdxl-turbo-int8",
        model_name="SDXL Turbo INT8",
        params={"prompt": "a ripe red apple on a wooden table, 8k", "steps": 1, "size": "512x512"}
    ) as tracker:
        img = pipe1.generate(
            prompt="a ripe red apple on a wooden table, 8k",
            steps=1,
            width=512,
            height=512,
            seed=42
        )
        tracker.set_result(f"{img.width}x{img.height} PNG")

    assert img.size == (512, 512)
    arr = np.array(img)
    assert not np.isnan(arr.mean()), "Image output should not contain NaNs"
    assert arr.max() > 0, "Image should not be completely black"

    out_file = OUTPUT_DIR / "test_pipeline_generation.png"
    img.save(out_file)
    print(f"  [OK] Real generation successful! Saved to: {out_file}")


if __name__ == "__main__":
    test_telemetry()
    test_track_model_execution_success()
    test_track_model_execution_error()
    test_pipeline_manager_and_real_sdxl()
    print("\nAll pipeline and telemetry tests PASSED successfully!")
