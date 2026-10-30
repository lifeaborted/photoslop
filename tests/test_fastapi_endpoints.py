import sys
from pathlib import Path
from PIL import Image
import io
import base64

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.api.models import list_models, get_active, switch_model, get_hardware, SwitchModelRequest
from backend.app.api.editor import paste_image, apply_adjustments, save_output, list_outputs, PasteRequest, AdjustmentsRequest, SaveImageRequest, pil_to_data_uri
from backend.app.api.generation import text_to_image, inpaint_image, text_edit_image, remove_background, TextToImageRequest, InpaintRequest, TextEditRequest, RemoveBackgroundRequest


def create_test_data_uri(color=(255, 100, 50), size=(128, 128)) -> str:
    img = Image.new("RGB", size, color=color)
    return pil_to_data_uri(img, format="PNG")


def test_models_api():
    print("\n[1/3] Testing Models API endpoints...")
    
    # 1. list_models
    res = list_models()
    assert res["status"] == "success"
    assert len(res["models"]) >= 4
    print(f"  [OK] list_models returned {len(res['models'])} models")

    # 2. get_active
    active = get_active()
    assert "txt2img" in active["active_models"]
    assert "inpaint" in active["active_models"]
    print("  [OK] get_active returned active models mapping")

    # 3. get_hardware
    hw = get_hardware()
    assert "vram_total_mb" in hw
    print(f"  [OK] get_hardware: GPU={hw['gpu_name']}, VRAM={hw['vram_used_mb']:.0f}MB")

    # 4. switch_model
    sw = switch_model(SwitchModelRequest(category="inpaint", model_id="flux1-fill-dev-q2"))
    assert sw["status"] == "success"
    assert sw["active_model"] == "flux1-fill-dev-q2"
    # Switch back
    switch_model(SwitchModelRequest(category="inpaint", model_id="sd15-inpainting-fp16"))
    print("  [OK] switch_model works dynamically")


def test_editor_api():
    print("\n[2/3] Testing Editor API endpoints...")
    data_uri = create_test_data_uri()

    # 1. paste
    paste_res = paste_image(PasteRequest(image_data=data_uri, name="TestPaste"))
    assert paste_res["status"] == "success"
    assert paste_res["layer"]["name"] == "TestPaste"
    print("  [OK] paste_image registered new layer")

    # 2. adjustments
    adj_res = apply_adjustments(AdjustmentsRequest(
        image_data=data_uri,
        brightness=1.2,
        contrast=1.1,
        saturation=1.3
    ))
    assert adj_res["status"] == "success"
    assert adj_res["width"] == 128
    print("  [OK] apply_adjustments processed colors")

    # 3. save & list outputs
    save_res = save_output(SaveImageRequest(
        image_data=data_uri,
        filename="test_api_export",
        format="PNG"
    ))
    assert save_res["status"] == "success"
    assert "test_api_export" in save_res["filename"]
    print(f"  [OK] save_output saved: {save_res['filename']}")

    outputs = list_outputs()
    assert outputs["status"] == "success"
    assert len(outputs["files"]) > 0
    print(f"  [OK] list_outputs found {len(outputs['files'])} export files")


def test_generation_api():
    print("\n[3/3] Testing AI Generation API endpoints...")
    data_uri = create_test_data_uri()
    mask_uri = create_test_data_uri(color=(255, 255, 255))

    # 1. remove-background
    bg_res = remove_background(RemoveBackgroundRequest(image_data=data_uri))
    assert bg_res["status"] == "success"
    assert bg_res["data_uri"].startswith("data:image/png;base64,")
    print("  [OK] remove_background returned RGBA data URI")

    # 2. inpaint
    inp_res = inpaint_image(InpaintRequest(
        image_data=data_uri,
        mask_data=mask_uri,
        prompt="vintage wooden bench",
        steps=10
    ))
    assert inp_res["status"] == "success"
    print(f"  [OK] inpaint completed in {inp_res['elapsed_seconds']}s")

    # 3. text-edit
    edit_res = text_edit_image(TextEditRequest(
        image_data=data_uri,
        prompt="retro vintage film look",
        strength=0.7
    ))
    assert edit_res["status"] == "success"
    print(f"  [OK] text_edit completed in {edit_res['elapsed_seconds']}s")

    # 4. txt2img with warm SDXL Turbo INT8
    txt_res = text_to_image(TextToImageRequest(
        prompt="a delicious glazed donut with colorful sprinkles, photorealistic",
        steps=1,
        width=512,
        height=512,
        seed=101
    ))
    assert txt_res["status"] == "success"
    assert txt_res["width"] == 512
    assert txt_res["height"] == 512
    print(f"  [OK] txt2img completed in {txt_res['elapsed_seconds']}s! Output width={txt_res['width']}")


if __name__ == "__main__":
    test_models_api()
    test_editor_api()
    test_generation_api()
    print("\nAll Backend API integration tests PASSED successfully!")
