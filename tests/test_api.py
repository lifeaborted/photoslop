import sys
from pathlib import Path
from PIL import Image
import io
import base64

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.api.editor import pil_to_data_uri, data_uri_to_pil
from backend.app.pipelines.rembg_pipeline import BackgroundRemovalPipeline
from backend.app.pipelines.inpaint_pipeline import InpaintPipeline
from backend.app.pipelines.txt2img_pipeline import TextToImagePipeline

def test_image_helpers():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    uri = pil_to_data_uri(img, format="PNG")
    assert uri.startswith("data:image/png;base64,")
    
    decoded = data_uri_to_pil(uri)
    assert decoded.size == (100, 100)

def test_pipelines_execution():
    test_img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    test_mask = Image.new("L", (256, 256), color=255)

    # 1. Background removal
    bg_pipe = BackgroundRemovalPipeline({"id": "test-rmbg", "name": "Test RMBG"})
    bg_pipe.load()
    res_bg = bg_pipe.process(test_img)
    assert res_bg.mode == "RGBA"

    # 2. Inpainting
    inpaint_pipe = InpaintPipeline({"id": "test-inpaint", "name": "Test Inpaint"})
    inpaint_pipe.load()
    res_inpaint = inpaint_pipe.inpaint(test_img, test_mask, prompt="cyberpunk neon car")
    assert res_inpaint.size == (256, 256)

    # 3. Txt2Img
    txt_pipe = TextToImagePipeline({"id": "test-txt2img", "name": "Test Txt2Img"})
    res_txt = txt_pipe.generate(prompt="beautiful landscape", width=256, height=256)
    assert res_txt.size == (256, 256)

if __name__ == "__main__":
    test_image_helpers()
    test_pipelines_execution()
    print("All pipeline unit tests passed successfully!")
