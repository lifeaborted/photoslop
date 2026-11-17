import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.models_manager import ModelsManager
from backend.app.config import MODELS_CONFIG_PATH, MODELS_DIR

def test_models_manager_loading():
    mm = ModelsManager(config_path=MODELS_CONFIG_PATH, models_dir=MODELS_DIR)
    models = mm.get_models()
    assert len(models) >= 4, "Should have at least 4 models configured"
    
    # Check default active models
    active = mm.get_active_models()
    assert "inpaint" in active
    assert "background_removal" in active
    assert "txt2img" in active

def test_dynamic_switching():
    mm = ModelsManager(config_path=MODELS_CONFIG_PATH, models_dir=MODELS_DIR)
    
    # Switch inpaint model
    res = mm.switch_model("inpaint", "flux1-fill-dev-q2")
    assert res["status"] == "success"
    assert mm.active_models["inpaint"] == "flux1-fill-dev-q2"

    # Switch back to SDXL Inpaint
    res2 = mm.switch_model("inpaint", "sdxl-inpaint-0.1")
    assert res2["status"] == "success"
    assert mm.active_models["inpaint"] == "sdxl-inpaint-0.1"

def test_invalid_model_switch():
    mm = ModelsManager(config_path=MODELS_CONFIG_PATH, models_dir=MODELS_DIR)
    try:
        mm.switch_model("inpaint", "non-existent-model-id")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    test_models_manager_loading()
    test_dynamic_switching()
    test_invalid_model_switch()
    print("All models manager unit tests passed successfully!")
