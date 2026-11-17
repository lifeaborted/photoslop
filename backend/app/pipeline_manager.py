from typing import Dict, Any, Optional
from backend.app.logger import logger
from backend.app.models_manager import models_manager
from backend.app.pipelines.base import BasePipeline
from backend.app.pipelines.txt2img_pipeline import TextToImagePipeline
from backend.app.pipelines.inpaint_pipeline import InpaintPipeline
from backend.app.pipelines.rembg_pipeline import BackgroundRemovalPipeline


class PipelineManager:
    """
    Manages active deep learning pipelines in memory.
    Enforces memory safety for 6GB VRAM GPUs (GTX 1660 Ti):
    - Caches the warm pipeline to eliminate reloading overhead (~2s inference).
    - Automatically unloads preceding heavy models before loading a new one to prevent OOM.
    """

    def __init__(self):
        self._active_pipelines: Dict[str, BasePipeline] = {}

    def get_pipeline(self, category: str, model_id: Optional[str] = None) -> BasePipeline:
        """
        Retrieves or initializes a warm loaded pipeline for the given category.
        If the model is already loaded and ready in memory, returns it immediately.
        """
        if not model_id:
            model_id = models_manager.active_models.get(category)
            if not model_id:
                raise ValueError(f"No active model specified for category: '{category}'")

        # 1. Check if warm pipeline is already loaded
        existing = self._active_pipelines.get(category)
        if existing and existing.is_loaded and existing.model_info.get("id") == model_id:
            return existing

        # 2. Get model definition
        model_info = models_manager.get_model_by_id(model_id)
        if not model_info:
            model_info = {"id": model_id, "name": model_id, "category": category}

        logger.info(
            f"PipelineManager: preparing pipeline for [{category}] with model '{model_id}' ({model_info.get('name')})"
        )

        # 3. VRAM hygiene: Unload any other heavy GPU pipeline before loading a new one
        # Especially important on 6GB VRAM Turing GPUs
        self._free_vram_for_new_model(category, model_id)

        # 4. Instantiate pipeline
        pipeline = self._create_pipeline_instance(category, model_info)

        # 5. Load model weights into VRAM/RAM
        success = pipeline.load()
        if not success:
            logger.warning(f"Pipeline load returned False for [{model_id}]. Operating in fallback mode.")

        self._active_pipelines[category] = pipeline
        models_manager.active_models[category] = model_id
        return pipeline

    def switch_model(self, category: str, model_id: str) -> None:
        """
        Handles dynamic model switch request.
        Unloads current pipeline in this category and marks new active model.
        """
        if category in self._active_pipelines:
            curr = self._active_pipelines[category]
            if curr.model_info.get("id") != model_id:
                logger.info(
                    f"PipelineManager: switching category '{category}' from {curr.model_info.get('id')} to {model_id}"
                )
                curr.unload()
                del self._active_pipelines[category]
                self._empty_torch_cuda_cache()

        models_manager.switch_model(category, model_id)

    def unload_all(self) -> None:
        """Unloads all active pipelines and clears GPU VRAM."""
        logger.info("PipelineManager: unloading all loaded pipelines...")
        for cat, pipe in list(self._active_pipelines.items()):
            try:
                pipe.unload()
            except Exception as e:
                logger.warning(f"Error unloading pipeline for [{cat}]: {e}")
        self._active_pipelines.clear()
        self._empty_torch_cuda_cache()

    def _free_vram_for_new_model(self, target_category: str, target_model_id: str):
        """Unloads pipelines that compete for heavy VRAM."""
        for cat, pipe in list(self._active_pipelines.items()):
            # If changing model in same category, or if switching between heavy generative pipelines
            if cat == target_category or cat in ("txt2img", "inpaint", "edit"):
                logger.info(
                    f"PipelineManager: unloading previous pipeline [{cat}: {pipe.model_info.get('id')}] to free VRAM"
                )
                pipe.unload()
                del self._active_pipelines[cat]

        self._empty_torch_cuda_cache()

    def _empty_torch_cuda_cache(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _create_pipeline_instance(self, category: str, model_info: Dict[str, Any]) -> BasePipeline:
        if category == "txt2img":
            return TextToImagePipeline(model_info)
        elif category in ("inpaint", "edit", "object_removal"):
            return InpaintPipeline(model_info)
        elif category == "background_removal":
            return BackgroundRemovalPipeline(model_info)
        else:
            raise ValueError(f"Unknown pipeline category: '{category}'")


pipeline_manager = PipelineManager()
