import { api } from "./api.js";
import { LayerManager } from "./layers.js";
import { CanvasEditor } from "./canvas.js";

class PepeShopApp {
  constructor() {
    this.canvasEl = document.getElementById("editor-canvas");
    this.layerManager = new LayerManager(800, 600, (lm) => this.onLayersChanged(lm));
    this.editor = new CanvasEditor(this.canvasEl, this.layerManager);

    this.modelsData = null;
    this.activeCategory = "inpaint";

    this.initUI();
    this.initClipboardAndDrop();
    this.loadModelsAndHardware();

    // Create default base white background layer
    const baseLayer = this.layerManager.createLayer("Фон (Базовый)");
    baseLayer.ctx.fillStyle = "#ffffff";
    baseLayer.ctx.fillRect(0, 0, 800, 600);
    this.editor.render();
  }

  async loadModelsAndHardware() {
    try {
      const data = await api.getModels();
      this.modelsData = data;
      this.renderModelSelectors(data);

      const hw = await api.getHardware();
      this.renderHardwareBadge(hw);
    } catch (err) {
      console.error("Failed to load models or hardware:", err);
    }
  }

  renderHardwareBadge(hw) {
    const badgeEl = document.getElementById("hw-status-badge");
    if (!badgeEl) return;
    if (hw && hw.gpu_available) {
      badgeEl.innerHTML = `
        <span class="hw-dot"></span>
        <span><b>GPU:</b> ${hw.gpu_name} (${hw.vram_used_mb}MB / ${hw.vram_total_mb}MB - ${hw.vram_percent}%)</span>
      `;
    } else {
      badgeEl.innerHTML = `
        <span class="hw-dot" style="background: var(--warning)"></span>
        <span><b>CPU Mode</b> (No GPU detected)</span>
      `;
    }
  }

  renderModelSelectors(data) {
    const selectInpaint = document.getElementById("select-inpaint-model");
    const selectBg = document.getElementById("select-bg-model");
    const selectTxt = document.getElementById("select-txt-model");

    if (!selectInpaint || !selectBg || !selectTxt) return;

    selectInpaint.innerHTML = "";
    selectBg.innerHTML = "";
    selectTxt.innerHTML = "";

    const active = data.active_models || {};

    data.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      const dlMark = m.is_downloaded ? "✓ " : "[HF] ";
      opt.textContent = `${dlMark}${m.name} (${m.precision.toUpperCase()})`;

      if (m.category === "inpaint" || m.category === "edit") {
        opt.selected = active.inpaint?.id === m.id || active.edit?.id === m.id;
        selectInpaint.appendChild(opt);
      } else if (m.category === "background_removal") {
        opt.selected = active.background_removal?.id === m.id;
        selectBg.appendChild(opt);
      } else if (m.category === "txt2img") {
        opt.selected = active.txt2img?.id === m.id;
        selectTxt.appendChild(opt);
      }
    });

    // Dynamic switch listeners
    selectInpaint.onchange = async () => {
      await api.switchModel("inpaint", selectInpaint.value);
      this.showToast(`Модель инпейнтинга переключена: ${selectInpaint.value}`);
    };
    selectBg.onchange = async () => {
      await api.switchModel("background_removal", selectBg.value);
      this.showToast(`Модель удаления фона переключена: ${selectBg.value}`);
    };
    selectTxt.onchange = async () => {
      await api.switchModel("txt2img", selectTxt.value);
      this.showToast(`Модель генерации переключена: ${selectTxt.value}`);
    };
  }

  initUI() {
    // Toolbar tool selection
    const toolBtns = document.querySelectorAll("[data-tool]");
    toolBtns.forEach((btn) => {
      btn.onclick = () => {
        toolBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const tool = btn.getAttribute("data-tool");
        this.editor.setTool(tool);
        this.showToast(`Выбран инструмент: ${tool}`);
      };
    });

    // Brush controls
    const brushSize = document.getElementById("brush-size");
    if (brushSize) {
      brushSize.oninput = (e) => this.editor.setBrushSize(e.target.value);
    }
    const brushColor = document.getElementById("brush-color");
    if (brushColor) {
      brushColor.oninput = (e) => this.editor.setBrushColor(e.target.value);
    }

    // Undo / Redo
    document.getElementById("btn-undo")?.addEventListener("click", () => this.editor.undo());
    document.getElementById("btn-redo")?.addEventListener("click", () => this.editor.redo());

    // Clear Mask
    document.getElementById("btn-clear-mask")?.addEventListener("click", () => {
      this.editor.clearMask();
      this.showToast("Маска инпейнтинга очищена");
    });

    // Batch File Upload Dialog Button
    const fileInput = document.getElementById("file-input");
    const btnOpenDialog = document.getElementById("btn-open-dialog");
    if (btnOpenDialog && fileInput) {
      btnOpenDialog.onclick = () => fileInput.click();
      fileInput.onchange = async (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
          await this.handleBatchFiles(files);
          fileInput.value = "";
        }
      };
    }

    // Background Removal Button
    document.getElementById("btn-remove-bg")?.addEventListener("click", async () => {
      await this.handleRemoveBackground();
    });

    // Inpaint Action Button
    document.getElementById("btn-run-inpaint")?.addEventListener("click", async () => {
      await this.handleInpaint();
    });

    // Text-Guided Edit Button
    document.getElementById("btn-run-edit")?.addEventListener("click", async () => {
      await this.handleTextEdit();
    });

    // Txt2Img Button
    document.getElementById("btn-run-txt2img")?.addEventListener("click", async () => {
      await this.handleTxt2Img();
    });

    // Adjustments
    const sliderBright = document.getElementById("adj-brightness");
    const sliderContrast = document.getElementById("adj-contrast");
    const sliderSat = document.getElementById("adj-saturation");
    const btnApplyAdj = document.getElementById("btn-apply-adj");

    if (btnApplyAdj) {
      btnApplyAdj.onclick = async () => {
        await this.handleAdjustments({
          brightness: parseFloat(sliderBright?.value || 1.0),
          contrast: parseFloat(sliderContrast?.value || 1.0),
          saturation: parseFloat(sliderSat?.value || 1.0)
        });
      };
    }

    // Export / Save Image to /output
    document.getElementById("btn-save-output")?.addEventListener("click", async () => {
      await this.handleSaveOutput("PNG");
    });
    document.getElementById("btn-save-jpg")?.addEventListener("click", async () => {
      await this.handleSaveOutput("JPG");
    });

    // Add Blank Layer
    document.getElementById("btn-add-layer")?.addEventListener("click", () => {
      this.layerManager.createLayer();
    });
  }

  initClipboardAndDrop() {
    // 1. Paste (Ctrl+V) listener
    window.addEventListener("paste", async (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          const file = items[i].getAsFile();
          if (file) {
            this.showToast("Вставка изображения из буфера обмена (Ctrl+V)...");
            const reader = new FileReader();
            reader.onload = async (event) => {
              const dataUrl = event.target.result;
              const res = await api.pasteImage(dataUrl, `Вставка_${Date.now()}`);
              const img = new Image();
              img.onload = () => {
                this.layerManager.createLayer(res.layer.name, img);
                this.showToast(`Создан новый слой: ${res.layer.name}`);
              };
              img.src = dataUrl;
            };
            reader.readAsDataURL(file);
          }
        }
      }
    });

    // 2. Drag & Drop
    const viewport = document.querySelector(".canvas-viewport");
    const dropzone = document.getElementById("dropzone-overlay");

    if (viewport && dropzone) {
      ["dragenter", "dragover"].forEach((evt) => {
        viewport.addEventListener(evt, (e) => {
          e.preventDefault();
          dropzone.classList.add("active");
        });
      });

      ["dragleave", "drop"].forEach((evt) => {
        viewport.addEventListener(evt, (e) => {
          e.preventDefault();
          dropzone.classList.remove("active");
        });
      });

      viewport.addEventListener("drop", async (e) => {
        const files = Array.from(e.dataTransfer?.files || []).filter((f) =>
          f.type.startsWith("image/")
        );
        if (files.length > 0) {
          await this.handleBatchFiles(files);
        }
      });
    }
  }

  async handleBatchFiles(files) {
    this.showToast(`Загрузка пачки из ${files.length} изображений...`);
    try {
      const res = await api.uploadBatch(files);
      const imagesWithNames = [];

      for (const layerData of res.layers) {
        await new Promise((resolve) => {
          const img = new Image();
          img.onload = () => {
            imagesWithNames.push({ name: layerData.name, img });
            resolve();
          };
          img.src = layerData.data_uri;
        });
      }

      // Automatically distribute batch images into separate layers!
      this.layerManager.addBatchImages(imagesWithNames);
      this.showToast(`Успешно распределено ${imagesWithNames.length} фото по отдельным слоям!`);
    } catch (err) {
      console.error(err);
      this.showToast(`Ошибка загрузки пачки: ${err.message}`, true);
    }
  }

  async handleRemoveBackground() {
    const activeLayer = this.layerManager.getActiveLayer();
    if (!activeLayer) return;

    this.showToast("Удаление фона с активного слоя (AI RMBG)...");
    this.setLoading(true);
    try {
      const activeModel = document.getElementById("select-bg-model")?.value;
      const layerDataUri = activeLayer.toDataURL("image/png");
      const res = await api.removeBackground(layerDataUri, activeModel);

      const img = new Image();
      img.onload = () => {
        activeLayer.setImage(img);
        this.editor.render();
        this.editor.saveState();
        this.layerManager.notify();
        this.setLoading(false);
        this.showToast(`Фон удалён за ${res.elapsed_seconds}с (${res.model_name})`);
      };
      img.src = res.data_uri;
    } catch (err) {
      this.setLoading(false);
      this.showToast(`Ошибка удаления фона: ${err.message}`, true);
    }
  }

  async handleInpaint() {
    const activeLayer = this.layerManager.getActiveLayer();
    const prompt = document.getElementById("ai-prompt")?.value?.trim();
    if (!prompt) {
      this.showToast("Введите текстовый промпт для инпейнтинга!", true);
      return;
    }

    const maskDataUri = this.editor.getMaskDataURL();
    const layerDataUri = activeLayer.toDataURL("image/png");
    const activeModel = document.getElementById("select-inpaint-model")?.value;

    this.showToast(`Инпейнтинг [${activeModel}]: "${prompt}"...`);
    this.setLoading(true);

    try {
      const res = await api.inpaint(layerDataUri, maskDataUri, prompt, {
        negativePrompt: document.getElementById("ai-neg-prompt")?.value,
        steps: parseInt(document.getElementById("ai-steps")?.value || 20, 10),
        seed: parseInt(document.getElementById("ai-seed")?.value || -1, 10),
        modelId: activeModel
      });

      const img = new Image();
      img.onload = () => {
        activeLayer.setImage(img);
        this.editor.clearMask();
        this.editor.render();
        this.editor.saveState();
        this.layerManager.notify();
        this.setLoading(false);
        this.showToast(`Инпейнтинг выполнен за ${res.elapsed_seconds}с!`);
      };
      img.src = res.data_uri;
    } catch (err) {
      this.setLoading(false);
      this.showToast(`Ошибка инпейнтинга: ${err.message}`, true);
    }
  }

  async handleTextEdit() {
    const activeLayer = this.layerManager.getActiveLayer();
    const prompt = document.getElementById("ai-prompt")?.value?.trim();
    if (!prompt) {
      this.showToast("Введите промпт для редактирования!", true);
      return;
    }

    const layerDataUri = activeLayer.toDataURL("image/png");
    const activeModel = document.getElementById("select-inpaint-model")?.value;

    this.showToast(`Редактирование по промпту: "${prompt}"...`);
    this.setLoading(true);

    try {
      const res = await api.textEdit(layerDataUri, prompt, {
        strength: 0.6,
        modelId: activeModel
      });

      const img = new Image();
      img.onload = () => {
        activeLayer.setImage(img);
        this.editor.render();
        this.editor.saveState();
        this.layerManager.notify();
        this.setLoading(false);
        this.showToast(`Текстовая правка выполнена за ${res.elapsed_seconds}с!`);
      };
      img.src = res.data_uri;
    } catch (err) {
      this.setLoading(false);
      this.showToast(`Ошибка редактирования: ${err.message}`, true);
    }
  }

  async handleTxt2Img() {
    const prompt = document.getElementById("ai-prompt")?.value?.trim();
    if (!prompt) {
      this.showToast("Введите промпт для генерации!", true);
      return;
    }

    const activeModel = document.getElementById("select-txt-model")?.value;
    this.showToast(`Генерация новой картинки: "${prompt}"...`);
    this.setLoading(true);

    try {
      const res = await api.txt2img(prompt, {
        modelId: activeModel,
        width: 800,
        height: 600
      });

      const img = new Image();
      img.onload = () => {
        const newLayer = this.layerManager.createLayer(`Gen: ${prompt.slice(0, 15)}`, img);
        this.setLoading(false);
        this.showToast(`Сгенерировано за ${res.elapsed_seconds}с в новый слой!`);
      };
      img.src = res.data_uri;
    } catch (err) {
      this.setLoading(false);
      this.showToast(`Ошибка генерации: ${err.message}`, true);
    }
  }

  async handleAdjustments(params) {
    const activeLayer = this.layerManager.getActiveLayer();
    if (!activeLayer) return;

    this.showToast("Применение коррекции цвета...");
    try {
      const layerDataUri = activeLayer.toDataURL("image/png");
      const res = await api.applyAdjustments(layerDataUri, params);

      const img = new Image();
      img.onload = () => {
        activeLayer.setImage(img);
        this.editor.render();
        this.editor.saveState();
        this.layerManager.notify();
        this.showToast("Цветокоррекция применена!");
      };
      img.src = res.data_uri;
    } catch (err) {
      this.showToast(`Ошибка коррекции: ${err.message}`, true);
    }
  }

  async handleSaveOutput(format = "PNG") {
    this.showToast(`Экспорт в /output (${format})...`);
    try {
      const mergedDataUri = this.editor.getMergedDataURL(
        format === "JPG" ? "image/jpeg" : "image/png"
      );
      const res = await api.saveOutput(mergedDataUri, null, format);
      this.showToast(`Файл успешно сохранён: ${res.filename} (${res.size_kb} KB)`);
    } catch (err) {
      this.showToast(`Ошибка сохранения: ${err.message}`, true);
    }
  }

  onLayersChanged(lm) {
    this.editor.render();
    const container = document.getElementById("layers-list");
    if (!container) return;

    container.innerHTML = "";
    // Render top-to-bottom (highest index rendered on top)
    const reversed = [...lm.layers].reverse();

    reversed.forEach((layer) => {
      const item = document.createElement("div");
      item.className = `layer-item ${layer.id === lm.activeLayerId ? "active" : ""}`;

      const thumb = document.createElement("img");
      thumb.className = "layer-thumb";
      thumb.src = layer.toDataURL("image/png");

      const info = document.createElement("div");
      info.className = "layer-info";
      info.innerHTML = `
        <div class="layer-name">${layer.name}</div>
        <div class="layer-meta">Непрозрачность: ${Math.round(layer.opacity * 100)}%</div>
      `;

      const actions = document.createElement("div");
      actions.className = "layer-actions";

      // Eye visibility toggle
      const btnEye = document.createElement("button");
      btnEye.className = "layer-action-btn";
      btnEye.innerHTML = layer.visible ? "👁️" : "🕶️";
      btnEye.title = "Видимость";
      btnEye.onclick = (e) => {
        e.stopPropagation();
        lm.toggleVisibility(layer.id);
      };

      // Delete layer
      const btnDel = document.createElement("button");
      btnDel.className = "layer-action-btn";
      btnDel.innerHTML = "🗑️";
      btnDel.title = "Удалить слой";
      btnDel.onclick = (e) => {
        e.stopPropagation();
        lm.removeLayer(layer.id);
      };

      actions.appendChild(btnEye);
      actions.appendChild(btnDel);

      item.appendChild(thumb);
      item.appendChild(info);
      item.appendChild(actions);

      item.onclick = () => lm.setActiveLayer(layer.id);
      container.appendChild(item);
    });
  }

  showToast(msg, isError = false) {
    const statusText = document.getElementById("status-text");
    if (statusText) {
      statusText.textContent = msg;
      statusText.style.color = isError ? "var(--danger)" : "var(--text-main)";
    }
  }

  setLoading(isLoading) {
    const spinner = document.getElementById("main-spinner");
    if (spinner) {
      spinner.style.display = isLoading ? "inline-block" : "none";
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.app = new PepeShopApp();
});
