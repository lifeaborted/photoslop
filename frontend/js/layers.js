export class Layer {
  constructor(id, name, width = 800, height = 600) {
    this.id = id;
    this.name = name;
    this.visible = true;
    this.opacity = 1.0;
    this.width = width;
    this.height = height;
    this.x = 0;
    this.y = 0;

    // Dedicated offscreen canvas for each layer
    this.canvas = document.createElement("canvas");
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext("2d");
  }

  setImage(img, x = 0, y = 0) {
    this.ctx.clearRect(0, 0, this.width, this.height);
    this.ctx.drawImage(img, x, y);
  }

  toDataURL(format = "image/png") {
    return this.canvas.toDataURL(format);
  }
}

export class LayerManager {
  constructor(canvasWidth = 800, canvasHeight = 600, onChangeCallback = null) {
    this.width = canvasWidth;
    this.height = canvasHeight;
    this.layers = [];
    this.activeLayerId = null;
    this.onChange = onChangeCallback || (() => {});
    this.layerCounter = 1;
  }

  createLayer(name = null, img = null) {
    const id = `layer_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
    const layerName = name || `Слой ${this.layerCounter++}`;
    const layer = new Layer(id, layerName, this.width, this.height);

    if (img) {
      // Center image if smaller or fit to canvas
      const scale = Math.min(this.width / img.width, this.height / img.height, 1);
      const dw = img.width * scale;
      const dh = img.height * scale;
      const dx = (this.width - dw) / 2;
      const dy = (this.height - dh) / 2;
      layer.ctx.drawImage(img, dx, dy, dw, dh);
    }

    this.layers.push(layer);
    this.activeLayerId = id;
    this.notify();
    return layer;
  }

  addBatchImages(imagesWithNames) {
    /**
     * Batch images insertion: distributes multiple images into separate layers automatically.
     */
    const createdLayers = [];
    imagesWithNames.forEach(({ name, img }) => {
      const layer = this.createLayer(name, img);
      createdLayers.push(layer);
    });
    this.notify();
    return createdLayers;
  }

  getActiveLayer() {
    return this.layers.find(l => l.id === this.activeLayerId) || this.layers[this.layers.length - 1];
  }

  setActiveLayer(id) {
    this.activeLayerId = id;
    this.notify();
  }

  removeLayer(id) {
    if (this.layers.length <= 1) return; // Keep at least one layer
    this.layers = this.layers.filter(l => l.id !== id);
    if (this.activeLayerId === id) {
      this.activeLayerId = this.layers[this.layers.length - 1].id;
    }
    this.notify();
  }

  toggleVisibility(id) {
    const layer = this.layers.find(l => l.id === id);
    if (layer) {
      layer.visible = !layer.visible;
      this.notify();
    }
  }

  setOpacity(id, opacity) {
    const layer = this.layers.find(l => l.id === id);
    if (layer) {
      layer.opacity = Math.max(0, Math.min(1, opacity));
      this.notify();
    }
  }

  moveLayerUp(id) {
    const idx = this.layers.findIndex(l => l.id === id);
    if (idx < this.layers.length - 1) {
      const temp = this.layers[idx];
      this.layers[idx] = this.layers[idx + 1];
      this.layers[idx + 1] = temp;
      this.notify();
    }
  }

  moveLayerDown(id) {
    const idx = this.layers.findIndex(l => l.id === id);
    if (idx > 0) {
      const temp = this.layers[idx];
      this.layers[idx] = this.layers[idx - 1];
      this.layers[idx - 1] = temp;
      this.notify();
    }
  }

  renderComposite(targetCtx) {
    targetCtx.clearRect(0, 0, this.width, this.height);
    for (const layer of this.layers) {
      if (layer.visible && layer.opacity > 0) {
        targetCtx.save();
        targetCtx.globalAlpha = layer.opacity;
        targetCtx.drawImage(layer.canvas, 0, 0);
        targetCtx.restore();
      }
    }
  }

  notify() {
    this.onChange(this);
  }
}
