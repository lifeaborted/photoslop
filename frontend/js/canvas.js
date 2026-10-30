export class CanvasEditor {
  constructor(canvasElement, layerManager) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext("2d");
    this.layerManager = layerManager;

    this.currentTool = "brush"; // brush, eraser, inpaint_mask, move
    this.brushColor = "#000000";
    this.brushSize = 20;

    // Mask overlay canvas for Inpaint
    this.maskCanvas = document.createElement("canvas");
    this.maskCanvas.width = this.canvas.width;
    this.maskCanvas.height = this.canvas.height;
    this.maskCtx = this.maskCanvas.getContext("2d");

    this.isDrawing = false;
    this.lastX = 0;
    this.lastY = 0;

    // Undo / Redo history
    this.historyStack = [];
    this.historyIndex = -1;
    this.maxHistory = 20;

    this.initEvents();
    this.saveState();
  }

  setTool(tool) {
    this.currentTool = tool;
  }

  setBrushSize(size) {
    this.brushSize = parseInt(size, 10);
  }

  setBrushColor(color) {
    this.brushColor = color;
  }

  clearMask() {
    this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    this.render();
  }

  getMaskDataURL() {
    // Generate black & white binary mask for inpaint backend
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = this.maskCanvas.width;
    tempCanvas.height = this.maskCanvas.height;
    const tempCtx = tempCanvas.getContext("2d");

    // Fill black
    tempCtx.fillStyle = "#000000";
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Get mask pixels
    const imgData = this.maskCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const data = imgData.data;

    const outData = tempCtx.createImageData(tempCanvas.width, tempCanvas.height);
    const outPixels = outData.data;

    for (let i = 0; i < data.length; i += 4) {
      const alpha = data[i + 3];
      const val = alpha > 20 ? 255 : 0;
      outPixels[i] = val;     // R
      outPixels[i + 1] = val; // G
      outPixels[i + 2] = val; // B
      outPixels[i + 3] = 255; // A
    }
    tempCtx.putImageData(outData, 0, 0);
    return tempCanvas.toDataURL("image/png");
  }

  initEvents() {
    this.canvas.addEventListener("mousedown", (e) => this.onMouseDown(e));
    window.addEventListener("mousemove", (e) => this.onMouseMove(e));
    window.addEventListener("mouseup", () => this.onMouseUp());
  }

  getPointerPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  onMouseDown(e) {
    if (e.button !== 0) return;
    this.isDrawing = true;
    const pos = this.getPointerPos(e);
    this.lastX = pos.x;
    this.lastY = pos.y;
    this.drawStroke(pos.x, pos.y);
  }

  onMouseMove(e) {
    if (!this.isDrawing) return;
    const pos = this.getPointerPos(e);
    this.drawStroke(pos.x, pos.y);
    this.lastX = pos.x;
    this.lastY = pos.y;
  }

  onMouseUp() {
    if (this.isDrawing) {
      this.isDrawing = false;
      this.saveState();
      this.layerManager.notify();
    }
  }

  drawStroke(x, y) {
    const activeLayer = this.layerManager.getActiveLayer();
    if (!activeLayer) return;

    if (this.currentTool === "inpaint_mask") {
      // Paint mask on maskCanvas
      this.maskCtx.save();
      this.maskCtx.strokeStyle = "rgba(255, 50, 50, 0.7)";
      this.maskCtx.fillStyle = "rgba(255, 50, 50, 0.7)";
      this.maskCtx.lineWidth = this.brushSize;
      this.maskCtx.lineCap = "round";
      this.maskCtx.lineJoin = "round";

      this.maskCtx.beginPath();
      this.maskCtx.moveTo(this.lastX, this.lastY);
      this.maskCtx.lineTo(x, y);
      this.maskCtx.stroke();
      this.maskCtx.restore();
    } else if (this.currentTool === "brush") {
      const ctx = activeLayer.ctx;
      ctx.save();
      ctx.strokeStyle = this.brushColor;
      ctx.fillStyle = this.brushColor;
      ctx.lineWidth = this.brushSize;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.globalCompositeOperation = "source-over";

      ctx.beginPath();
      ctx.moveTo(this.lastX, this.lastY);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.restore();
    } else if (this.currentTool === "eraser") {
      const ctx = activeLayer.ctx;
      ctx.save();
      ctx.lineWidth = this.brushSize;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.globalCompositeOperation = "destination-out";

      ctx.beginPath();
      ctx.moveTo(this.lastX, this.lastY);
      ctx.lineTo(x, y);
      ctx.stroke();
      ctx.restore();
    }

    this.render();
  }

  render() {
    // 1. Composite all active layers
    this.layerManager.renderComposite(this.ctx);

    // 2. Draw inpainting mask overlay if any
    this.ctx.drawImage(this.maskCanvas, 0, 0);
  }

  saveState() {
    const activeLayer = this.layerManager.getActiveLayer();
    if (!activeLayer) return;

    if (this.historyIndex < this.historyStack.length - 1) {
      this.historyStack = this.historyStack.slice(0, this.historyIndex + 1);
    }

    const state = activeLayer.ctx.getImageData(0, 0, activeLayer.width, activeLayer.height);
    this.historyStack.push({
      layerId: activeLayer.id,
      state
    });

    if (this.historyStack.length > this.maxHistory) {
      this.historyStack.shift();
    } else {
      this.historyIndex++;
    }
  }

  undo() {
    if (this.historyIndex > 0) {
      this.historyIndex--;
      const { layerId, state } = this.historyStack[this.historyIndex];
      const layer = this.layerManager.layers.find(l => l.id === layerId);
      if (layer) {
        layer.ctx.putImageData(state, 0, 0);
        this.render();
        this.layerManager.notify();
      }
    }
  }

  redo() {
    if (this.historyIndex < this.historyStack.length - 1) {
      this.historyIndex++;
      const { layerId, state } = this.historyStack[this.historyIndex];
      const layer = this.layerManager.layers.find(l => l.id === layerId);
      if (layer) {
        layer.ctx.putImageData(state, 0, 0);
        this.render();
        this.layerManager.notify();
      }
    }
  }

  getMergedDataURL(format = "image/png") {
    const outCanvas = document.createElement("canvas");
    outCanvas.width = this.canvas.width;
    outCanvas.height = this.canvas.height;
    const outCtx = outCanvas.getContext("2d");
    this.layerManager.renderComposite(outCtx);
    return outCanvas.toDataURL(format);
  }
}
