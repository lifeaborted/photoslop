const API_BASE = "";

export const api = {
  async getModels() {
    const res = await fetch(`${API_BASE}/api/models`);
    return await res.json();
  },

  async switchModel(category, modelId) {
    const res = await fetch(`${API_BASE}/api/models/switch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, model_id: modelId })
    });
    return await res.json();
  },

  async getHardware() {
    try {
      const res = await fetch(`${API_BASE}/api/models/hardware`);
      return await res.json();
    } catch {
      return null;
    }
  },

  async uploadBatch(files) {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    const res = await fetch(`${API_BASE}/api/editor/upload`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async pasteImage(imageData, name = "Pasted_Layer") {
    const res = await fetch(`${API_BASE}/api/editor/paste`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_data: imageData, name })
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async applyAdjustments(imageData, params) {
    const res = await fetch(`${API_BASE}/api/editor/adjustments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: imageData,
        brightness: params.brightness ?? 1.0,
        contrast: params.contrast ?? 1.0,
        saturation: params.saturation ?? 1.0,
        rotate_angle: params.rotate ?? 0.0,
        resize_width: params.width,
        resize_height: params.height
      })
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async saveOutput(imageData, filename, format = "PNG") {
    const res = await fetch(`${API_BASE}/api/editor/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: imageData,
        filename,
        format
      })
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async removeBackground(imageData, modelId) {
    const res = await fetch(`${API_BASE}/api/generation/remove-background`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: imageData,
        model_id: modelId
      })
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async inpaint(imageData, maskData, prompt, options = {}) {
    const res = await fetch(`${API_BASE}/api/generation/inpaint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: imageData,
        mask_data: maskData,
        prompt,
        negative_prompt: options.negativePrompt || "",
        steps: options.steps || 20,
        guidance_scale: options.guidanceScale || 7.5,
        seed: options.seed || null,
        model_id: options.modelId || null
      })
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async textEdit(imageData, prompt, options = {}) {
    const res = await fetch(`${API_BASE}/api/generation/text-edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data: imageData,
        prompt,
        strength: options.strength || 0.5,
        seed: options.seed || null,
        model_id: options.modelId || null
      })
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  },

  async txt2img(prompt, options = {}) {
    const res = await fetch(`${API_BASE}/api/generation/txt2img`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        negative_prompt: options.negativePrompt || "",
        width: options.width || 512,
        height: options.height || 512,
        steps: options.steps || 4,
        seed: options.seed || null,
        model_id: options.modelId || null
      })
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }
};
