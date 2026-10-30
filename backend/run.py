import sys
import subprocess
from pathlib import Path

# Add project root to sys.path so "backend.app..." imports resolve cleanly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Auto-detect and switch to .venv python if started with system python lacking CUDA PyTorch
_venv_python = _root / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    try:
        import torch
        if not torch.cuda.is_available():
            raise ImportError("Torch installed without CUDA")
    except ImportError:
        print(f"[*] PepeShop: Re-launching with project virtualenv: {_venv_python}")
        res = subprocess.run([str(_venv_python)] + sys.argv)
        sys.exit(res.returncode)

import uvicorn
from backend.app.config import HOST, PORT, DEBUG

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host=HOST, port=PORT, reload=DEBUG)
