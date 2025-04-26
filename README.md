
# CUDA Programs

This folder contains several `.cu` files for various CUDA operations.

## How to Compile and Run (in VSCode Terminal)

1. **Open terminal** inside this folder.

2. **Compile a file** (example: `vecAdd.cu`):
   ```bash
   nvcc vecAdd.cu -o vecAdd
   ```

3. **Run the executable**:
   ```bash
   ./vecAdd
   ```

---

## Notes
- Make sure you have **CUDA toolkit** installed (`nvcc --version` to check).
- If needed, add compiler flags like `-arch=sm_75` for newer GPUs.

---
