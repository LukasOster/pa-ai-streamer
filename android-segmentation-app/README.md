# Real-Time RF-DETR Object Detection Android App

Android application for real-time object detection using RF-DETR with ONNX Runtime inference on live camera stream.

## Features

- Real-time camera preview with CameraX
- **ONNX Runtime** inference (preferred for RF-DETR)
- Alternative TFLite inference support
- Configurable frame skipping for performance (N=1/2/3)
- Rolling-average FPS counter
- Hardware acceleration support

## Current Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Camera Thread  │ ──► │ Inference Thread │ ──► │  Render Thread  │
│  (CameraX)      │     │  (ONNX Runtime)  │     │  (SurfaceView)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Project Structure

```
android-segmentation-app/
├── app/src/main/
│   ├── assets/
│   │   └── rf_detr_detection.onnx    # ONNX model (~115MB)
│   ├── java/com/segmentation/app/
│   │   ├── MainActivity.kt           # Main activity, orchestration
│   │   ├── ONNXModel.kt              # ONNX Runtime inference wrapper
│   │   ├── CameraManager.kt          # CameraX frame extraction
│   │   ├── OverlayRenderer.kt        # Bounding box rendering
│   │   └── TFLiteModel.kt            # TFLite alternative (legacy)
│   └── res/
└── build.gradle.kts
```

## Quick Start

### 1. Build and Run

1. Open `android-segmentation-app` in Android Studio
2. Connect Android device (USB debugging enabled)
3. Click Run or press `Shift+F10`
4. Grant camera permission

### 2. Default Model

The app includes RF-DETR Medium pretrained on COCO (80 object classes).

---

# Integrating Custom Models

## Prerequisites

- Python 3.8+
- PyTorch
- RF-DETR library (`pip install rfdetr`)
- ONNX (`pip install onnx onnxruntime`)

## Step 1: Train Your Model

Train RF-DETR on your custom dataset. See [RF-DETR documentation](https://github.com/roboflow/rf-detr).

Your trained model will produce a `.pth` weights file.

## Step 2: Export to ONNX

### 2.1 Configure Export Script

Edit `model-export/export_to_tflite_v2.py`:

```python
# =============================================================================
# CONFIGURATION - MODIFY THESE FOR YOUR MODEL
# =============================================================================

MODEL_TYPE = "medium"      # Options: "nano", "small", "medium", "base", "large"
NUM_CLASSES = 5            # Your number of classes (INCLUDING background if applicable)
WEIGHTS_PATH = "my_custom_model.pth"  # Path to your trained weights

INPUT_HEIGHT = 512         # Model input size
INPUT_WIDTH = 512

ONNX_PATH = "my_custom_model.onnx"  # Output filename
```

### 2.2 Run Export

```bash
cd model-export

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch rfdetr onnx onnxruntime

# Run export
python export_to_tflite_v2.py
```

### 2.3 Verify Export

The script will output verification results:

```
Boxes shape: (300, 4)
Boxes range: [0.0065, 1.0311]  # Should NOT be all zeros
Logits shape: (300, N)         # N = your num_classes
[OK] ONNX boxes have valid values
```

## Step 3: Deploy to Android

### 3.1 Copy Model to Assets

```bash
cp my_custom_model.onnx ../android-segmentation-app/app/src/main/assets/
```

### 3.2 Update MainActivity.kt

```kotlin
companion object {
    private const val ONNX_MODEL_FILE = "my_custom_model.onnx"  // Your model filename
    private const val USE_ONNX = true  // Use ONNX Runtime
}
```

### 3.3 Update Class Labels

**For custom models with continuous class IDs (0 to N-1):**

```kotlin
private val classLabels = listOf(
    "Background",    // Class 0 (if your model has background class)
    "Defect_A",      // Class 1
    "Defect_B",      // Class 2
    "Defect_C",      // Class 3
    // ... add all your classes
)
```

**For COCO-pretrained models (IDs with gaps):**

```kotlin
// COCO uses non-consecutive IDs: 1,2,3...11, 13,14...
// Gaps at: 12, 26, 29, 30, 45, 66, 68, 69, 71, 83
private val cocoClassMap = mapOf(
    0 to "Background",
    1 to "Person", 2 to "Bicycle", 3 to "Car",
    // ... see MainActivity.kt for full mapping
    72 to "TV", 73 to "Laptop", 74 to "Mouse",
)
private val classLabels = (0..90).map { cocoClassMap[it] ?: "Class $it" }
```

### 3.4 Adjust Confidence Threshold

```kotlin
onnxModel = ONNXModel(this).apply {
    classLabels = this@MainActivity.classLabels
    confidenceThreshold = 0.1f  // RF-DETR typically needs lower threshold (0.1-0.3)
}
```

### 3.5 Rebuild App

Build and deploy from Android Studio.

---

## Model Input/Output Specification

### Input Format

| Property | Value |
|----------|-------|
| Shape | `[1, 3, 512, 512]` (NCHW) |
| Type | Float32 |
| Channel Order | RGB |
| Normalization | ImageNet: `(pixel/255 - mean) / std` |
| Mean | `[0.485, 0.456, 0.406]` |
| Std | `[0.229, 0.224, 0.225]` |

### Output Format

| Output | Shape | Description |
|--------|-------|-------------|
| `boxes` | `[300, 4]` | Normalized CXCYWH coordinates |
| `logits` | `[300, num_classes]` | Pre-sigmoid class logits |

### Post-processing

```
confidence = sigmoid(max_logit)

# Convert CXCYWH to corners
left = cx - w/2
top = cy - h/2
right = cx + w/2
bottom = cy + h/2
```

---

## Configuration Options

### ONNXModel Settings

```kotlin
onnxModel = ONNXModel(this).apply {
    confidenceThreshold = 0.1f      // Detection threshold (0.0 - 1.0)
    classLabels = listOf(...)       // Class names
}
```

### OverlayRenderer Settings

```kotlin
renderer = OverlayRenderer(binding.overlayView).apply {
    boxStrokeWidth = 6f             // Bounding box line width
    labelTextSize = 40f             // Label font size
    showLabels = true               // Show class names
    showConfidence = true           // Show confidence percentages
    maskAlpha = 0.4f                // For segmentation mode
}
```

### Performance Tuning

```kotlin
// Process every Nth frame (higher = faster but less responsive)
cameraManager.setFrameSkip(2)  // Options: 1, 2, 3
```

---

## Troubleshooting

### App crashes on startup

**Symptom**: `OutOfMemoryError` or immediate crash

**Solution**:
1. Add to `AndroidManifest.xml`:
   ```xml
   <application android:largeHeap="true" ...>
   ```
2. Model is loaded to cache first to avoid memory issues

### No detections showing

**Possible causes**:
1. `confidenceThreshold` too high - try lowering to 0.05f
2. Wrong class labels - verify mapping matches model output
3. Model weights not loaded during export - check export logs for "[OK] Loaded weights"

### Wrong class labels

**For COCO models**: Use the ID mapping with gaps (see above)
**For custom models**: Use continuous 0 to N-1 indexing

### Native crash (SIGABRT)

**Symptom**: `Scudo ERROR: invalid chunk state`

**Cause**: Double-free of ONNX tensors

**Solution**: Only call `outputs.close()`, don't manually close individual tensors

### Low FPS / Slow inference

1. Increase frame skip to 3
2. Use smaller model variant (nano, small)
3. Reduce input resolution
4. Consider INT8 quantization

### All-zero bounding boxes (TFLite)

**Cause**: RF-DETR's reference point computation breaks when `return_intermediate=False`

**Solution**: Use ONNX Runtime instead of TFLite (set `USE_ONNX = true`)

---

## Performance Characteristics

| Model | Size | Inference Time* | Memory |
|-------|------|-----------------|--------|
| RF-DETR Nano | ~25MB | ~150ms | ~200MB |
| RF-DETR Small | ~50MB | ~300ms | ~300MB |
| RF-DETR Medium | ~115MB | ~700ms | ~500MB |
| RF-DETR Base | ~200MB | ~1200ms | ~800MB |

*On mid-range Android device (CPU inference)

**Note**: RF-DETR produces lower confidence scores (10-40%) compared to YOLO/SSD. This is normal behavior for this architecture.

---

## Requirements

- **Android**: API 26+ (Android 8.0 Oreo)
- **Storage**: ~200MB (app + model)
- **RAM**: 2GB+ recommended

## Dependencies

| Library | Version |
|---------|---------|
| ONNX Runtime | 1.16.3 |
| CameraX | 1.3.1 |
| TensorFlow Lite | 2.13.0 (optional) |
| Kotlin | 1.9.x |

---

## License

MIT License

## Acknowledgments

- [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow
- [ONNX Runtime](https://onnxruntime.ai/)
- [CameraX](https://developer.android.com/training/camerax)
