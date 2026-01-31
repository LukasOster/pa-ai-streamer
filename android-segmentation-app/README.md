# Real-Time ML Inference Android App

Android-App für Echtzeit-ML-Inferenz mit TensorFlow Lite auf dem Live-Kamera-Stream.

**Unterstützt beide Modi:**
- **Object Detection** - Bounding Boxes mit Labels
- **Semantic Segmentation** - Pixel-weise Klassenmasken

Der Modus wird automatisch basierend auf dem Output-Tensor-Shape erkannt.

## Features

- Live-Kamera-Preview mit CameraX
- TensorFlow Lite Inference mit Hardware-Beschleunigung (GPU/NNAPI/CPU)
- **Object Detection**: Farbige Bounding Boxes mit Klassen-Labels und Konfidenz
- **Segmentation**: Halbtransparentes farbiges Overlay
- Frame-Skipping für optimale Performance (N=1/2/3)
- Rolling-Average FPS-Counter
- Umschaltbare Delegates zur Laufzeit

## Projektstruktur

```
app/src/main/java/com/segmentation/app/
├── MainActivity.kt      # Haupt-Activity, koordiniert alle Komponenten
├── CameraManager.kt     # CameraX Setup, Frame-Extraktion, Threading
├── TFLiteModel.kt       # Unified TFLite Inference (Detection + Segmentation)
└── OverlayRenderer.kt   # Rendert Boxes ODER Masken auf SurfaceView
```

## Setup

### 1. Model-Datei hinzufügen

Kopieren Sie Ihre TFLite-Modell-Datei in den Assets-Ordner:

```bash
cp your_model.tflite app/src/main/assets/rf_detr_segmentation.tflite
```

### 2. Model-Anforderungen

**Input (beide Modi):**
- Shape: `[1, H, W, 3]` (NHWC Format)
- Typ: Float32
- Werte: RGB, normalisiert auf `[0, 1]`

**Output - Object Detection:**
```
Output 0: boxes  - [1, num_queries, 4]  (normalized coordinates)
Output 1: logits - [1, num_queries, num_classes] (class scores)
```

Unterstützte Box-Formate:
- `CXCYWH`: Center X, Center Y, Width, Height (RF-DETR Standard)
- `XYXY`: x1, y1, x2, y2 (Ecken)
- `XYWH`: x, y, Width, Height (Top-Left + Größe)

**Output - Semantic Segmentation:**
```
Output 0: mask - [1, H, W] oder [1, H, W, num_classes]
```

Bei Multi-Channel Output wird Argmax automatisch angewendet.

### 3. Klassen-Labels konfigurieren

In `MainActivity.kt`:

```kotlin
private val classLabels = listOf(
    "Background",
    "Defect",
    "Person",
    // ... weitere Labels
)
```

### 4. Projekt öffnen

1. Android Studio öffnen
2. "Open an existing Android Studio project"
3. Ordner `android-segmentation-app` auswählen
4. Gradle Sync abwarten

### 5. App starten

1. Android-Gerät verbinden (USB-Debugging aktiviert)
2. Run-Button klicken oder `Shift+F10`
3. Kamera-Berechtigung erteilen

## Technische Details

### Auto-Detection Logik

Das Modell wird automatisch klassifiziert:

| Kriterium | Erkannter Typ |
|-----------|---------------|
| 2+ Outputs mit Shape `[1, N, 4]` und `[1, N, C]` | Object Detection |
| 1 Output mit Shape `[1, H, W]` oder `[1, H, W, C]` | Segmentation |

### Threading-Modell

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Camera Thread  │ ──► │ Inference Thread │ ──► │  Render Thread  │
│  (CameraX)      │     │  (ExecutorService)│     │  (SurfaceView)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

- **Camera Thread**: Wird NIEMALS blockiert
- **Inference Thread**: Dedizierter Background-Thread für ML
- **Render Thread**: SurfaceView Rendering unabhängig vom UI-Thread

### Hardware-Delegates

| Priorität | Delegate | Beschreibung |
|-----------|----------|--------------|
| 1 | GPU | Beste Performance auf den meisten Geräten |
| 2 | NNAPI | Neural Network API (Android 8.1+) |
| 3 | CPU | Immer verfügbar, aber langsamer |

## Konfiguration

### TFLiteModel

```kotlin
model = TFLiteModel(this).apply {
    // Klassen-Labels
    classLabels = listOf("Background", "Cat", "Dog")

    // Detection-spezifisch
    confidenceThreshold = 0.5f  // Mindest-Konfidenz
    nmsThreshold = 0.4f         // Non-Maximum Suppression IoU
    boxFormat = TFLiteModel.BoxFormat.CXCYWH
}
```

### OverlayRenderer

```kotlin
renderer = OverlayRenderer(binding.overlayView).apply {
    // Segmentation
    maskAlpha = 0.4f           // Masken-Transparenz (0-1)

    // Detection
    boxStrokeWidth = 6f        // Box-Liniendicke
    labelTextSize = 40f        // Label-Textgröße
    showLabels = true          // Labels anzeigen
    showConfidence = true      // Konfidenz anzeigen
}
```

## Farbtabelle

| Klasse | Farbe | RGB |
|--------|-------|-----|
| 0 | Transparent | (Background) |
| 1 | Rot | (255, 0, 0) |
| 2 | Grün | (0, 255, 0) |
| 3 | Blau | (0, 0, 255) |
| 4 | Gelb | (255, 255, 0) |
| 5 | Magenta | (255, 0, 255) |
| 6+ | Auto-generiert | (HSV-basiert) |

## Model Export (PyTorch → TFLite)

Für RF-DETR Modelle:

```bash
cd model-export

# Abhängigkeiten installieren
pip install torch torchvision onnx onnx-tf tensorflow rfdetr

# Export ausführen
python export_to_tflite.py
```

Das Skript führt folgende Schritte aus:
1. PyTorch → ONNX
2. ONNX → TensorFlow SavedModel
3. TensorFlow SavedModel → TFLite

## Anforderungen

- **minSdk**: 26 (Android 8.0)
- **targetSdk**: 34 (Android 14)
- **Kotlin**: 1.9.22
- **CameraX**: 1.3.1
- **TensorFlow Lite**: 2.13.0

## Troubleshooting

**"Failed to load model":**
- Prüfen ob `.tflite` Datei in `assets/` liegt
- Dateiname muss mit `MODEL_FILE` in MainActivity übereinstimmen

**Falsche Detektionen/Masken:**
- Box-Format prüfen (`CXCYWH` vs `XYXY`)
- Confidence Threshold anpassen
- Input-Normalisierung prüfen (muss [0,1] sein)

**Niedrige FPS:**
- Frame-Skip auf N=3 erhöhen
- GPU Delegate prüfen
- Model quantisieren (INT8)

**Boxes an falscher Position:**
- Koordinaten-System prüfen (normalisiert vs pixel)
- Rotation-Handling kontrollieren

## Lizenz

MIT License
