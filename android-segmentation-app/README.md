# Real-Time Segmentation Android App

Android-App für Echtzeit-Bildsegmentierung mit TensorFlow Lite auf dem Live-Kamera-Stream.

## Features

- Live-Kamera-Preview mit CameraX
- TensorFlow Lite Inference mit Hardware-Beschleunigung (GPU/NNAPI/CPU)
- Farbiges, halbtransparentes Segmentierungs-Overlay
- Frame-Skipping für optimale Performance (N=1/2/3)
- Rolling-Average FPS-Counter
- Umschaltbare Delegates zur Laufzeit

## Projektstruktur

```
app/src/main/java/com/segmentation/app/
├── MainActivity.kt           # Haupt-Activity, koordiniert alle Komponenten
├── CameraManager.kt          # CameraX Setup, Frame-Extraktion, Threading
├── TFLiteSegmentationModel.kt # TensorFlow Lite Inference Wrapper
└── SegmentationRenderer.kt   # Overlay-Rendering auf SurfaceView
```

## Setup

### 1. Model-Datei hinzufügen

Kopieren Sie Ihre TFLite-Modell-Datei in den Assets-Ordner:

```
app/src/main/assets/rf_detr_segmentation.tflite
```

### 2. Model-Anforderungen

**Input:**
- Shape: `[1, H, W, 3]` (NHWC Format)
- Typ: Float32
- Werte: RGB, normalisiert auf `[0, 1]`

**Output (unterstützte Varianten):**

*Variante A - Semantische Segmentierung:*
- Shape: `[1, H, W]` oder `[H, W]`
- Wert pro Pixel: Klassen-ID (Integer)

*Variante B - Multi-Channel (Softmax):*
- Shape: `[1, H, W, C]` wobei C = Anzahl Klassen
- Argmax wird automatisch angewendet

### 3. Projekt öffnen

1. Android Studio öffnen
2. "Open an existing Android Studio project"
3. Ordner `android-segmentation-app` auswählen
4. Gradle Sync abwarten

### 4. App starten

1. Android-Gerät verbinden (USB-Debugging aktiviert)
2. Run-Button klicken oder `Shift+F10`
3. Kamera-Berechtigung erteilen

## Technische Details

### Threading-Modell

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Camera Thread  │ ──► │ Inference Thread │ ──► │  Render Thread  │
│  (CameraX)      │     │  (ExecutorService)│     │  (SurfaceView)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
      │                                                   ▲
      │ Frame Skip (N)                                    │
      └───────────────────────────────────────────────────┘
                    Letzte Maske weiter anzeigen
```

- **Camera Thread**: Wird NIEMALS blockiert
- **Inference Thread**: Dedizierter Background-Thread für ML
- **Render Thread**: SurfaceView Rendering unabhängig vom UI-Thread

### Frame-Skipping

- N=1: Jedes Frame wird analysiert (höchste Last)
- N=2: Jedes zweite Frame (Standard, gute Balance)
- N=3: Jedes dritte Frame (niedrigste Last)

Zwischen den Inferenz-Frames wird die letzte Maske weiter angezeigt.

### Hardware-Delegates

**Priorität:**
1. **GPU Delegate** (Primär): Beste Performance auf den meisten Geräten
2. **NNAPI Delegate** (Fallback): Nutzt Neural Network API
3. **CPU** (Letzter Fallback): Immer verfügbar

Automatisches Fallback wenn bevorzugter Delegate nicht unterstützt wird.

## Farbtabelle

Vordefinierte Farben für Segmentierungsklassen:

| Klasse | Farbe        | RGB           |
|--------|--------------|---------------|
| 0      | Transparent  | (Background)  |
| 1      | Rot          | (255, 0, 0)   |
| 2      | Grün         | (0, 255, 0)   |
| 3      | Blau         | (0, 0, 255)   |
| 4      | Gelb         | (255, 255, 0) |
| 5      | Magenta      | (255, 0, 255) |
| 6      | Cyan         | (0, 255, 255) |
| 7      | Orange       | (255, 128, 0) |
| 8      | Lila         | (128, 0, 255) |
| ...    | Auto-generiert | (HSV-basiert) |

Klassen > 20 werden automatisch mit eindeutigen Farben versehen.

## Anpassungen

### Eigene Klassenfarben

```kotlin
// In MainActivity nach Renderer-Initialisierung:
renderer.setClassColor(1, Color.rgb(255, 100, 50))  // Klasse 1 = Custom Orange
```

### Mask Alpha ändern

```kotlin
renderer.maskAlpha = 0.6f  // 60% Deckkraft
```

### Temporal Smoothing aktivieren

```kotlin
renderer.enableSmoothing = true
renderer.smoothingFactor = 0.7f  // Höher = schnellere Reaktion
```

## Anforderungen

- **minSdk**: 26 (Android 8.0)
- **targetSdk**: 34 (Android 14)
- **Kotlin**: 1.9.22
- **CameraX**: 1.3.1
- **TensorFlow Lite**: 2.14.0

## Bekannte Einschränkungen

1. Modell muss NHWC-Format verwenden (nicht NCHW)
2. Front-Kamera wird aktuell nicht unterstützt
3. Landschaft-Modus deaktiviert für einfacheres Handling

## Troubleshooting

**"Failed to load model":**
- Prüfen ob `rf_detr_segmentation.tflite` in `assets/` liegt
- Modell-Dateiname muss exakt stimmen

**Niedrige FPS:**
- Frame-Skip auf N=3 erhöhen
- GPU Delegate prüfen (sollte aktiv sein)
- Modell-Input-Auflösung reduzieren

**Overlay passt nicht zum Preview:**
- Model Output-Shape prüfen
- Rotation-Handling in CameraManager kontrollieren

## Lizenz

MIT License - Verwenden Sie den Code wie Sie möchten.
