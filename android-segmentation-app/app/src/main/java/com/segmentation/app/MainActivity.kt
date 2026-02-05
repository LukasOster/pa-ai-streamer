package com.segmentation.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.segmentation.app.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.LinkedList
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Main Activity for Real-time ML Inference App
 *
 * Supports both:
 * - Object Detection (bounding boxes)
 * - Semantic Segmentation (pixel masks)
 *
 * Model type is auto-detected or can be explicitly configured.
 *
 * ARCHITECTURE:
 * - CameraManager: CameraX preview and frame extraction
 * - TFLiteModel: Unified TensorFlow Lite inference (detection + segmentation)
 * - OverlayRenderer: Draws boxes or masks on SurfaceView
 *
 * THREADING:
 * - UI Thread: Updates UI, handles input
 * - Camera Analysis Thread: Extracts frames (never blocked)
 * - Inference Thread: Runs TFLite model
 * - Render Thread: Draws overlay (SurfaceView)
 */
class MainActivity : AppCompatActivity(), CameraManager.FrameCallback {

    companion object {
        private const val TAG = "MainActivity"
        private const val FPS_WINDOW_SIZE = 30

        // Model configuration
        private const val TFLITE_MODEL_FILE = "rf_detr_segmentation.tflite"
        private const val ONNX_MODEL_FILE = "rf_detr_detection.onnx"

        // Use ONNX Runtime (has working box outputs)
        private const val USE_ONNX = true
    }

    // View binding
    private lateinit var binding: ActivityMainBinding

    // Core components
    private lateinit var cameraManager: CameraManager
    private lateinit var tfliteModel: TFLiteModel
    private var onnxModel: ONNXModel? = null
    private lateinit var renderer: OverlayRenderer

    // Inference executor
    private lateinit var inferenceExecutor: ExecutorService

    // Coroutine scope
    private val mainScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // State
    private val modelReady = AtomicBoolean(false)

    // FPS tracking
    private val cameraFpsWindow = LinkedList<Float>()
    private val inferenceFpsWindow = LinkedList<Float>()

    // Current settings
    private var currentFrameSkip = 2
    private var currentDelegate = TFLiteModel.DelegateType.GPU

    // COCO class labels using ORIGINAL COCO IDs (with gaps)
    // RF-DETR outputs these original IDs, not continuous 0-90
    private val cocoClassMap = mapOf(
        0 to "Background",
        1 to "Person", 2 to "Bicycle", 3 to "Car", 4 to "Motorcycle", 5 to "Airplane",
        6 to "Bus", 7 to "Train", 8 to "Truck", 9 to "Boat", 10 to "Traffic Light",
        11 to "Fire Hydrant", 13 to "Stop Sign", 14 to "Parking Meter", 15 to "Bench",
        16 to "Bird", 17 to "Cat", 18 to "Dog", 19 to "Horse", 20 to "Sheep",
        21 to "Cow", 22 to "Elephant", 23 to "Bear", 24 to "Zebra", 25 to "Giraffe",
        27 to "Backpack", 28 to "Umbrella", 31 to "Handbag", 32 to "Tie", 33 to "Suitcase",
        34 to "Frisbee", 35 to "Skis", 36 to "Snowboard", 37 to "Sports Ball", 38 to "Kite",
        39 to "Baseball Bat", 40 to "Baseball Glove", 41 to "Skateboard", 42 to "Surfboard",
        43 to "Tennis Racket", 44 to "Bottle", 46 to "Wine Glass", 47 to "Cup", 48 to "Fork",
        49 to "Knife", 50 to "Spoon", 51 to "Bowl", 52 to "Banana", 53 to "Apple",
        54 to "Sandwich", 55 to "Orange", 56 to "Broccoli", 57 to "Carrot", 58 to "Hot Dog",
        59 to "Pizza", 60 to "Donut", 61 to "Cake", 62 to "Chair", 63 to "Couch",
        64 to "Potted Plant", 65 to "Bed", 67 to "Dining Table", 70 to "Toilet", 72 to "TV",
        73 to "Laptop", 74 to "Mouse", 75 to "Remote", 76 to "Keyboard", 77 to "Cell Phone",
        78 to "Microwave", 79 to "Oven", 80 to "Toaster", 81 to "Sink", 82 to "Refrigerator",
        84 to "Book", 85 to "Clock", 86 to "Vase", 87 to "Scissors", 88 to "Teddy Bear",
        89 to "Hair Drier", 90 to "Toothbrush"
    )

    // Convert map to list for ONNXModel (it uses list index for label lookup)
    private val classLabels = (0..90).map { cocoClassMap[it] ?: "Class $it" }

    // Permission launcher
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            initializeComponents()
        } else {
            Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "=== onCreate START ===")
        super.onCreate(savedInstanceState)

        try {
            Log.d(TAG, "Setting up fullscreen...")
            setupFullscreen()

            Log.d(TAG, "Inflating layout...")
            binding = ActivityMainBinding.inflate(layoutInflater)
            setContentView(binding.root)

            Log.d(TAG, "Setting window flags...")
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

            Log.d(TAG, "Setting up controls...")
            setupControls()

            Log.d(TAG, "Checking camera permission...")
            checkCameraPermission()

            Log.i(TAG, "=== onCreate COMPLETE ===")
        } catch (e: Exception) {
            Log.e(TAG, "=== onCreate FAILED ===")
            Log.e(TAG, "Error: ${e.message}", e)
            throw e
        }
    }

    private fun setupFullscreen() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).let { controller ->
            controller.hide(WindowInsetsCompat.Type.systemBars())
            controller.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    private fun setupControls() {
        // Frame skip buttons
        binding.btnFrameN1.setOnClickListener { setFrameSkip(1) }
        binding.btnFrameN2.setOnClickListener { setFrameSkip(2) }
        binding.btnFrameN3.setOnClickListener { setFrameSkip(3) }

        // Delegate buttons
        binding.btnGpu.setOnClickListener { switchDelegate(TFLiteModel.DelegateType.GPU) }
        binding.btnNnapi.setOnClickListener { switchDelegate(TFLiteModel.DelegateType.NNAPI) }
        binding.btnCpu.setOnClickListener { switchDelegate(TFLiteModel.DelegateType.CPU) }

        updateFrameSkipButtons()
        updateDelegateButtons()
    }

    private fun setFrameSkip(n: Int) {
        currentFrameSkip = n
        if (::cameraManager.isInitialized) {
            cameraManager.setFrameSkip(n)
        }
        updateFrameSkipButtons()
        Log.i(TAG, "Frame skip set to $n")
    }

    private fun updateFrameSkipButtons() {
        binding.btnFrameN1.alpha = if (currentFrameSkip == 1) 1.0f else 0.5f
        binding.btnFrameN2.alpha = if (currentFrameSkip == 2) 1.0f else 0.5f
        binding.btnFrameN3.alpha = if (currentFrameSkip == 3) 1.0f else 0.5f
    }

    private fun switchDelegate(delegate: TFLiteModel.DelegateType) {
        // Delegate switching only applies to TFLite mode
        if (USE_ONNX) {
            Toast.makeText(this, "Delegate switching not available in ONNX mode", Toast.LENGTH_SHORT).show()
            return
        }

        if (!modelReady.get()) return
        if (currentDelegate == delegate) return

        modelReady.set(false)
        updateStatusDisplay("Switching...")

        mainScope.launch {
            try {
                val actualDelegate = withContext(Dispatchers.Default) {
                    tfliteModel.initialize(TFLITE_MODEL_FILE, delegate)
                }
                currentDelegate = actualDelegate
                modelReady.set(true)
                updateDelegateButtons()
                updateStatusDisplay("${tfliteModel.resolvedModelType.name} | ${actualDelegate.name}")
                Log.i(TAG, "Switched to $actualDelegate delegate")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to switch delegate: ${e.message}", e)
                Toast.makeText(this@MainActivity, "Failed to switch delegate", Toast.LENGTH_SHORT).show()
                updateStatusDisplay("Error")
                modelReady.set(true)
            }
        }
    }

    private fun updateDelegateButtons() {
        binding.btnGpu.alpha = if (currentDelegate == TFLiteModel.DelegateType.GPU) 1.0f else 0.5f
        binding.btnNnapi.alpha = if (currentDelegate == TFLiteModel.DelegateType.NNAPI) 1.0f else 0.5f
        binding.btnCpu.alpha = if (currentDelegate == TFLiteModel.DelegateType.CPU) 1.0f else 0.5f
    }

    private fun updateStatusDisplay(text: String) {
        binding.delegateTextView.text = text
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                initializeComponents()
            }
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show()
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
            else -> {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun initializeComponents() {
        Log.i(TAG, "=== initializeComponents START ===")
        Log.i(TAG, "USE_ONNX = $USE_ONNX")

        try {
            // Inference executor
            Log.d(TAG, "Creating inference executor...")
            inferenceExecutor = Executors.newSingleThreadExecutor { r ->
                Thread(r, "InferenceThread").apply {
                    priority = Thread.MAX_PRIORITY
                }
            }
            Log.d(TAG, "Inference executor created")

            // Model - use ONNX or TFLite based on configuration
            Log.d(TAG, "Creating model wrapper...")
            if (USE_ONNX) {
                Log.d(TAG, "Creating ONNXModel...")
                onnxModel = ONNXModel(this).apply {
                    classLabels = this@MainActivity.classLabels
                    confidenceThreshold = 0.1f
                }
                Log.d(TAG, "ONNXModel created")
            } else {
                Log.d(TAG, "Creating TFLiteModel...")
                tfliteModel = TFLiteModel(this).apply {
                    classLabels = this@MainActivity.classLabels
                    confidenceThreshold = 0.1f
                    boxFormat = TFLiteModel.BoxFormat.CXCYWH
                    useImageNetNormalization = true
                }
                Log.d(TAG, "TFLiteModel created")
            }

            // Renderer
            Log.d(TAG, "Creating renderer...")
            renderer = OverlayRenderer(binding.overlayView).apply {
                maskAlpha = 0.4f
                boxStrokeWidth = 6f
                labelTextSize = 40f
                showLabels = true
                showConfidence = true
            }
            Log.d(TAG, "Renderer created")

            // Camera
            Log.d(TAG, "Creating camera manager...")
            cameraManager = CameraManager(this, this, binding.previewView).apply {
                setFrameCallback(this@MainActivity)
                setFrameSkip(currentFrameSkip)
                setFpsCallback { fps -> addCameraFps(fps) }
            }
            Log.d(TAG, "Camera manager created")

            Log.i(TAG, "=== initializeComponents COMPLETE ===")
            initializeModel()

        } catch (e: Exception) {
            Log.e(TAG, "=== initializeComponents FAILED ===")
            Log.e(TAG, "Error: ${e.message}", e)
            throw e
        }
    }

    private fun initializeModel() {
        Log.i(TAG, "=== initializeModel START ===")
        updateStatusDisplay("Loading...")

        mainScope.launch {
            try {
                if (USE_ONNX) {
                    Log.d(TAG, "Initializing ONNX model: $ONNX_MODEL_FILE")
                    withContext(Dispatchers.Default) {
                        onnxModel?.initialize(ONNX_MODEL_FILE)
                    }
                    Log.d(TAG, "ONNX model initialized, setting modelReady=true")
                    modelReady.set(true)
                    updateStatusDisplay("ONNX | DETECTION")
                    Log.i(TAG, "ONNX model ready")
                    Log.i(TAG, "Input size: ${onnxModel?.inputWidth}x${onnxModel?.inputHeight}")
                } else {
                    Log.d(TAG, "Initializing TFLite model: $TFLITE_MODEL_FILE")
                    val actualDelegate = withContext(Dispatchers.Default) {
                        tfliteModel.initialize(TFLITE_MODEL_FILE, currentDelegate)
                    }
                    currentDelegate = actualDelegate
                    modelReady.set(true)
                    updateDelegateButtons()
                    val modelTypeStr = tfliteModel.resolvedModelType.name
                    updateStatusDisplay("$modelTypeStr | ${actualDelegate.name}")
                    Log.i(TAG, "TFLite model initialized: $modelTypeStr with $actualDelegate")
                    Log.i(TAG, "Input size: ${tfliteModel.inputWidth}x${tfliteModel.inputHeight}")
                }

                // Start camera
                Log.d(TAG, "Starting camera...")
                cameraManager.start()
                Log.i(TAG, "=== initializeModel COMPLETE ===")

            } catch (e: Exception) {
                Log.e(TAG, "=== initializeModel FAILED ===")
                Log.e(TAG, "Error: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    val modelFile = if (USE_ONNX) ONNX_MODEL_FILE else TFLITE_MODEL_FILE
                    Toast.makeText(
                        this@MainActivity,
                        "Failed to load model: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    updateStatusDisplay("Error: ${e.message}")
                }
            }
        }
    }

    /**
     * CameraManager.FrameCallback - called on camera analysis thread.
     */
    override fun onFrameAvailable(bitmap: Bitmap, rotationDegrees: Int) {
        if (!modelReady.get()) {
            cameraManager.onInferenceComplete()
            return
        }

        inferenceExecutor.execute {
            try {
                val inferenceStart = System.currentTimeMillis()

                if (USE_ONNX) {
                    // Run ONNX inference
                    val result = onnxModel?.runInference(bitmap)
                    if (result != null) {
                        renderer.updateDetections(result.detections)
                        if (result.detections.isNotEmpty()) {
                            Log.d(TAG, "ONNX detected ${result.detections.size} objects")
                            result.detections.forEach { det ->
                                Log.d(TAG, "  ${det.label}: ${det.confidence} @ ${det.boundingBox}")
                            }
                        }
                    }
                } else {
                    // Run TFLite inference
                    val result = tfliteModel.runInference(bitmap)

                    when (result) {
                        is TFLiteModel.InferenceResult.Detections -> {
                            renderer.updateDetections(result.result.detections)

                            // Log detection count periodically
                            if (result.result.detections.isNotEmpty()) {
                                Log.v(TAG, "Detected ${result.result.detections.size} objects")
                            }
                        }
                        is TFLiteModel.InferenceResult.Segmentation -> {
                            renderer.updateMask(result.result)
                        }
                        null -> {
                            Log.w(TAG, "Inference returned null")
                        }
                    }
                }

                // Track inference FPS
                val inferenceTime = System.currentTimeMillis() - inferenceStart
                val inferenceFps = 1000f / inferenceTime.coerceAtLeast(1)
                addInferenceFps(inferenceFps)

                updateFpsDisplay()

            } catch (e: Exception) {
                Log.e(TAG, "Inference error: ${e.message}", e)
                e.printStackTrace()
            } catch (t: Throwable) {
                Log.e(TAG, "Inference throwable: ${t.message}", t)
                t.printStackTrace()
            } finally {
                cameraManager.onInferenceComplete()

                try {
                    if (!bitmap.isRecycled) {
                        bitmap.recycle()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error recycling bitmap: ${e.message}")
                }
            }
        }
    }

    override fun onFrameSkipped() {
        // Previous result continues to be displayed
    }

    @Synchronized
    private fun addCameraFps(fps: Float) {
        cameraFpsWindow.addLast(fps)
        if (cameraFpsWindow.size > FPS_WINDOW_SIZE) {
            cameraFpsWindow.removeFirst()
        }
    }

    @Synchronized
    private fun addInferenceFps(fps: Float) {
        inferenceFpsWindow.addLast(fps)
        if (inferenceFpsWindow.size > FPS_WINDOW_SIZE) {
            inferenceFpsWindow.removeFirst()
        }
    }

    @Synchronized
    private fun getAverageFps(): Pair<Float, Float> {
        val cameraAvg = if (cameraFpsWindow.isNotEmpty()) {
            cameraFpsWindow.average().toFloat()
        } else 0f

        val inferenceAvg = if (inferenceFpsWindow.isNotEmpty()) {
            inferenceFpsWindow.average().toFloat()
        } else 0f

        return Pair(cameraAvg, inferenceAvg)
    }

    private fun updateFpsDisplay() {
        val (cameraFps, inferenceFps) = getAverageFps()

        runOnUiThread {
            binding.fpsTextView.text = String.format(
                "FPS: %.1f | Inf: %.1f",
                cameraFps,
                inferenceFps
            )
        }
    }

    override fun onResume() {
        super.onResume()
        if (::cameraManager.isInitialized && modelReady.get() && !cameraManager.isRunning()) {
            cameraManager.start()
        }
    }

    override fun onPause() {
        super.onPause()
        if (::cameraManager.isInitialized) {
            cameraManager.stop()
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        mainScope.cancel()

        if (::cameraManager.isInitialized) {
            cameraManager.shutdown()
        }

        if (::inferenceExecutor.isInitialized) {
            inferenceExecutor.shutdown()
        }

        if (::renderer.isInitialized) {
            renderer.release()
        }

        onnxModel?.close()

        if (::tfliteModel.isInitialized) {
            tfliteModel.close()
        }
    }
}
