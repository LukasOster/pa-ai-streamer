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
        private const val MODEL_FILE = "rf_detr_segmentation.tflite"
    }

    // View binding
    private lateinit var binding: ActivityMainBinding

    // Core components
    private lateinit var cameraManager: CameraManager
    private lateinit var model: TFLiteModel
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

    // Class labels (configure based on your model)
    private val classLabels = listOf(
        "Background",
        "Defect",
        // Add more labels as needed
    )

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
        super.onCreate(savedInstanceState)

        setupFullscreen()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setupControls()
        checkCameraPermission()
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
        if (!modelReady.get()) return
        if (currentDelegate == delegate) return

        modelReady.set(false)
        updateStatusDisplay("Switching...")

        mainScope.launch {
            try {
                val actualDelegate = withContext(Dispatchers.Default) {
                    model.initialize(MODEL_FILE, delegate)
                }
                currentDelegate = actualDelegate
                modelReady.set(true)
                updateDelegateButtons()
                updateStatusDisplay("${model.resolvedModelType.name} | ${actualDelegate.name}")
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
        Log.i(TAG, "Initializing components")

        // Inference executor
        inferenceExecutor = Executors.newSingleThreadExecutor { r ->
            Thread(r, "InferenceThread").apply {
                priority = Thread.MAX_PRIORITY
            }
        }

        // Model
        model = TFLiteModel(this).apply {
            classLabels = this@MainActivity.classLabels
            confidenceThreshold = 0.1f  // Low threshold for debugging
            boxFormat = TFLiteModel.BoxFormat.CXCYWH  // RF-DETR uses center format
            useImageNetNormalization = true  // RF-DETR requires ImageNet normalization
        }

        // Renderer
        renderer = OverlayRenderer(binding.overlayView).apply {
            maskAlpha = 0.4f
            boxStrokeWidth = 6f
            labelTextSize = 40f
            showLabels = true
            showConfidence = true
        }

        // Camera
        cameraManager = CameraManager(this, this, binding.previewView).apply {
            setFrameCallback(this@MainActivity)
            setFrameSkip(currentFrameSkip)
            setFpsCallback { fps -> addCameraFps(fps) }
        }

        initializeModel()
    }

    private fun initializeModel() {
        updateStatusDisplay("Loading...")

        mainScope.launch {
            try {
                val actualDelegate = withContext(Dispatchers.Default) {
                    model.initialize(MODEL_FILE, currentDelegate)
                }

                currentDelegate = actualDelegate
                modelReady.set(true)
                updateDelegateButtons()

                val modelTypeStr = model.resolvedModelType.name
                updateStatusDisplay("$modelTypeStr | ${actualDelegate.name}")

                Log.i(TAG, "Model initialized: $modelTypeStr with $actualDelegate")
                Log.i(TAG, "Input size: ${model.inputWidth}x${model.inputHeight}")

                // Start camera
                cameraManager.start()

            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize model: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@MainActivity,
                        "Failed to load model. Ensure $MODEL_FILE is in assets.",
                        Toast.LENGTH_LONG
                    ).show()
                    updateStatusDisplay("Error: Model not found")
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

                // Run unified inference
                val result = model.runInference(bitmap)

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

                // Track inference FPS
                val inferenceTime = System.currentTimeMillis() - inferenceStart
                val inferenceFps = 1000f / inferenceTime.coerceAtLeast(1)
                addInferenceFps(inferenceFps)

                updateFpsDisplay()

            } catch (e: Exception) {
                Log.e(TAG, "Inference error: ${e.message}", e)
            } finally {
                cameraManager.onInferenceComplete()

                if (!bitmap.isRecycled) {
                    bitmap.recycle()
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

        if (::model.isInitialized) {
            model.close()
        }
    }
}
