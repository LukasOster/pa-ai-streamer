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
 * Main Activity for Real-time Segmentation App
 *
 * ARCHITECTURE OVERVIEW:
 * - CameraManager: Handles CameraX preview and frame extraction
 * - TFLiteSegmentationModel: Runs TensorFlow Lite inference
 * - SegmentationRenderer: Draws mask overlay on SurfaceView
 *
 * THREADING MODEL:
 * - UI Thread: Updates UI elements, handles user input
 * - Camera Analysis Thread: Extracts frames from camera (never blocked)
 * - Inference Thread: Runs TFLite model (separate from camera thread)
 * - Render Thread: Draws mask overlay (via SurfaceView)
 *
 * FRAME FLOW:
 * 1. Camera captures frame → Analysis thread
 * 2. Every Nth frame sent to inference thread
 * 3. Inference produces mask → Render thread
 * 4. Previous mask shown between inferences
 */
class MainActivity : AppCompatActivity(), CameraManager.FrameCallback {

    companion object {
        private const val TAG = "MainActivity"
        private const val FPS_WINDOW_SIZE = 30  // Rolling window for FPS average
    }

    // View binding
    private lateinit var binding: ActivityMainBinding

    // Core components
    private lateinit var cameraManager: CameraManager
    private lateinit var segmentationModel: TFLiteSegmentationModel
    private lateinit var renderer: SegmentationRenderer

    // Inference executor - dedicated thread for ML inference
    private lateinit var inferenceExecutor: ExecutorService

    // Coroutine scope for async operations
    private val mainScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // Model initialization state
    private val modelReady = AtomicBoolean(false)

    // FPS tracking with rolling average
    private val cameraFpsWindow = LinkedList<Float>()
    private val inferenceFpsWindow = LinkedList<Float>()
    private var lastInferenceTime = 0L

    // Current settings
    private var currentFrameSkip = 2
    private var currentDelegate = TFLiteSegmentationModel.DelegateType.GPU

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

        // Setup edge-to-edge display
        setupFullscreen()

        // Initialize view binding
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Keep screen on during segmentation
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Setup UI controls
        setupControls()

        // Check and request camera permission
        checkCameraPermission()
    }

    /**
     * Setup fullscreen immersive mode.
     */
    private fun setupFullscreen() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).let { controller ->
            controller.hide(WindowInsetsCompat.Type.systemBars())
            controller.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    /**
     * Setup UI control buttons.
     */
    private fun setupControls() {
        // Frame skip buttons
        binding.btnFrameN1.setOnClickListener { setFrameSkip(1) }
        binding.btnFrameN2.setOnClickListener { setFrameSkip(2) }
        binding.btnFrameN3.setOnClickListener { setFrameSkip(3) }

        // Delegate buttons
        binding.btnGpu.setOnClickListener { switchDelegate(TFLiteSegmentationModel.DelegateType.GPU) }
        binding.btnNnapi.setOnClickListener { switchDelegate(TFLiteSegmentationModel.DelegateType.NNAPI) }
        binding.btnCpu.setOnClickListener { switchDelegate(TFLiteSegmentationModel.DelegateType.CPU) }

        // Set initial button states
        updateFrameSkipButtons()
        updateDelegateButtons()
    }

    /**
     * Set frame skip value and update UI.
     */
    private fun setFrameSkip(n: Int) {
        currentFrameSkip = n
        if (::cameraManager.isInitialized) {
            cameraManager.setFrameSkip(n)
        }
        updateFrameSkipButtons()
        Log.i(TAG, "Frame skip set to $n")
    }

    /**
     * Update frame skip button visual states.
     */
    private fun updateFrameSkipButtons() {
        binding.btnFrameN1.alpha = if (currentFrameSkip == 1) 1.0f else 0.5f
        binding.btnFrameN2.alpha = if (currentFrameSkip == 2) 1.0f else 0.5f
        binding.btnFrameN3.alpha = if (currentFrameSkip == 3) 1.0f else 0.5f
    }

    /**
     * Switch to a different TFLite delegate.
     */
    private fun switchDelegate(delegate: TFLiteSegmentationModel.DelegateType) {
        if (!modelReady.get()) return
        if (currentDelegate == delegate) return

        modelReady.set(false)
        updateDelegateDisplay("Switching...")

        mainScope.launch {
            try {
                val actualDelegate = withContext(Dispatchers.Default) {
                    segmentationModel.initialize(delegate)
                }
                currentDelegate = actualDelegate
                modelReady.set(true)
                updateDelegateButtons()
                updateDelegateDisplay(actualDelegate.name)
                Log.i(TAG, "Switched to $actualDelegate delegate")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to switch delegate: ${e.message}", e)
                Toast.makeText(this@MainActivity, "Failed to switch delegate", Toast.LENGTH_SHORT).show()
                updateDelegateDisplay(currentDelegate.name)
                modelReady.set(true)
            }
        }
    }

    /**
     * Update delegate button visual states.
     */
    private fun updateDelegateButtons() {
        binding.btnGpu.alpha = if (currentDelegate == TFLiteSegmentationModel.DelegateType.GPU) 1.0f else 0.5f
        binding.btnNnapi.alpha = if (currentDelegate == TFLiteSegmentationModel.DelegateType.NNAPI) 1.0f else 0.5f
        binding.btnCpu.alpha = if (currentDelegate == TFLiteSegmentationModel.DelegateType.CPU) 1.0f else 0.5f
    }

    /**
     * Update delegate display text.
     */
    private fun updateDelegateDisplay(text: String) {
        binding.delegateTextView.text = text
    }

    /**
     * Check camera permission and request if needed.
     */
    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
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

    /**
     * Initialize all components after permission is granted.
     */
    private fun initializeComponents() {
        Log.i(TAG, "Initializing components")

        // Initialize inference executor
        inferenceExecutor = Executors.newSingleThreadExecutor { r ->
            Thread(r, "InferenceThread").apply {
                priority = Thread.MAX_PRIORITY
            }
        }

        // Initialize segmentation model
        segmentationModel = TFLiteSegmentationModel(this)

        // Initialize renderer
        renderer = SegmentationRenderer(binding.overlayView)
        renderer.maskAlpha = 0.4f

        // Initialize camera manager
        cameraManager = CameraManager(this, this, binding.previewView)
        cameraManager.setFrameCallback(this)
        cameraManager.setFrameSkip(currentFrameSkip)

        // Setup FPS callback
        cameraManager.setFpsCallback { fps ->
            addCameraFps(fps)
        }

        // Initialize model async and start camera when ready
        initializeModel()
    }

    /**
     * Initialize TFLite model on background thread.
     */
    private fun initializeModel() {
        updateDelegateDisplay("Loading...")

        mainScope.launch {
            try {
                val actualDelegate = withContext(Dispatchers.Default) {
                    segmentationModel.initialize(currentDelegate)
                }

                currentDelegate = actualDelegate
                modelReady.set(true)
                updateDelegateButtons()
                updateDelegateDisplay(actualDelegate.name)

                Log.i(TAG, "Model initialized with ${actualDelegate.name} delegate")
                Log.i(TAG, "Model input size: ${segmentationModel.inputWidth}x${segmentationModel.inputHeight}")

                // Start camera after model is ready
                cameraManager.start()

            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize model: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@MainActivity,
                        "Failed to load model. Please ensure rf_detr_segmentation.tflite is in assets.",
                        Toast.LENGTH_LONG
                    ).show()
                    updateDelegateDisplay("Error")
                }
            }
        }
    }

    /**
     * CameraManager.FrameCallback - called when frame is ready for analysis.
     * This runs on the camera analysis thread.
     */
    override fun onFrameAvailable(bitmap: Bitmap, rotationDegrees: Int) {
        if (!modelReady.get()) {
            cameraManager.onInferenceComplete()
            return
        }

        // Submit inference to dedicated executor
        inferenceExecutor.execute {
            try {
                val inferenceStart = System.currentTimeMillis()

                // Run segmentation
                val result = segmentationModel.runInference(bitmap)

                if (result != null) {
                    // Update renderer with new mask
                    renderer.updateMask(result)

                    // Track inference FPS
                    val inferenceTime = System.currentTimeMillis() - inferenceStart
                    val inferenceFps = 1000f / inferenceTime.coerceAtLeast(1)
                    addInferenceFps(inferenceFps)

                    // Update FPS display
                    updateFpsDisplay()
                }

            } catch (e: Exception) {
                Log.e(TAG, "Inference error: ${e.message}", e)
            } finally {
                // Always signal completion to allow next frame
                cameraManager.onInferenceComplete()

                // Recycle the bitmap after inference
                if (!bitmap.isRecycled) {
                    bitmap.recycle()
                }
            }
        }
    }

    /**
     * CameraManager.FrameCallback - called when frame is skipped.
     */
    override fun onFrameSkipped() {
        // Frame was skipped due to frame skip setting
        // Previous mask continues to be displayed
    }

    /**
     * Add camera FPS sample to rolling window.
     */
    @Synchronized
    private fun addCameraFps(fps: Float) {
        cameraFpsWindow.addLast(fps)
        if (cameraFpsWindow.size > FPS_WINDOW_SIZE) {
            cameraFpsWindow.removeFirst()
        }
    }

    /**
     * Add inference FPS sample to rolling window.
     */
    @Synchronized
    private fun addInferenceFps(fps: Float) {
        inferenceFpsWindow.addLast(fps)
        if (inferenceFpsWindow.size > FPS_WINDOW_SIZE) {
            inferenceFpsWindow.removeFirst()
        }
    }

    /**
     * Calculate rolling average FPS.
     */
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

    /**
     * Update FPS display on UI thread.
     */
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
        // Restart camera if it was stopped
        if (::cameraManager.isInitialized && modelReady.get() && !cameraManager.isRunning()) {
            cameraManager.start()
        }
    }

    override fun onPause() {
        super.onPause()
        // Stop camera to save battery
        if (::cameraManager.isInitialized) {
            cameraManager.stop()
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        // Cancel coroutines
        mainScope.cancel()

        // Shutdown components
        if (::cameraManager.isInitialized) {
            cameraManager.shutdown()
        }

        if (::inferenceExecutor.isInitialized) {
            inferenceExecutor.shutdown()
        }

        if (::renderer.isInitialized) {
            renderer.release()
        }

        if (::segmentationModel.isInitialized) {
            segmentationModel.close()
        }
    }
}
