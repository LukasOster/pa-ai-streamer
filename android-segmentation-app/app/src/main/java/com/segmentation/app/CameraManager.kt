package com.segmentation.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import android.util.Size
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * CameraX manager for live camera preview and frame analysis.
 *
 * DESIGN PRINCIPLES:
 * 1. Camera thread is NEVER blocked by inference
 * 2. Frame skipping (N-th frame analysis) is enforced
 * 3. Latest frame strategy ensures freshest data
 *
 * Uses STRATEGY_KEEP_ONLY_LATEST for ImageAnalysis to ensure we always
 * process the most recent frame and don't build up backlog.
 */
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView
) {
    companion object {
        private const val TAG = "CameraManager"

        // Target resolution for analysis (lower = faster inference)
        private const val TARGET_ANALYSIS_WIDTH = 640
        private const val TARGET_ANALYSIS_HEIGHT = 480
    }

    /**
     * Callback interface for frame analysis results
     */
    interface FrameCallback {
        /**
         * Called when a frame is ready for analysis.
         * IMPORTANT: This is called on the analysis executor thread.
         *
         * @param bitmap The frame as a Bitmap (RGB)
         * @param rotationDegrees Rotation needed to display correctly
         */
        fun onFrameAvailable(bitmap: Bitmap, rotationDegrees: Int)

        /**
         * Called when a frame is skipped due to frame skip setting.
         */
        fun onFrameSkipped()
    }

    // Camera provider
    private var cameraProvider: ProcessCameraProvider? = null

    // Analysis executor - dedicated thread for image analysis
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "CameraAnalysisThread").apply {
            priority = Thread.MAX_PRIORITY - 1
        }
    }

    // Frame callback
    private var frameCallback: FrameCallback? = null

    // Frame skip configuration
    private val frameSkip = AtomicInteger(2)  // Process every Nth frame (default: 2)
    private val frameCounter = AtomicInteger(0)

    // Inference running flag to prevent queuing multiple frames
    private val inferenceRunning = AtomicBoolean(false)

    // FPS tracking
    private val lastFrameTime = AtomicLong(0)
    private var fpsCallback: ((Float) -> Unit)? = null

    // Camera state
    private val isRunning = AtomicBoolean(false)

    // Reusable buffers for YUV to RGB conversion
    private var yuvBytes: ByteArray? = null
    private var rgbBytes: IntArray? = null

    /**
     * Set the frame skip value (N = process every Nth frame).
     * @param n Frame skip value (1 = every frame, 2 = every other frame, etc.)
     */
    fun setFrameSkip(n: Int) {
        frameSkip.set(n.coerceAtLeast(1))
        Log.d(TAG, "Frame skip set to $n")
    }

    /**
     * Get current frame skip value.
     */
    fun getFrameSkip(): Int = frameSkip.get()

    /**
     * Set callback for frame analysis.
     */
    fun setFrameCallback(callback: FrameCallback) {
        this.frameCallback = callback
    }

    /**
     * Set callback for FPS updates.
     */
    fun setFpsCallback(callback: (Float) -> Unit) {
        this.fpsCallback = callback
    }

    /**
     * Start camera preview and analysis.
     */
    fun start() {
        if (isRunning.getAndSet(true)) {
            Log.w(TAG, "Camera already running")
            return
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get camera provider: ${e.message}", e)
                isRunning.set(false)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Stop camera and release resources.
     */
    fun stop() {
        if (!isRunning.getAndSet(false)) return

        try {
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            Log.w(TAG, "Error unbinding camera: ${e.message}")
        }
    }

    /**
     * Shutdown and release all resources.
     */
    fun shutdown() {
        stop()
        analysisExecutor.shutdown()
    }

    /**
     * Bind camera use cases (Preview + ImageAnalysis).
     */
    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return

        // Unbind previous use cases
        provider.unbindAll()

        // Camera selector - prefer back camera
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        // Preview use case
        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        // Image analysis use case
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(TARGET_ANALYSIS_WIDTH, TARGET_ANALYSIS_HEIGHT))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also {
                it.setAnalyzer(analysisExecutor, FrameAnalyzer())
            }

        try {
            // Bind use cases to camera
            provider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )
            Log.i(TAG, "Camera use cases bound successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to bind camera use cases: ${e.message}", e)
        }
    }

    /**
     * Frame analyzer that implements frame skipping and non-blocking analysis.
     */
    private inner class FrameAnalyzer : ImageAnalysis.Analyzer {

        override fun analyze(image: ImageProxy) {
            try {
                // Update FPS tracking
                trackFps()

                val currentFrame = frameCounter.incrementAndGet()
                val skip = frameSkip.get()

                // Check if we should process this frame
                if (currentFrame % skip != 0) {
                    // Skip this frame
                    frameCallback?.onFrameSkipped()
                    return
                }

                // Check if previous inference is still running
                if (!inferenceRunning.compareAndSet(false, true)) {
                    // Previous inference still running, skip this frame
                    Log.v(TAG, "Skipping frame - inference still running")
                    return
                }

                // Convert ImageProxy to Bitmap
                val bitmap = imageProxyToBitmap(image)
                val rotationDegrees = image.imageInfo.rotationDegrees

                if (bitmap != null) {
                    // Apply rotation if needed
                    val rotatedBitmap = if (rotationDegrees != 0) {
                        rotateBitmap(bitmap, rotationDegrees)
                    } else {
                        bitmap
                    }

                    // Notify callback (still on analysis thread)
                    frameCallback?.onFrameAvailable(rotatedBitmap, rotationDegrees)

                    // Clean up if we created a rotated copy
                    if (rotatedBitmap !== bitmap) {
                        bitmap.recycle()
                    }
                } else {
                    inferenceRunning.set(false)
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error analyzing frame: ${e.message}", e)
                inferenceRunning.set(false)
            } finally {
                // Always close the image to allow next frame
                image.close()
            }
        }
    }

    /**
     * Track frames per second and notify callback.
     */
    private fun trackFps() {
        val currentTime = System.currentTimeMillis()
        val lastTime = lastFrameTime.getAndSet(currentTime)

        if (lastTime > 0) {
            val deltaMs = currentTime - lastTime
            if (deltaMs > 0) {
                val fps = 1000f / deltaMs
                fpsCallback?.invoke(fps)
            }
        }
    }

    /**
     * Convert CameraX ImageProxy (YUV_420_888) to RGB Bitmap.
     *
     * This is optimized for speed over quality - we use YuvImage + JPEG encoding
     * which is hardware-accelerated on most devices.
     */
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        return try {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            // Allocate NV21 buffer if needed
            val nv21Size = ySize + uSize + vSize
            val nv21 = yuvBytes?.takeIf { it.size >= nv21Size } ?: ByteArray(nv21Size).also { yuvBytes = it }

            // Copy Y plane
            yBuffer.get(nv21, 0, ySize)

            // Convert UV planes from YUV420 to NV21 format
            // In NV21, V comes before U (interleaved)
            val uvPixelStride = image.planes[1].pixelStride
            val uvRowStride = image.planes[1].rowStride

            if (uvPixelStride == 2) {
                // Planes are already interleaved (common case)
                // Just need to reorder: YUV420 has UVUV, NV21 needs VUVU
                vBuffer.get(nv21, ySize, vSize)
                uBuffer.get(nv21, ySize + 1, uSize)
            } else {
                // Planes are not interleaved, need to manually interleave
                var offset = ySize
                for (row in 0 until image.height / 2) {
                    for (col in 0 until image.width / 2) {
                        val uvIndex = row * uvRowStride + col * uvPixelStride
                        nv21[offset++] = vBuffer.get(uvIndex)
                        nv21[offset++] = uBuffer.get(uvIndex)
                    }
                }
            }

            // Reset buffer positions
            yBuffer.rewind()
            uBuffer.rewind()
            vBuffer.rewind()

            // Convert NV21 to JPEG then decode to Bitmap
            val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            val outputStream = ByteArrayOutputStream()

            // Compress to JPEG (quality 90 is good balance of speed/quality)
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, outputStream)

            val jpegBytes = outputStream.toByteArray()
            BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert ImageProxy to Bitmap: ${e.message}", e)
            null
        }
    }

    /**
     * Rotate bitmap by specified degrees.
     */
    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        val matrix = Matrix().apply {
            postRotate(degrees.toFloat())
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Signal that inference is complete, allowing next frame to be processed.
     */
    fun onInferenceComplete() {
        inferenceRunning.set(false)
    }

    /**
     * Check if camera is currently running.
     */
    fun isRunning(): Boolean = isRunning.get()
}
