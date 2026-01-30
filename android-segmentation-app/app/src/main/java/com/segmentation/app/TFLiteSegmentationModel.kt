package com.segmentation.app

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * TensorFlow Lite Segmentation Model wrapper.
 * Handles model loading, delegate configuration, and inference.
 *
 * ASSUMPTION: Model file is named "rf_detr_segmentation.tflite" and placed in assets folder.
 *
 * MODEL INPUT:
 *   - Shape: [1, H, W, 3] where H and W are determined at runtime from model metadata
 *   - Format: RGB, float32, normalized to [0, 1]
 *
 * MODEL OUTPUT (supports two variants):
 *   Variant A - Instance Segmentation:
 *     - Output 0: Class mask [1, H, W] with class IDs (int or float)
 *     - Output 1 (optional): Instance masks [N, H, W] or confidence scores
 *
 *   Variant B - Semantic Segmentation:
 *     - Single output: [1, H, W] or [H, W] with class IDs per pixel
 *
 * The code auto-detects the output format based on tensor shapes.
 */
class TFLiteSegmentationModel(private val context: Context) {

    companion object {
        private const val TAG = "TFLiteSegModel"
        private const val MODEL_FILENAME = "rf_detr_segmentation.tflite"

        // Default model dimensions (will be overwritten by actual model metadata)
        private const val DEFAULT_INPUT_HEIGHT = 512
        private const val DEFAULT_INPUT_WIDTH = 512
    }

    /**
     * Delegate types for hardware acceleration
     */
    enum class DelegateType {
        GPU,
        NNAPI,
        CPU
    }

    /**
     * Result class holding segmentation output
     */
    data class SegmentationResult(
        val classMask: Array<IntArray>,  // [H][W] array of class IDs
        val width: Int,
        val height: Int,
        val inferenceTimeMs: Long
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as SegmentationResult
            return classMask.contentDeepEquals(other.classMask) &&
                   width == other.width && height == other.height
        }

        override fun hashCode(): Int {
            var result = classMask.contentDeepHashCode()
            result = 31 * result + width
            result = 31 * result + height
            return result
        }
    }

    // Interpreter and delegate references
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null

    // Model dimensions (loaded from model)
    var inputWidth: Int = DEFAULT_INPUT_WIDTH
        private set
    var inputHeight: Int = DEFAULT_INPUT_HEIGHT
        private set

    // Current delegate type
    var currentDelegate: DelegateType = DelegateType.GPU
        private set

    // Thread safety lock
    private val lock = ReentrantLock()

    // Pre-allocated buffers for inference
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null
    private var outputArray: Array<Array<FloatArray>>? = null

    // Reusable pixel array for bitmap processing
    private var pixelBuffer: IntArray? = null

    /**
     * Initialize the model with the specified delegate.
     * Falls back to lower-tier delegates if preferred delegate fails.
     *
     * @param preferredDelegate The preferred hardware delegate
     * @return The actual delegate that was successfully initialized
     */
    fun initialize(preferredDelegate: DelegateType = DelegateType.GPU): DelegateType {
        lock.withLock {
            // Close existing interpreter if any
            close()

            val delegatesToTry = when (preferredDelegate) {
                DelegateType.GPU -> listOf(DelegateType.GPU, DelegateType.NNAPI, DelegateType.CPU)
                DelegateType.NNAPI -> listOf(DelegateType.NNAPI, DelegateType.CPU)
                DelegateType.CPU -> listOf(DelegateType.CPU)
            }

            for (delegateType in delegatesToTry) {
                try {
                    val success = initializeWithDelegate(delegateType)
                    if (success) {
                        currentDelegate = delegateType
                        Log.i(TAG, "Successfully initialized with $delegateType delegate")
                        return delegateType
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to initialize with $delegateType: ${e.message}")
                }
            }

            throw RuntimeException("Failed to initialize model with any delegate")
        }
    }

    /**
     * Attempt initialization with a specific delegate type.
     */
    private fun initializeWithDelegate(delegateType: DelegateType): Boolean {
        val modelBuffer = loadModelFile()

        val options = Interpreter.Options().apply {
            setNumThreads(4)

            when (delegateType) {
                DelegateType.GPU -> {
                    // Check GPU compatibility
                    val compatList = CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        val gpuOptions = GpuDelegate.Options().apply {
                            setQuantizedModelsAllowed(true)
                            setPrecisionLossAllowed(true)
                        }
                        gpuDelegate = GpuDelegate(gpuOptions)
                        addDelegate(gpuDelegate)
                        Log.d(TAG, "GPU delegate configured")
                    } else {
                        Log.w(TAG, "GPU delegate not supported on this device")
                        return false
                    }
                }
                DelegateType.NNAPI -> {
                    try {
                        nnapiDelegate = NnApiDelegate()
                        addDelegate(nnapiDelegate)
                        Log.d(TAG, "NNAPI delegate configured")
                    } catch (e: Exception) {
                        Log.w(TAG, "NNAPI delegate not available: ${e.message}")
                        return false
                    }
                }
                DelegateType.CPU -> {
                    // CPU fallback - no delegate needed
                    Log.d(TAG, "Using CPU (no hardware delegate)")
                }
            }
        }

        interpreter = Interpreter(modelBuffer, options)

        // Read model input/output shapes
        readModelDimensions()

        // Allocate buffers
        allocateBuffers()

        return true
    }

    /**
     * Load TFLite model from assets folder.
     */
    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILENAME)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Read input/output dimensions from the loaded model.
     */
    private fun readModelDimensions() {
        interpreter?.let { interp ->
            // Get input tensor info
            val inputTensor = interp.getInputTensor(0)
            val inputShape = inputTensor.shape()

            // Expected: [1, H, W, 3] or [1, 3, H, W] (NCHW)
            Log.d(TAG, "Input shape: ${inputShape.contentToString()}")

            when {
                inputShape.size == 4 && inputShape[3] == 3 -> {
                    // NHWC format: [1, H, W, 3]
                    inputHeight = inputShape[1]
                    inputWidth = inputShape[2]
                }
                inputShape.size == 4 && inputShape[1] == 3 -> {
                    // NCHW format: [1, 3, H, W]
                    inputHeight = inputShape[2]
                    inputWidth = inputShape[3]
                }
                else -> {
                    Log.w(TAG, "Unexpected input shape, using defaults")
                    inputHeight = DEFAULT_INPUT_HEIGHT
                    inputWidth = DEFAULT_INPUT_WIDTH
                }
            }

            Log.i(TAG, "Model input dimensions: ${inputWidth}x${inputHeight}")

            // Log output tensor info for debugging
            val numOutputs = interp.outputTensorCount
            for (i in 0 until numOutputs) {
                val outputTensor = interp.getOutputTensor(i)
                Log.d(TAG, "Output $i shape: ${outputTensor.shape().contentToString()}, type: ${outputTensor.dataType()}")
            }
        }
    }

    /**
     * Allocate reusable buffers based on model dimensions.
     */
    private fun allocateBuffers() {
        // Input buffer: [1, H, W, 3] * 4 bytes per float
        val inputBufferSize = 1 * inputHeight * inputWidth * 3 * 4
        inputBuffer = ByteBuffer.allocateDirect(inputBufferSize).apply {
            order(ByteOrder.nativeOrder())
        }

        // Output buffer: [1, H, W] * 4 bytes per float (assuming semantic segmentation)
        // Note: For instance segmentation, this may need adjustment based on actual model output
        val outputBufferSize = 1 * inputHeight * inputWidth * 4
        outputBuffer = ByteBuffer.allocateDirect(outputBufferSize).apply {
            order(ByteOrder.nativeOrder())
        }

        // Alternative output array for models that output multi-dimensional arrays
        outputArray = Array(1) { Array(inputHeight) { FloatArray(inputWidth) } }

        // Pixel buffer for bitmap processing
        pixelBuffer = IntArray(inputWidth * inputHeight)
    }

    /**
     * Run segmentation inference on a bitmap.
     *
     * @param bitmap Input bitmap (will be scaled to model input size)
     * @return SegmentationResult with class mask and timing info
     */
    fun runInference(bitmap: Bitmap): SegmentationResult? {
        lock.withLock {
            val interp = interpreter ?: run {
                Log.e(TAG, "Interpreter not initialized")
                return null
            }

            val startTime = System.currentTimeMillis()

            try {
                // Scale bitmap to model input size if needed
                val scaledBitmap = if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
                    Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
                } else {
                    bitmap
                }

                // Convert bitmap to input buffer
                preprocessBitmap(scaledBitmap)

                // Run inference
                val outputMap = runModelInference(interp)

                // Post-process output to class mask
                val classMask = postprocessOutput(outputMap)

                val inferenceTime = System.currentTimeMillis() - startTime

                // Recycle scaled bitmap if we created it
                if (scaledBitmap !== bitmap) {
                    scaledBitmap.recycle()
                }

                return SegmentationResult(
                    classMask = classMask,
                    width = inputWidth,
                    height = inputHeight,
                    inferenceTimeMs = inferenceTime
                )

            } catch (e: Exception) {
                Log.e(TAG, "Inference failed: ${e.message}", e)
                return null
            }
        }
    }

    /**
     * Convert bitmap pixels to normalized float buffer for model input.
     * RGB values are normalized to [0, 1] range.
     */
    private fun preprocessBitmap(bitmap: Bitmap) {
        val buffer = inputBuffer ?: return
        buffer.rewind()

        val pixels = pixelBuffer ?: IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        // Convert each pixel to normalized RGB floats
        for (pixel in pixels) {
            // Extract RGB values and normalize to [0, 1]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
    }

    /**
     * Execute model inference with appropriate output handling.
     * Automatically detects output format (single tensor vs multiple tensors).
     */
    private fun runModelInference(interp: Interpreter): Map<Int, Any> {
        val inputBuffer = this.inputBuffer ?: throw IllegalStateException("Input buffer not allocated")
        inputBuffer.rewind()

        val numOutputs = interp.outputTensorCount
        val outputMap = mutableMapOf<Int, Any>()

        // Allocate output tensors based on model specification
        for (i in 0 until numOutputs) {
            val outputTensor = interp.getOutputTensor(i)
            val shape = outputTensor.shape()

            // Create appropriately shaped output buffer
            val outputArray = when (shape.size) {
                2 -> Array(shape[0]) { FloatArray(shape[1]) }
                3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
                4 -> Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } } }
                else -> {
                    // Fallback to flat buffer
                    val totalSize = shape.fold(1) { acc, dim -> acc * dim }
                    FloatArray(totalSize)
                }
            }
            outputMap[i] = outputArray
        }

        // Run inference
        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        return outputMap
    }

    /**
     * Post-process model output to produce a 2D class mask.
     * Handles both semantic segmentation (single class per pixel) and
     * instance segmentation (multiple classes) output formats.
     *
     * ASSUMPTION: Primary segmentation output is at index 0.
     * Shape is expected to be [1, H, W] or [H, W] with class IDs.
     * If output is [1, H, W, C], argmax is applied across channels.
     */
    private fun postprocessOutput(outputMap: Map<Int, Any>): Array<IntArray> {
        val primaryOutput = outputMap[0] ?: throw IllegalStateException("No output tensor found")

        val classMask = Array(inputHeight) { IntArray(inputWidth) }

        when (primaryOutput) {
            // Shape: [1, H, W] - batch of 1, semantic mask
            is Array<*> -> {
                @Suppress("UNCHECKED_CAST")
                when (val inner = primaryOutput[0]) {
                    // [H, W] float array
                    is Array<*> -> {
                        val floatMask = inner as Array<FloatArray>
                        for (y in 0 until inputHeight) {
                            for (x in 0 until inputWidth) {
                                // Round float to int class ID
                                classMask[y][x] = floatMask[y][x].toInt().coerceAtLeast(0)
                            }
                        }
                    }
                    // Single FloatArray (flattened)
                    is FloatArray -> {
                        for (y in 0 until inputHeight) {
                            for (x in 0 until inputWidth) {
                                classMask[y][x] = inner[y * inputWidth + x].toInt().coerceAtLeast(0)
                            }
                        }
                    }
                    else -> {
                        // Try to handle [1, H, W, C] format with argmax
                        processMultiChannelOutput(primaryOutput, classMask)
                    }
                }
            }
            // Flat float array
            is FloatArray -> {
                for (y in 0 until inputHeight) {
                    for (x in 0 until inputWidth) {
                        classMask[y][x] = primaryOutput[y * inputWidth + x].toInt().coerceAtLeast(0)
                    }
                }
            }
            else -> {
                Log.w(TAG, "Unexpected output type: ${primaryOutput.javaClass}")
            }
        }

        return classMask
    }

    /**
     * Handle multi-channel output format [1, H, W, C] by applying argmax.
     * This is used when the model outputs per-class probabilities instead of class IDs.
     */
    @Suppress("UNCHECKED_CAST")
    private fun processMultiChannelOutput(output: Any, classMask: Array<IntArray>) {
        try {
            // Attempt to cast as [1, H, W, C] format
            val batch = output as Array<Array<Array<FloatArray>>>
            val data = batch[0]  // First batch element

            for (y in 0 until inputHeight.coerceAtMost(data.size)) {
                for (x in 0 until inputWidth.coerceAtMost(data[y].size)) {
                    val channelValues = data[y][x]
                    // Find argmax (class with highest probability)
                    var maxIdx = 0
                    var maxVal = channelValues[0]
                    for (c in 1 until channelValues.size) {
                        if (channelValues[c] > maxVal) {
                            maxVal = channelValues[c]
                            maxIdx = c
                        }
                    }
                    classMask[y][x] = maxIdx
                }
            }
        } catch (e: ClassCastException) {
            Log.w(TAG, "Could not process multi-channel output: ${e.message}")
        }
    }

    /**
     * Release all resources.
     */
    fun close() {
        lock.withLock {
            interpreter?.close()
            interpreter = null

            gpuDelegate?.close()
            gpuDelegate = null

            nnapiDelegate?.close()
            nnapiDelegate = null

            inputBuffer = null
            outputBuffer = null
            outputArray = null
            pixelBuffer = null
        }
    }

    /**
     * Check if model is initialized and ready for inference.
     */
    fun isInitialized(): Boolean = lock.withLock { interpreter != null }
}
