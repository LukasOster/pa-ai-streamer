package com.segmentation.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
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
 * Unified TensorFlow Lite Model wrapper supporting both:
 * - Object Detection (bounding boxes + class labels)
 * - Semantic Segmentation (pixel-wise class masks)
 *
 * The model type is auto-detected based on output tensor shapes,
 * or can be explicitly set.
 *
 * OBJECT DETECTION OUTPUT:
 *   - boxes: [1, num_queries, 4] - normalized xyxy or cxcywh coordinates
 *   - logits: [1, num_queries, num_classes] - class scores
 *
 * SEGMENTATION OUTPUT:
 *   - mask: [1, H, W] with class IDs, or
 *   - mask: [1, H, W, C] with per-class probabilities (argmax applied)
 */
class TFLiteModel(private val context: Context) {

    companion object {
        private const val TAG = "TFLiteModel"

        // Default model dimensions
        private const val DEFAULT_INPUT_HEIGHT = 512
        private const val DEFAULT_INPUT_WIDTH = 512

        // Detection thresholds
        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.5f
        const val DEFAULT_NMS_THRESHOLD = 0.4f
        const val MAX_DETECTIONS = 100
    }

    /**
     * Model type enumeration
     */
    enum class ModelType {
        OBJECT_DETECTION,
        SEGMENTATION,
        AUTO_DETECT  // Will be resolved to one of the above based on output shape
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
     * Bounding box format enumeration
     */
    enum class BoxFormat {
        XYXY,      // [x1, y1, x2, y2] - corners
        CXCYWH,    // [cx, cy, w, h] - center + size
        XYWH       // [x, y, w, h] - top-left + size
    }

    /**
     * Single detection result
     */
    data class Detection(
        val boundingBox: RectF,  // Normalized coordinates [0, 1]
        val classId: Int,
        val confidence: Float,
        val label: String? = null
    )

    /**
     * Object detection result
     */
    data class DetectionResult(
        val detections: List<Detection>,
        val inferenceTimeMs: Long
    )

    /**
     * Segmentation result
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

    /**
     * Unified inference result
     */
    sealed class InferenceResult {
        data class Detections(val result: DetectionResult) : InferenceResult()
        data class Segmentation(val result: SegmentationResult) : InferenceResult()
    }

    // Interpreter and delegate references
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null

    // Model dimensions
    var inputWidth: Int = DEFAULT_INPUT_WIDTH
        private set
    var inputHeight: Int = DEFAULT_INPUT_HEIGHT
        private set

    // Model configuration
    var modelType: ModelType = ModelType.AUTO_DETECT
        private set
    var resolvedModelType: ModelType = ModelType.AUTO_DETECT
        private set
    var boxFormat: BoxFormat = BoxFormat.CXCYWH
    var confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD
    var nmsThreshold: Float = DEFAULT_NMS_THRESHOLD

    // Class labels (optional)
    var classLabels: List<String> = emptyList()

    // Current delegate type
    var currentDelegate: DelegateType = DelegateType.GPU
        private set

    // Thread safety
    private val lock = ReentrantLock()

    // Pre-allocated buffers
    private var inputBuffer: ByteBuffer? = null
    private var pixelBuffer: IntArray? = null

    // Model file name
    private var modelFileName: String = "rf_detr_segmentation.tflite"

    /**
     * Initialize with specified model file and delegate.
     */
    fun initialize(
        modelFile: String = "rf_detr_segmentation.tflite",
        preferredDelegate: DelegateType = DelegateType.GPU,
        modelType: ModelType = ModelType.AUTO_DETECT
    ): DelegateType {
        lock.withLock {
            close()

            this.modelFileName = modelFile
            this.modelType = modelType

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
                        Log.i(TAG, "Model type: $resolvedModelType")
                        return delegateType
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to initialize with $delegateType: ${e.message}")
                }
            }

            throw RuntimeException("Failed to initialize model with any delegate")
        }
    }

    private fun initializeWithDelegate(delegateType: DelegateType): Boolean {
        val modelBuffer = loadModelFile()

        val options = Interpreter.Options().apply {
            setNumThreads(4)

            when (delegateType) {
                DelegateType.GPU -> {
                    val compatList = CompatibilityList()
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        gpuDelegate = GpuDelegate(compatList.bestOptionsForThisDevice)
                        addDelegate(gpuDelegate)
                        Log.d(TAG, "GPU delegate configured")
                    } else {
                        Log.w(TAG, "GPU delegate not supported")
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
                    Log.d(TAG, "Using CPU (no hardware delegate)")
                }
            }
        }

        interpreter = Interpreter(modelBuffer, options)

        readModelDimensions()
        detectModelType()
        allocateBuffers()

        return true
    }

    private fun loadModelFile(): MappedByteBuffer {
        Log.d(TAG, "Loading model file: $modelFileName")

        // List available assets for debugging
        try {
            val assetList = context.assets.list("") ?: emptyArray()
            Log.d(TAG, "Available assets: ${assetList.joinToString()}")
        } catch (e: Exception) {
            Log.w(TAG, "Could not list assets: ${e.message}")
        }

        val assetFileDescriptor = context.assets.openFd(modelFileName)
        Log.d(TAG, "Asset file descriptor obtained, length: ${assetFileDescriptor.declaredLength}")

        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun readModelDimensions() {
        interpreter?.let { interp ->
            val inputTensor = interp.getInputTensor(0)
            val inputShape = inputTensor.shape()
            Log.d(TAG, "Input shape: ${inputShape.contentToString()}")

            when {
                inputShape.size == 4 && inputShape[3] == 3 -> {
                    // NHWC: [1, H, W, 3]
                    inputHeight = inputShape[1]
                    inputWidth = inputShape[2]
                }
                inputShape.size == 4 && inputShape[1] == 3 -> {
                    // NCHW: [1, 3, H, W]
                    inputHeight = inputShape[2]
                    inputWidth = inputShape[3]
                }
                else -> {
                    Log.w(TAG, "Unexpected input shape, using defaults")
                }
            }

            Log.i(TAG, "Model input: ${inputWidth}x${inputHeight}")
        }
    }

    /**
     * Auto-detect model type based on output tensor shapes.
     */
    private fun detectModelType() {
        if (modelType != ModelType.AUTO_DETECT) {
            resolvedModelType = modelType
            return
        }

        interpreter?.let { interp ->
            val numOutputs = interp.outputTensorCount
            Log.d(TAG, "Number of outputs: $numOutputs")

            for (i in 0 until numOutputs) {
                val tensor = interp.getOutputTensor(i)
                val shape = tensor.shape()
                Log.d(TAG, "Output $i: ${shape.contentToString()} (${tensor.dataType()})")
            }

            // Heuristic for detection:
            // - Usually 2 outputs: boxes [1, N, 4] and logits [1, N, C]
            // - Or combined [1, N, 4+C]
            //
            // Heuristic for segmentation:
            // - Usually 1 output: [1, H, W] or [1, H, W, C] or [1, C, H, W]

            if (numOutputs >= 2) {
                val out0 = interp.getOutputTensor(0).shape()
                val out1 = interp.getOutputTensor(1).shape()

                // Check if it looks like boxes + logits
                val hasBoxesShape = out0.size == 3 && out0[2] == 4
                val hasLogitsShape = out1.size == 3 && out1[2] > 1

                if (hasBoxesShape || hasLogitsShape) {
                    resolvedModelType = ModelType.OBJECT_DETECTION
                    Log.i(TAG, "Auto-detected: OBJECT_DETECTION")
                    return
                }
            }

            // Check for segmentation pattern
            if (numOutputs == 1) {
                val out0 = interp.getOutputTensor(0).shape()

                // [1, H, W] or [1, H, W, C] where H,W are similar to input
                if (out0.size == 3 || out0.size == 4) {
                    val h = if (out0.size == 3) out0[1] else out0[1]
                    val w = if (out0.size == 3) out0[2] else out0[2]

                    // If output spatial dims are close to input, likely segmentation
                    if (h > 32 && w > 32) {
                        resolvedModelType = ModelType.SEGMENTATION
                        Log.i(TAG, "Auto-detected: SEGMENTATION")
                        return
                    }
                }
            }

            // Default to detection if unsure
            resolvedModelType = ModelType.OBJECT_DETECTION
            Log.i(TAG, "Auto-detect fallback: OBJECT_DETECTION")
        }
    }

    private fun allocateBuffers() {
        val inputBufferSize = 1 * inputHeight * inputWidth * 3 * 4
        inputBuffer = ByteBuffer.allocateDirect(inputBufferSize).apply {
            order(ByteOrder.nativeOrder())
        }
        pixelBuffer = IntArray(inputWidth * inputHeight)
    }

    /**
     * Run inference and return appropriate result type.
     */
    fun runInference(bitmap: Bitmap): InferenceResult? {
        return when (resolvedModelType) {
            ModelType.OBJECT_DETECTION -> {
                runDetection(bitmap)?.let { InferenceResult.Detections(it) }
            }
            ModelType.SEGMENTATION -> {
                runSegmentation(bitmap)?.let { InferenceResult.Segmentation(it) }
            }
            else -> null
        }
    }

    /**
     * Run object detection inference.
     */
    fun runDetection(bitmap: Bitmap): DetectionResult? {
        lock.withLock {
            val interp = interpreter ?: return null
            val startTime = System.currentTimeMillis()

            try {
                val scaledBitmap = if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
                    Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
                } else {
                    bitmap
                }

                preprocessBitmap(scaledBitmap)

                val outputs = runModelInference(interp)
                val detections = postprocessDetections(outputs, bitmap.width, bitmap.height)

                val inferenceTime = System.currentTimeMillis() - startTime

                if (scaledBitmap !== bitmap) {
                    scaledBitmap.recycle()
                }

                return DetectionResult(detections, inferenceTime)

            } catch (e: Exception) {
                Log.e(TAG, "Detection inference failed: ${e.message}", e)
                return null
            }
        }
    }

    /**
     * Run segmentation inference.
     */
    fun runSegmentation(bitmap: Bitmap): SegmentationResult? {
        lock.withLock {
            val interp = interpreter ?: return null
            val startTime = System.currentTimeMillis()

            try {
                val scaledBitmap = if (bitmap.width != inputWidth || bitmap.height != inputHeight) {
                    Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
                } else {
                    bitmap
                }

                preprocessBitmap(scaledBitmap)

                val outputs = runModelInference(interp)
                val classMask = postprocessSegmentation(outputs)

                val inferenceTime = System.currentTimeMillis() - startTime

                if (scaledBitmap !== bitmap) {
                    scaledBitmap.recycle()
                }

                return SegmentationResult(classMask, inputWidth, inputHeight, inferenceTime)

            } catch (e: Exception) {
                Log.e(TAG, "Segmentation inference failed: ${e.message}", e)
                return null
            }
        }
    }

    // ImageNet normalization constants (used by RF-DETR)
    private val imagenetMeans = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val imagenetStds = floatArrayOf(0.229f, 0.224f, 0.225f)

    // Set to true to use ImageNet normalization (required for RF-DETR)
    var useImageNetNormalization: Boolean = true

    private fun preprocessBitmap(bitmap: Bitmap) {
        val buffer = inputBuffer ?: return
        buffer.rewind()

        val pixels = pixelBuffer ?: IntArray(inputWidth * inputHeight)
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        for (pixel in pixels) {
            // Extract RGB and normalize to [0, 1]
            var r = ((pixel shr 16) and 0xFF) / 255.0f
            var g = ((pixel shr 8) and 0xFF) / 255.0f
            var b = (pixel and 0xFF) / 255.0f

            // Apply ImageNet normalization if enabled
            if (useImageNetNormalization) {
                r = (r - imagenetMeans[0]) / imagenetStds[0]
                g = (g - imagenetMeans[1]) / imagenetStds[1]
                b = (b - imagenetMeans[2]) / imagenetStds[2]
            }

            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
    }

    private fun runModelInference(interp: Interpreter): Map<Int, Any> {
        val inputBuffer = this.inputBuffer ?: throw IllegalStateException("Input buffer not allocated")
        inputBuffer.rewind()

        val numOutputs = interp.outputTensorCount
        val outputMap = mutableMapOf<Int, Any>()

        for (i in 0 until numOutputs) {
            val outputTensor = interp.getOutputTensor(i)
            val shape = outputTensor.shape()

            val outputArray: Any = when (shape.size) {
                1 -> FloatArray(shape[0])
                2 -> Array(shape[0]) { FloatArray(shape[1]) }
                3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
                4 -> Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } } }
                else -> FloatArray(shape.fold(1) { acc, dim -> acc * dim })
            }
            outputMap[i] = outputArray
        }

        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
        return outputMap
    }

    /**
     * Post-process detection outputs.
     * Handles multiple output formats:
     * - [N, 4] / [N, C] (no batch dimension)
     * - [1, N, 4] / [1, N, C] (with batch dimension)
     */
    @Suppress("UNCHECKED_CAST")
    private fun postprocessDetections(
        outputs: Map<Int, Any>,
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            // Find boxes [N, 4] and logits [N, C] tensors
            var boxes: Array<FloatArray>? = null
            var logits: Array<FloatArray>? = null

            for ((idx, output) in outputs) {
                Log.d(TAG, "Processing output $idx: ${output.javaClass.simpleName}")

                when (output) {
                    // 2D array: [N, D] - no batch dimension
                    is Array<*> -> {
                        if (output.isNotEmpty() && output[0] is FloatArray) {
                            val arr = output as Array<FloatArray>
                            val lastDim = arr[0].size
                            Log.d(TAG, "Output $idx: [${arr.size}, $lastDim]")

                            when (lastDim) {
                                4 -> {
                                    boxes = arr
                                    Log.d(TAG, "Found boxes at output $idx")
                                }
                                else -> {
                                    if (lastDim > 4) {
                                        logits = arr
                                        Log.d(TAG, "Found logits at output $idx (classes=$lastDim)")
                                    }
                                }
                            }
                        }
                        // 3D array: [1, N, D] - with batch dimension
                        else if (output.isNotEmpty() && output[0] is Array<*>) {
                            val outer = output[0] as? Array<FloatArray>
                            if (outer != null && outer.isNotEmpty()) {
                                val lastDim = outer[0].size
                                Log.d(TAG, "Output $idx: [1, ${outer.size}, $lastDim]")

                                when (lastDim) {
                                    4 -> {
                                        boxes = outer
                                        Log.d(TAG, "Found boxes at output $idx (with batch)")
                                    }
                                    else -> {
                                        if (lastDim > 4) {
                                            logits = outer
                                            Log.d(TAG, "Found logits at output $idx (with batch, classes=$lastDim)")
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (boxes == null || logits == null) {
                Log.w(TAG, "Could not find boxes or logits in outputs. boxes=$boxes, logits=$logits")
                return detections
            }

            val numQueries = boxes.size
            Log.d(TAG, "Processing $numQueries queries")

            // Log first few samples for debugging
            for (i in 0 until minOf(3, numQueries)) {
                Log.d(TAG, "Sample $i - box: ${boxes[i].contentToString()}, logits max: ${logits[i].maxOrNull()}")
            }

            for (i in 0 until numQueries) {
                val box = boxes[i]
                val classLogits = logits[i]

                // Find best class (argmax) and apply softmax for confidence
                var maxLogit = Float.NEGATIVE_INFINITY
                var bestClass = 0
                for (c in classLogits.indices) {
                    if (classLogits[c] > maxLogit) {
                        maxLogit = classLogits[c]
                        bestClass = c
                    }
                }

                // Use sigmoid for confidence (DETR typically uses sigmoid, not softmax)
                val confidence = sigmoid(maxLogit)

                if (confidence < confidenceThreshold) continue

                // Log detection before filtering
                if (i < 10 || confidence > 0.3f) {
                    Log.d(TAG, "Detection $i: class=$bestClass conf=$confidence box=${box.contentToString()}")
                }

                // Convert box format to RectF [0, 1]
                val rectF = convertBoxToRectF(box)

                // Clamp to valid range
                rectF.left = rectF.left.coerceIn(0f, 1f)
                rectF.top = rectF.top.coerceIn(0f, 1f)
                rectF.right = rectF.right.coerceIn(0f, 1f)
                rectF.bottom = rectF.bottom.coerceIn(0f, 1f)

                // Skip invalid boxes
                if (rectF.width() <= 0 || rectF.height() <= 0) continue

                val label = if (bestClass < classLabels.size) classLabels[bestClass] else null

                detections.add(Detection(rectF, bestClass, confidence, label))
            }

            // Sort by confidence and apply NMS
            val sorted = detections.sortedByDescending { it.confidence }
            val nmsResult = applyNMS(sorted, nmsThreshold)

            return nmsResult.take(MAX_DETECTIONS)

        } catch (e: Exception) {
            Log.e(TAG, "Error post-processing detections: ${e.message}", e)
            return detections
        }
    }

    private fun convertBoxToRectF(box: FloatArray): RectF {
        return when (boxFormat) {
            BoxFormat.XYXY -> {
                RectF(box[0], box[1], box[2], box[3])
            }
            BoxFormat.CXCYWH -> {
                val cx = box[0]
                val cy = box[1]
                val w = box[2]
                val h = box[3]
                RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            }
            BoxFormat.XYWH -> {
                RectF(box[0], box[1], box[0] + box[2], box[1] + box[3])
            }
        }
    }

    private fun sigmoid(x: Float): Float {
        return (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
    }

    /**
     * Apply Non-Maximum Suppression to remove overlapping boxes.
     */
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return detections

        val result = mutableListOf<Detection>()
        val active = BooleanArray(detections.size) { true }

        for (i in detections.indices) {
            if (!active[i]) continue

            result.add(detections[i])

            for (j in i + 1 until detections.size) {
                if (!active[j]) continue

                // Only suppress same class
                if (detections[i].classId != detections[j].classId) continue

                val iou = calculateIoU(detections[i].boundingBox, detections[j].boundingBox)
                if (iou > iouThreshold) {
                    active[j] = false
                }
            }
        }

        return result
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val intersectLeft = maxOf(a.left, b.left)
        val intersectTop = maxOf(a.top, b.top)
        val intersectRight = minOf(a.right, b.right)
        val intersectBottom = minOf(a.bottom, b.bottom)

        if (intersectLeft >= intersectRight || intersectTop >= intersectBottom) {
            return 0f
        }

        val intersectArea = (intersectRight - intersectLeft) * (intersectBottom - intersectTop)
        val aArea = a.width() * a.height()
        val bArea = b.width() * b.height()
        val unionArea = aArea + bArea - intersectArea

        return if (unionArea > 0) intersectArea / unionArea else 0f
    }

    /**
     * Post-process segmentation output to class mask.
     */
    @Suppress("UNCHECKED_CAST")
    private fun postprocessSegmentation(outputs: Map<Int, Any>): Array<IntArray> {
        val primaryOutput = outputs[0] ?: throw IllegalStateException("No output tensor")
        val classMask = Array(inputHeight) { IntArray(inputWidth) }

        when (primaryOutput) {
            is Array<*> -> {
                when (val inner = primaryOutput[0]) {
                    is Array<*> -> {
                        // [1, H, W] format
                        if (inner[0] is FloatArray) {
                            val floatMask = inner as Array<FloatArray>
                            for (y in 0 until inputHeight.coerceAtMost(floatMask.size)) {
                                for (x in 0 until inputWidth.coerceAtMost(floatMask[y].size)) {
                                    classMask[y][x] = floatMask[y][x].toInt().coerceAtLeast(0)
                                }
                            }
                        } else if (inner[0] is Array<*>) {
                            // [1, H, W, C] format - apply argmax
                            processMultiChannelSegmentation(primaryOutput as Array<Array<Array<FloatArray>>>, classMask)
                        }
                    }
                    is FloatArray -> {
                        for (y in 0 until inputHeight) {
                            for (x in 0 until inputWidth) {
                                classMask[y][x] = inner[y * inputWidth + x].toInt().coerceAtLeast(0)
                            }
                        }
                    }
                }
            }
            is FloatArray -> {
                for (y in 0 until inputHeight) {
                    for (x in 0 until inputWidth) {
                        classMask[y][x] = primaryOutput[y * inputWidth + x].toInt().coerceAtLeast(0)
                    }
                }
            }
        }

        return classMask
    }

    private fun processMultiChannelSegmentation(output: Array<Array<Array<FloatArray>>>, classMask: Array<IntArray>) {
        val data = output[0]
        for (y in 0 until inputHeight.coerceAtMost(data.size)) {
            for (x in 0 until inputWidth.coerceAtMost(data[y].size)) {
                val channels = data[y][x]
                var maxIdx = 0
                var maxVal = channels[0]
                for (c in 1 until channels.size) {
                    if (channels[c] > maxVal) {
                        maxVal = channels[c]
                        maxIdx = c
                    }
                }
                classMask[y][x] = maxIdx
            }
        }
    }

    fun close() {
        lock.withLock {
            interpreter?.close()
            interpreter = null
            gpuDelegate?.close()
            gpuDelegate = null
            nnapiDelegate?.close()
            nnapiDelegate = null
            inputBuffer = null
            pixelBuffer = null
        }
    }

    fun isInitialized(): Boolean = lock.withLock { interpreter != null }
}
