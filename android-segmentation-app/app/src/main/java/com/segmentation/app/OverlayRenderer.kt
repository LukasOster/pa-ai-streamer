package com.segmentation.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Unified Overlay Renderer supporting both:
 * - Object Detection: Bounding boxes with labels
 * - Segmentation: Colored pixel masks
 *
 * Uses SurfaceView for hardware-accelerated rendering independent of UI thread.
 */
class OverlayRenderer(private val surfaceView: SurfaceView) : SurfaceHolder.Callback {

    companion object {
        private const val TAG = "OverlayRenderer"
        private const val DEFAULT_MASK_ALPHA = 0.4f
        private const val DEFAULT_BOX_STROKE_WIDTH = 4f
        private const val DEFAULT_LABEL_TEXT_SIZE = 36f
        private const val MAX_CLASSES = 256

        /**
         * Pre-defined color palette for classes.
         */
        private val CLASS_COLORS = intArrayOf(
            Color.TRANSPARENT,       // 0: Background
            Color.rgb(255, 0, 0),    // 1: Red
            Color.rgb(0, 255, 0),    // 2: Green
            Color.rgb(0, 0, 255),    // 3: Blue
            Color.rgb(255, 255, 0),  // 4: Yellow
            Color.rgb(255, 0, 255),  // 5: Magenta
            Color.rgb(0, 255, 255),  // 6: Cyan
            Color.rgb(255, 128, 0),  // 7: Orange
            Color.rgb(128, 0, 255),  // 8: Purple
            Color.rgb(0, 255, 128),  // 9: Spring Green
            Color.rgb(255, 128, 128),// 10: Light Red
            Color.rgb(128, 255, 128),// 11: Light Green
            Color.rgb(128, 128, 255),// 12: Light Blue
            Color.rgb(255, 200, 0),  // 13: Gold
            Color.rgb(200, 0, 255),  // 14: Violet
            Color.rgb(0, 200, 255),  // 15: Sky Blue
            Color.rgb(255, 100, 100),// 16: Salmon
            Color.rgb(100, 255, 100),// 17: Lime
            Color.rgb(100, 100, 255),// 18: Periwinkle
            Color.rgb(255, 180, 0),  // 19: Amber
            Color.rgb(180, 0, 255),  // 20: Electric Purple
        )
    }

    // Surface state
    private val surfaceHolder: SurfaceHolder = surfaceView.holder
    private val surfaceReady = AtomicBoolean(false)
    private var surfaceWidth: Int = 0
    private var surfaceHeight: Int = 0

    // Current data (thread-safe)
    private val currentDetections = AtomicReference<List<TFLiteModel.Detection>?>(null)
    private val currentMask = AtomicReference<SegmentationMaskData?>(null)

    // Rendering state
    private val renderLock = ReentrantLock()

    // Configuration
    var maskAlpha: Float = DEFAULT_MASK_ALPHA
        set(value) {
            field = value.coerceIn(0f, 1f)
            updateColorTable()
        }

    var boxStrokeWidth: Float = DEFAULT_BOX_STROKE_WIDTH
    var labelTextSize: Float = DEFAULT_LABEL_TEXT_SIZE
    var showLabels: Boolean = true
    var showConfidence: Boolean = true

    // Pre-computed color table
    private var colorTableWithAlpha = IntArray(MAX_CLASSES)

    // Reusable bitmap for mask rendering
    private var maskBitmap: Bitmap? = null
    private var maskPixels: IntArray? = null

    // Paints
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = DEFAULT_BOX_STROKE_WIDTH
        isAntiAlias = true
    }

    private val boxFillPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val labelBackgroundPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(180, 0, 0, 0)
    }

    private val labelTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = DEFAULT_LABEL_TEXT_SIZE
        isAntiAlias = true
        isFakeBoldText = true
    }

    private val bitmapPaint = Paint().apply {
        isAntiAlias = false
        isFilterBitmap = true
    }

    /**
     * Data class for segmentation mask transfer
     */
    data class SegmentationMaskData(
        val classMask: Array<IntArray>,
        val width: Int,
        val height: Int
    )

    init {
        surfaceHolder.addCallback(this)
        surfaceHolder.setFormat(android.graphics.PixelFormat.TRANSLUCENT)
        surfaceView.setZOrderOnTop(true)
        updateColorTable()
    }

    private fun updateColorTable() {
        val alpha = (maskAlpha * 255).toInt()
        for (i in 0 until MAX_CLASSES) {
            val baseColor = if (i < CLASS_COLORS.size) {
                CLASS_COLORS[i]
            } else {
                generateColorForClass(i)
            }
            colorTableWithAlpha[i] = if (i == 0) {
                Color.TRANSPARENT
            } else {
                Color.argb(alpha, Color.red(baseColor), Color.green(baseColor), Color.blue(baseColor))
            }
        }
    }

    private fun generateColorForClass(classId: Int): Int {
        val hue = (classId * 137.508f) % 360f
        val saturation = 0.7f + (classId % 3) * 0.1f
        val value = 0.8f + (classId % 2) * 0.1f
        return Color.HSVToColor(floatArrayOf(hue, saturation, value))
    }

    /**
     * Get the full color (without mask alpha) for a class.
     */
    fun getClassColor(classId: Int): Int {
        return if (classId < CLASS_COLORS.size) {
            CLASS_COLORS[classId]
        } else {
            generateColorForClass(classId)
        }
    }

    // ==================== DETECTION RENDERING ====================

    /**
     * Update with detection results and trigger render.
     */
    fun updateDetections(detections: List<TFLiteModel.Detection>) {
        currentDetections.set(detections)
        currentMask.set(null)  // Clear mask when showing detections

        if (surfaceReady.get()) {
            render()
        }
    }

    // ==================== SEGMENTATION RENDERING ====================

    /**
     * Update with segmentation result and trigger render.
     */
    fun updateMask(result: TFLiteModel.SegmentationResult) {
        val maskData = SegmentationMaskData(
            classMask = result.classMask,
            width = result.width,
            height = result.height
        )
        currentMask.set(maskData)
        currentDetections.set(null)  // Clear detections when showing mask

        if (surfaceReady.get()) {
            render()
        }
    }

    // ==================== MAIN RENDER ====================

    /**
     * Render current overlay (detections or mask).
     */
    fun render() {
        if (!surfaceReady.get()) return

        renderLock.withLock {
            val canvas: Canvas? = try {
                surfaceHolder.lockCanvas()
            } catch (e: Exception) {
                Log.w(TAG, "Failed to lock canvas: ${e.message}")
                null
            }

            canvas?.let { c ->
                try {
                    // Clear with transparent
                    c.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

                    // Draw mask if available
                    currentMask.get()?.let { maskData ->
                        drawMask(c, maskData)
                    }

                    // Draw detections if available
                    currentDetections.get()?.let { detections ->
                        drawDetections(c, detections)
                    }

                } finally {
                    try {
                        surfaceHolder.unlockCanvasAndPost(c)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to unlock canvas: ${e.message}")
                    }
                }
            }
        }
    }

    private fun drawDetections(canvas: Canvas, detections: List<TFLiteModel.Detection>) {
        if (detections.isEmpty()) return

        labelTextPaint.textSize = labelTextSize
        boxPaint.strokeWidth = boxStrokeWidth

        for (detection in detections) {
            val color = getClassColor(detection.classId)

            // Convert normalized coordinates to canvas coordinates
            val rect = RectF(
                detection.boundingBox.left * surfaceWidth,
                detection.boundingBox.top * surfaceHeight,
                detection.boundingBox.right * surfaceWidth,
                detection.boundingBox.bottom * surfaceHeight
            )

            // Draw box fill (semi-transparent)
            boxFillPaint.color = Color.argb(40, Color.red(color), Color.green(color), Color.blue(color))
            canvas.drawRect(rect, boxFillPaint)

            // Draw box outline
            boxPaint.color = color
            canvas.drawRect(rect, boxPaint)

            // Draw label
            if (showLabels || showConfidence) {
                val labelText = buildLabelText(detection)
                drawLabel(canvas, rect, labelText, color)
            }
        }
    }

    private fun buildLabelText(detection: TFLiteModel.Detection): String {
        val parts = mutableListOf<String>()

        if (showLabels) {
            val label = detection.label ?: "Class ${detection.classId}"
            parts.add(label)
        }

        if (showConfidence) {
            val conf = String.format("%.0f%%", detection.confidence * 100)
            parts.add(conf)
        }

        return parts.joinToString(" ")
    }

    private fun drawLabel(canvas: Canvas, boxRect: RectF, text: String, color: Int) {
        val textBounds = android.graphics.Rect()
        labelTextPaint.getTextBounds(text, 0, text.length, textBounds)

        val padding = 8f
        val labelHeight = textBounds.height() + padding * 2
        val labelWidth = textBounds.width() + padding * 2

        // Position label above box, or inside if no room
        var labelTop = boxRect.top - labelHeight
        if (labelTop < 0) {
            labelTop = boxRect.top
        }

        val labelLeft = boxRect.left

        // Draw background
        labelBackgroundPaint.color = Color.argb(200, Color.red(color), Color.green(color), Color.blue(color))
        canvas.drawRect(
            labelLeft,
            labelTop,
            labelLeft + labelWidth,
            labelTop + labelHeight,
            labelBackgroundPaint
        )

        // Draw text
        canvas.drawText(
            text,
            labelLeft + padding,
            labelTop + labelHeight - padding,
            labelTextPaint
        )
    }

    private fun drawMask(canvas: Canvas, maskData: SegmentationMaskData) {
        val maskWidth = maskData.width
        val maskHeight = maskData.height
        val classMask = maskData.classMask

        // Ensure bitmap is allocated
        if (maskBitmap?.width != maskWidth || maskBitmap?.height != maskHeight) {
            maskBitmap?.recycle()
            maskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888)
            maskPixels = IntArray(maskWidth * maskHeight)
        }

        val pixels = maskPixels ?: return
        val bitmap = maskBitmap ?: return

        // Convert mask to colored pixels
        var pixelIndex = 0
        for (y in 0 until maskHeight) {
            for (x in 0 until maskWidth) {
                val classId = classMask[y][x].coerceIn(0, MAX_CLASSES - 1)
                pixels[pixelIndex++] = colorTableWithAlpha[classId]
            }
        }

        bitmap.setPixels(pixels, 0, maskWidth, 0, 0, maskWidth, maskHeight)

        // Calculate destination rect (fill surface, maintain aspect ratio)
        val destRect = calculateDestRect(maskWidth, maskHeight)

        canvas.drawBitmap(
            bitmap,
            Rect(0, 0, maskWidth, maskHeight),
            destRect,
            bitmapPaint
        )
    }

    private fun calculateDestRect(srcWidth: Int, srcHeight: Int): Rect {
        val srcAspect = srcWidth.toFloat() / srcHeight
        val dstAspect = surfaceWidth.toFloat() / surfaceHeight

        val scaledWidth: Int
        val scaledHeight: Int

        if (srcAspect > dstAspect) {
            scaledWidth = surfaceWidth
            scaledHeight = (surfaceWidth / srcAspect).toInt()
        } else {
            scaledHeight = surfaceHeight
            scaledWidth = (surfaceHeight * srcAspect).toInt()
        }

        val left = (surfaceWidth - scaledWidth) / 2
        val top = (surfaceHeight - scaledHeight) / 2

        return Rect(left, top, left + scaledWidth, top + scaledHeight)
    }

    /**
     * Clear the overlay.
     */
    fun clear() {
        currentDetections.set(null)
        currentMask.set(null)

        if (surfaceReady.get()) {
            renderLock.withLock {
                val canvas = try {
                    surfaceHolder.lockCanvas()
                } catch (e: Exception) { null }

                canvas?.let {
                    it.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
                    try {
                        surfaceHolder.unlockCanvasAndPost(it)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to unlock canvas: ${e.message}")
                    }
                }
            }
        }
    }

    /**
     * Release resources.
     */
    fun release() {
        clear()
        maskBitmap?.recycle()
        maskBitmap = null
        maskPixels = null
    }

    // SurfaceHolder.Callback

    override fun surfaceCreated(holder: SurfaceHolder) {
        Log.d(TAG, "Surface created")
        surfaceReady.set(true)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        Log.d(TAG, "Surface changed: ${width}x${height}")
        surfaceWidth = width
        surfaceHeight = height

        // Re-render at new size
        if (currentDetections.get() != null || currentMask.get() != null) {
            render()
        }
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        Log.d(TAG, "Surface destroyed")
        surfaceReady.set(false)
    }
}
