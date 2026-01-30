package com.segmentation.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.Rect
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Renderer for segmentation mask overlay on camera preview.
 *
 * DESIGN DECISION: Using SurfaceView for overlay rendering because:
 * 1. Hardware-accelerated rendering on dedicated surface
 * 2. Independent from UI thread - can render at any rate
 * 3. Better performance for real-time updates vs Canvas on View
 * 4. Direct pixel access without intermediate bitmap copies
 *
 * Alternative approaches considered:
 * - TextureView: More flexible but higher overhead for simple overlays
 * - Custom View with Canvas: Simpler but tied to UI thread refresh rate
 *
 * The SurfaceView is positioned over the CameraX PreviewView with
 * transparent background, showing only the colored mask regions.
 */
class SegmentationRenderer(private val surfaceView: SurfaceView) : SurfaceHolder.Callback {

    companion object {
        private const val TAG = "SegmentationRenderer"

        // Default alpha for mask overlay (0.4 = 40% opacity)
        private const val DEFAULT_ALPHA = 0.4f

        // Maximum number of classes supported in color table
        private const val MAX_CLASSES = 256

        /**
         * Pre-defined color palette for segmentation classes.
         * Uses distinct, easily distinguishable colors.
         * Class 0 is typically background (transparent).
         *
         * Color format: ARGB where A is set by maskAlpha
         */
        private val CLASS_COLORS = intArrayOf(
            Color.TRANSPARENT,       // 0: Background - transparent
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

    // Current mask data (thread-safe reference)
    private val currentMask = AtomicReference<SegmentationMaskData?>(null)

    // Previous mask for temporal smoothing
    private var previousMask: Array<IntArray>? = null

    // Rendering state
    private val renderLock = ReentrantLock()

    // Configurable alpha for overlay (0.0 - 1.0)
    var maskAlpha: Float = DEFAULT_ALPHA
        set(value) {
            field = value.coerceIn(0f, 1f)
            updateColorTable()
        }

    // Enable/disable exponential moving average smoothing
    var enableSmoothing: Boolean = false

    // Smoothing factor for EMA (0 = use previous only, 1 = use current only)
    var smoothingFactor: Float = 0.7f

    // Pre-computed color table with alpha applied
    private var colorTableWithAlpha = IntArray(MAX_CLASSES)

    // Reusable bitmap for mask rendering
    private var maskBitmap: Bitmap? = null
    private var maskPixels: IntArray? = null

    // Paint for bitmap drawing
    private val bitmapPaint = Paint().apply {
        isAntiAlias = false
        isFilterBitmap = true  // Enable bilinear filtering for smooth scaling
    }

    /**
     * Data class for thread-safe mask transfer
     */
    data class SegmentationMaskData(
        val classMask: Array<IntArray>,
        val width: Int,
        val height: Int
    )

    init {
        surfaceHolder.addCallback(this)
        // Make surface transparent so camera shows through
        surfaceHolder.setFormat(android.graphics.PixelFormat.TRANSLUCENT)
        surfaceView.setZOrderOnTop(true)

        // Initialize color table with default alpha
        updateColorTable()
    }

    /**
     * Update the color lookup table with current alpha value.
     */
    private fun updateColorTable() {
        val alpha = (maskAlpha * 255).toInt()
        for (i in 0 until MAX_CLASSES) {
            val baseColor = if (i < CLASS_COLORS.size) {
                CLASS_COLORS[i]
            } else {
                // Generate colors for additional classes using hash
                generateColorForClass(i)
            }

            // Apply alpha to color (skip for transparent background class)
            colorTableWithAlpha[i] = if (i == 0) {
                Color.TRANSPARENT
            } else {
                Color.argb(alpha, Color.red(baseColor), Color.green(baseColor), Color.blue(baseColor))
            }
        }
    }

    /**
     * Generate a deterministic color for class IDs beyond the predefined palette.
     */
    private fun generateColorForClass(classId: Int): Int {
        // Use golden ratio for hue distribution
        val hue = (classId * 137.508f) % 360f
        val saturation = 0.7f + (classId % 3) * 0.1f
        val value = 0.8f + (classId % 2) * 0.1f

        return android.graphics.Color.HSVToColor(floatArrayOf(hue, saturation, value))
    }

    /**
     * Update the mask to be rendered.
     * Thread-safe - can be called from inference thread.
     *
     * @param result Segmentation result from model inference
     */
    fun updateMask(result: TFLiteSegmentationModel.SegmentationResult) {
        val newMaskData = SegmentationMaskData(
            classMask = result.classMask,
            width = result.width,
            height = result.height
        )

        currentMask.set(newMaskData)

        // Trigger render on surface
        if (surfaceReady.get()) {
            renderMask()
        }
    }

    /**
     * Render the current mask to the surface.
     * Called when new mask data is available or surface needs refresh.
     */
    fun renderMask() {
        if (!surfaceReady.get()) return

        renderLock.withLock {
            val maskData = currentMask.get()

            val canvas: Canvas? = try {
                surfaceHolder.lockCanvas()
            } catch (e: Exception) {
                Log.w(TAG, "Failed to lock canvas: ${e.message}")
                null
            }

            canvas?.let { c ->
                try {
                    // Clear canvas with transparent
                    c.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

                    if (maskData != null) {
                        drawMaskToCanvas(c, maskData)
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

    /**
     * Draw the segmentation mask to canvas with proper scaling.
     */
    private fun drawMaskToCanvas(canvas: Canvas, maskData: SegmentationMaskData) {
        val maskWidth = maskData.width
        val maskHeight = maskData.height
        val classMask = maskData.classMask

        // Apply temporal smoothing if enabled
        val effectiveMask = if (enableSmoothing && previousMask != null) {
            applyTemporalSmoothing(classMask, previousMask!!)
        } else {
            classMask
        }

        // Store for next frame smoothing
        previousMask = classMask

        // Ensure bitmap is allocated at correct size
        if (maskBitmap?.width != maskWidth || maskBitmap?.height != maskHeight) {
            maskBitmap?.recycle()
            maskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888)
            maskPixels = IntArray(maskWidth * maskHeight)
        }

        val pixels = maskPixels ?: return
        val bitmap = maskBitmap ?: return

        // Convert class mask to colored pixels
        var pixelIndex = 0
        for (y in 0 until maskHeight) {
            for (x in 0 until maskWidth) {
                val classId = effectiveMask[y][x].coerceIn(0, MAX_CLASSES - 1)
                pixels[pixelIndex++] = colorTableWithAlpha[classId]
            }
        }

        // Copy pixels to bitmap
        bitmap.setPixels(pixels, 0, maskWidth, 0, 0, maskWidth, maskHeight)

        // Calculate destination rect to fit surface (maintain aspect ratio and center)
        val destRect = calculateDestinationRect(maskWidth, maskHeight, surfaceWidth, surfaceHeight)

        // Draw bitmap scaled to surface
        canvas.drawBitmap(
            bitmap,
            Rect(0, 0, maskWidth, maskHeight),
            destRect,
            bitmapPaint
        )
    }

    /**
     * Apply exponential moving average smoothing between frames.
     * This reduces flickering when class predictions are unstable.
     *
     * Note: This is a simple per-pixel mode filter. For more sophisticated
     * temporal consistency, optical flow or mask tracking would be needed.
     */
    private fun applyTemporalSmoothing(
        currentMask: Array<IntArray>,
        previousMask: Array<IntArray>
    ): Array<IntArray> {
        val height = currentMask.size
        val width = if (height > 0) currentMask[0].size else 0

        // Simple smoothing: keep previous class if confidence is similar
        // This is a heuristic since we don't have per-pixel confidence here
        return if (smoothingFactor >= 1.0f) {
            currentMask
        } else {
            Array(height) { y ->
                IntArray(width) { x ->
                    if (y < previousMask.size && x < previousMask[y].size) {
                        // Use random threshold based on smoothing factor
                        // Higher smoothingFactor = more responsive to new predictions
                        if (Math.random() < smoothingFactor) {
                            currentMask[y][x]
                        } else {
                            previousMask[y][x]
                        }
                    } else {
                        currentMask[y][x]
                    }
                }
            }
        }
    }

    /**
     * Calculate destination rectangle for centered, aspect-ratio-preserving scaling.
     */
    private fun calculateDestinationRect(
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int
    ): Rect {
        val srcAspect = srcWidth.toFloat() / srcHeight
        val dstAspect = dstWidth.toFloat() / dstHeight

        val scaledWidth: Int
        val scaledHeight: Int

        if (srcAspect > dstAspect) {
            // Source is wider - fit to width
            scaledWidth = dstWidth
            scaledHeight = (dstWidth / srcAspect).toInt()
        } else {
            // Source is taller - fit to height
            scaledHeight = dstHeight
            scaledWidth = (dstHeight * srcAspect).toInt()
        }

        // Center in destination
        val left = (dstWidth - scaledWidth) / 2
        val top = (dstHeight - scaledHeight) / 2

        return Rect(left, top, left + scaledWidth, top + scaledHeight)
    }

    /**
     * Clear the overlay (show transparent).
     */
    fun clear() {
        currentMask.set(null)
        previousMask = null

        if (surfaceReady.get()) {
            renderLock.withLock {
                val canvas = try {
                    surfaceHolder.lockCanvas()
                } catch (e: Exception) {
                    null
                }

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

    // SurfaceHolder.Callback implementation

    override fun surfaceCreated(holder: SurfaceHolder) {
        Log.d(TAG, "Surface created")
        surfaceReady.set(true)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        Log.d(TAG, "Surface changed: ${width}x${height}")
        surfaceWidth = width
        surfaceHeight = height

        // Re-render current mask at new size
        if (currentMask.get() != null) {
            renderMask()
        }
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        Log.d(TAG, "Surface destroyed")
        surfaceReady.set(false)
    }

    /**
     * Set custom color for a specific class.
     *
     * @param classId The class ID (0 = background)
     * @param color RGB color (alpha will be applied based on maskAlpha)
     */
    fun setClassColor(classId: Int, color: Int) {
        if (classId in 0 until MAX_CLASSES) {
            val alpha = (maskAlpha * 255).toInt()
            colorTableWithAlpha[classId] = if (classId == 0) {
                Color.TRANSPARENT
            } else {
                Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color))
            }
        }
    }

    /**
     * Get current color for a class (with alpha applied).
     */
    fun getClassColor(classId: Int): Int {
        return if (classId in 0 until MAX_CLASSES) {
            colorTableWithAlpha[classId]
        } else {
            Color.TRANSPARENT
        }
    }
}
