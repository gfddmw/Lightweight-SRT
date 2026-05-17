package com.example.srt_app.ml

import android.graphics.PointF
import android.util.Log
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import java.util.*
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 核心识别引擎：负责特征提取、缓冲管理、滑动窗口、VAD 及缺失帧补偿。
 */
class SignRecognitionEngine(
    private val translator: LightweightTranslator,
    private val scope: CoroutineScope,
    private val windowSize: Int = 64,
    private val stride: Int = 16,
    private val maxMissingFrames: Int = 5
) {
    private val TAG = "SignRecognitionEngine"

    // 特征维度: 21(点) * 3(XYZ) * 2(手) = 126
    private val featureDimension = 126
    private val frameBuffer = Collections.synchronizedList(mutableListOf<FloatArray>())
    
    // 状态流
    private val _translationFlow = MutableSharedFlow<String>()
    val translationFlow = _translationFlow.asSharedFlow()

    private val _isProcessing = AtomicBoolean(false)
    
    // VAD 与稳定性相关
    private var lastWristPosition: PointF? = null
    private var missingFrameCount = 0
    private var lastValidFeatures: FloatArray? = null
    private var smoothedFeatures: FloatArray? = null
    private val smoothingFactor = 0.4f
    
    // 滑动窗口计数
    private var framesSinceLastInference = 0

    /**
     * 处理 MediaPipe 返回的结果
     */
    fun onHandResults(result: HandLandmarkerResult?, isFrontCamera: Boolean) {
        scope.launch(Dispatchers.Default) {
            val currentFeatures = if (result != null && result.landmarks().isNotEmpty()) {
                // 1. 成功检测到手
                missingFrameCount = 0
                val extracted = extractFeatures(result, isFrontCamera)
                val smoothed = performSmoothing(extracted)
                lastValidFeatures = smoothed
                updateVAD(result)
                smoothed
            } else {
                // 2. 缺失帧补偿逻辑
                handleMissingFrame()
            }

            if (currentFeatures != null) {
                addToBuffer(currentFeatures)
            } else {
                resetBuffer()
            }
        }
    }

    private fun handleMissingFrame(): FloatArray? {
        return if (missingFrameCount < maxMissingFrames && lastValidFeatures != null) {
            missingFrameCount++
            Log.d(TAG, ">>> [STABILITY] Missing frame compensated ($missingFrameCount/$maxMissingFrames)")
            lastValidFeatures // 复用最后一帧有效特征
        } else {
            null // 彻底丢失追踪
        }
    }

    private fun addToBuffer(features: FloatArray) {
        synchronized(frameBuffer) {
            frameBuffer.add(features)
            if (frameBuffer.size > windowSize) {
                frameBuffer.removeAt(0)
            }
        }

        framesSinceLastInference++

        // 3. 滑动窗口触发逻辑
        // 只有缓冲区填满，且达到步长阈值时才触发
        if (frameBuffer.size == windowSize && framesSinceLastInference >= stride) {
            if (!_isProcessing.get()) {
                framesSinceLastInference = 0
                triggerInference()
            }
        }
    }

    private fun triggerInference() {
        val input = prepareInputAsync()
        scope.launch(Dispatchers.Default) {
            if (_isProcessing.compareAndSet(false, true)) {
                try {
                    Log.d(TAG, ">>> [ENGINE] Starting Inference...")
                    val result = translator.translate(input)
                    Log.d(TAG, ">>> [ENGINE] Inference finished. Result: ${result.text}, Confidence: ${result.confidence}")
                    
                    if (result.text.isNotEmpty() && result.confidence > 0.3f) { // 放宽到 0.3
                        _translationFlow.emit(result.text)
                    } else if (result.text.isNotEmpty()) {
                        Log.w(TAG, "Result '${result.text}' rejected due to low confidence: ${result.confidence}")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Inference error", e)
                } finally {
                    _isProcessing.set(false)
                }
            } else {
                Log.w(TAG, "Inference skipped: already processing")
            }
        }
    }

    /**
     * 优化点 1.2: 异步非阻塞式特征准备
     */
    private fun prepareInputAsync(): FloatArray {
        return synchronized(frameBuffer) {
            val reshaped = FloatArray(3 * windowSize * 21 * 2)
            // 采用更高效的内存布局填充，匹配 [1, 3, 64, 21, 2]
            for (c in 0 until 3) {
                val channelOffset = c * (windowSize * 21 * 2)
                for (t in 0 until windowSize) {
                    val timeOffset = t * (21 * 2)
                    val frameFeatures = frameBuffer[t]
                    for (v in 0 until 21) {
                        val jointOffset = v * 2
                        // 左手/右手特征在提取时已分配到 features[0..62] 和 features[63..125]
                        reshaped[channelOffset + timeOffset + jointOffset] = frameFeatures[v * 3 + c] // Hand 0
                        reshaped[channelOffset + timeOffset + jointOffset + 1] = frameFeatures[63 + v * 3 + c] // Hand 1
                    }
                }
            }
            reshaped
        }
    }

    private fun extractFeatures(result: HandLandmarkerResult, isFrontCamera: Boolean): FloatArray {
        val features = FloatArray(featureDimension)
        val allLandmarks = result.landmarks()
        val allHandedness = result.handedness()

        for (i in allLandmarks.indices) {
            if (i >= 2) break // 只处理两只手
            
            val hand = allLandmarks[i]
            val category = allHandedness[i][0].categoryName()
            
            // 镜像逻辑补偿
            val isRightHand = if (isFrontCamera) category == "Left" else category == "Right"
            val offset = if (isRightHand) 0 else 63

            for (j in 0 until 21) {
                features[offset + j * 3] = hand[j].x()
                features[offset + j * 3 + 1] = hand[j].y()
                features[offset + j * 3 + 2] = hand[j].z()
            }
        }
        return features
    }

    private fun performSmoothing(current: FloatArray): FloatArray {
        val prev = smoothedFeatures ?: return current.also { smoothedFeatures = it }
        val next = FloatArray(featureDimension)
        for (i in 0 until featureDimension) {
            next[i] = current[i] * smoothingFactor + prev[i] * (1f - smoothingFactor)
        }
        smoothedFeatures = next
        return next
    }

    private fun updateVAD(result: HandLandmarkerResult) {
        val wrist = result.landmarks()[0][0]
        val currentWristPos = PointF(wrist.x(), wrist.y())
        lastWristPosition = currentWristPos
        // 此处可扩展更复杂的静默检测逻辑
    }

    private fun resetBuffer() {
        synchronized(frameBuffer) {
            frameBuffer.clear()
        }
        framesSinceLastInference = 0
        lastValidFeatures = null
        smoothedFeatures = null
        Log.d(TAG, ">>> [STABILITY] Buffer reset due to tracking loss")
    }
}
