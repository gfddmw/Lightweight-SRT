package com.example.srt_app.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.srt_app.ml.HandLandmarkerHelper

/**
 * 负责视频帧获取与缓存的图像分析器。
 * 绑定到 CameraX 的 ImageAnalysis 用例中。
 */
class SignFrameAnalyzer(
    private val handLandmarkerHelper: HandLandmarkerHelper?,
    private val isFrontCamera: () -> Boolean,
    private val isExternalProcessing: () -> Boolean
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        // 如果外部正在忙，或者识别器未就绪，则丢弃当前帧
        if (isExternalProcessing() || handLandmarkerHelper == null) {
            image.close()
            return
        }

        // 使用 MediaPipe 进行手部关键点检测
        handLandmarkerHelper.detectLiveStream(
            imageProxy = image,
            isFrontCamera = isFrontCamera()
        )
    }
}
