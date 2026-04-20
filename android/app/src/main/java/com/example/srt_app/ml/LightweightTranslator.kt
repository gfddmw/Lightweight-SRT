package com.example.srt_app.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * 负责 PyTorch Mobile 轻量级模型 (ST-GCN) 的加载与端侧推理。
 */
class LightweightTranslator(private val context: Context) {

    private var module: Module? = null
    private var classLabels: List<String> = emptyList()

    private val MODEL_NAME = "st_gcn_student.ptl"
    private val LABEL_NAME = "wlasl_class_list.txt"

    /**
     * 初始化模型加载与配置
     */
    fun initialize() {
        try {
            // 1. 加载 PyTorch Lite 模型
            val modelPath = assetFilePath(context, MODEL_NAME)
            module = LiteModuleLoader.load(modelPath)
            Log.d("LightweightTranslator", "Model loaded successfully: $MODEL_NAME")

            // 2. 加载类别标签
            loadLabels()
        } catch (e: Exception) {
            Log.e("LightweightTranslator", "Error loading model or labels", e)
        }
    }

    private fun loadLabels() {
        try {
            val content = context.assets.open(LABEL_NAME).bufferedReader().use { it.readText() }
            classLabels = content.lines()
                .filter { it.isNotBlank() }
                .map { line ->
                    // 标签格式通常为 "index\tlabel" 或 "index label"
                    val parts = line.split(Regex("\\s+"), 2)
                    if (parts.size > 1) parts[1] else line
                }
            Log.d("LightweightTranslator", "Labels loaded: ${classLabels.size} classes")
        } catch (e: IOException) {
            Log.e("LightweightTranslator", "Error loading labels", e)
        }
    }

    data class TranslationResult(val text: String, val confidence: Float)

    /**
     * 接收预处理后的连续帧特征进行手语推理。
     * 输入格式要求：(1, 3, 64, 42, 1) -> (Batch, Channels, Time, Joints, Person)
     */
    suspend fun translate(framesFeature: FloatArray): TranslationResult {
        return withContext(Dispatchers.Default) {
            if (module == null) {
                Log.e("LightweightTranslator", "Inference failed: Module not initialized")
                return@withContext TranslationResult("", 0.0f)
            }

            try {
                val startTime = System.currentTimeMillis()
                
                // 1. 输入数据校验
                val expectedSize = 1 * 3 * 64 * 42 * 1
                if (framesFeature.size != expectedSize) {
                    Log.e("LightweightTranslator", "Input size mismatch! Expected $expectedSize, got ${framesFeature.size}")
                    return@withContext TranslationResult("", 0.0f)
                }

                // 2. 将输入的特征转换为 Tensor
                // 输入形状调整为 ST-GCN 标准格式: [1, 3, 64, 21, 2]
                // N=Batch, C=XYZ, T=Frames, V=Joints, M=Hands
                val inputTensor = Tensor.fromBlob(framesFeature, longArrayOf(1, 3, 64, 21, 2))
                Log.d("LightweightTranslator", ">>> [INFERENCE] Start: Tensor shape (1, 3, 64, 21, 2)")

                // 3. 执行推理
                val outputValue = module!!.forward(IValue.from(inputTensor))
                val outputTensor = outputValue.toTensor()
                val scores = outputTensor.dataAsFloatArray
                
                val inferenceTime = System.currentTimeMillis() - startTime
                Log.d("LightweightTranslator", ">>> [INFERENCE] End: Cost ${inferenceTime}ms, Output classes: ${scores.size}")

                // 4. 解析输出 (找到 Top-3 置信度用于调试)
                val topResults = scores.withIndex()
                    .sortedByDescending { it.value }
                    .take(3)
                
                for (res in topResults) {
                    val label = if (res.index < classLabels.size) classLabels[res.index] else "Unknown(${res.index})"
                    Log.d("LightweightTranslator", "Top Result: [$label] Index: ${res.index}, Score: ${res.value}")
                }

                val bestResult = topResults.first()
                if (bestResult.index < classLabels.size) {
                    val label = classLabels[bestResult.index]
                    TranslationResult(label, bestResult.value)
                } else {
                    Log.w("LightweightTranslator", "Best index ${bestResult.index} out of labels bounds (${classLabels.size})")
                    TranslationResult("", 0.0f)
                }
            } catch (e: Exception) {
                Log.e("LightweightTranslator", "Inference error", e)
                TranslationResult("", 0.0f)
            }
        }
    }

    /**
     * 释放模型资源
     */
    fun close() {
        module = null
    }

    /**
     * 辅助函数：将 Assets 中的模型文件复制到内部存储，以便 PyTorch 加载
     */
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        try {
            context.assets.open(assetName).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (`is`.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        } catch (e: IOException) {
            Log.e("LightweightTranslator", "Error copying asset $assetName", e)
        }
        return ""
    }
}
