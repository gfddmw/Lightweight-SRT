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
            classLabels = content.lines().filter { it.isNotBlank() }
            Log.d("LightweightTranslator", "Labels loaded: ${classLabels.size} classes")
        } catch (e: IOException) {
            Log.e("LightweightTranslator", "Error loading labels", e)
        }
    }

    /**
     * 接收预处理后的连续帧特征进行手语推理。
     * 输入格式要求：(1, 3, 64, 21, 1) -> (Batch, Channels, Time, Joints, Person)
     * 这里的 framesFeature 应该是按照这个顺序排列好的 FloatArray
     */
    suspend fun translate(framesFeature: FloatArray): String {
        return withContext(Dispatchers.Default) {
            if (module == null) return@withContext ""

            try {
                // 1. 将输入的特征转换为 Tensor
                // 输入形状: [1, 3, 64, 21, 1]
                val inputTensor = Tensor.fromBlob(framesFeature, longArrayOf(1, 3, 64, 21, 1))

                // 2. 执行推理
                val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()
                val scores = outputTensor.dataAsFloatArray

                // 3. 解析输出 (ArgMax)
                var maxScore = -1.0f
                var maxIdx = -1
                for (i in scores.indices) {
                    if (scores[i] > maxScore) {
                        maxScore = scores[i]
                        maxIdx = i
                    }
                }

                if (maxIdx != -1 && maxIdx < classLabels.size) {
                    val result = classLabels[maxIdx]
                    Log.d("LightweightTranslator", "Prediction: $result (score: $maxScore)")
                    result
                } else {
                    ""
                }
            } catch (e: Exception) {
                Log.e("LightweightTranslator", "Inference error", e)
                ""
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
