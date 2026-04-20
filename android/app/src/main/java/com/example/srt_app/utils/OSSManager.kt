package com.example.srt_app.utils

import android.content.Context
import android.net.Uri
import android.util.Log
import com.alibaba.sdk.android.oss.ClientConfiguration
import com.alibaba.sdk.android.oss.OSSClient
import com.alibaba.sdk.android.oss.common.auth.OSSPlainTextAKSKCredentialProvider
import com.alibaba.sdk.android.oss.model.PutObjectRequest
import com.alibaba.sdk.android.oss.callback.OSSCompletedCallback
import com.alibaba.sdk.android.oss.model.PutObjectResult
import java.util.UUID

class OSSManager(private val context: Context) {

    // ======= 请务必确认以下两项与阿里云控制台完全一致 =======
    private val bucketName = "srt-app"
    private val endpoint = "oss-cn-beijing.aliyuncs.com"
    // =====================================================

    private val accessKeyId = ""
    private val accessKeySecret = ""

    private val oss: OSSClient by lazy {
        val credentialProvider = OSSPlainTextAKSKCredentialProvider(accessKeyId, accessKeySecret)

        // 配置网络参数，增加超时时间防止 Stream Closed
        val conf = ClientConfiguration()
        conf.connectionTimeout = 15 * 1000 // 连接超时
        conf.socketTimeout = 15 * 1000     // 读写超时
        conf.maxErrorRetry = 2             // 失败重试次数

        OSSClient(context, "https://$endpoint", credentialProvider, conf)
    }

    /**
     * 使用字节数组上传，更稳定，防止文件流关闭错误
     */
    fun uploadImage(uri: Uri, onSuccess: (String) -> Unit, onFailure: (Exception) -> Unit) {
        val fileName = "avatars/${UUID.randomUUID()}.jpg"
        Log.d("OSS_DEBUG", ">>> Start Upload (Bytes Mode): $fileName")

        try {
            // 1. 直接读取所有字节到内存
            val inputStream = context.contentResolver.openInputStream(uri)
            val data = inputStream?.readBytes()
            inputStream?.close()

            if (data == null || data.isEmpty()) {
                onFailure(Exception("Selected image data is empty"))
                return
            }

            Log.d("OSS_DEBUG", ">>> Image loaded into memory: ${data.size} bytes")

            // 2. 创建 PutObjectRequest，直接传入字节数组
            val put = PutObjectRequest(bucketName, fileName, data)

            // 3. 执行异步上传
            oss.asyncPutObject(put, object : OSSCompletedCallback<PutObjectRequest, PutObjectResult> {
                override fun onSuccess(request: PutObjectRequest?, result: PutObjectResult?) {
                    val url = "https://$bucketName.$endpoint/$fileName"
                    Log.i("OSS_DEBUG", ">>> UPLOAD SUCCESS: $url")
                    onSuccess(url)
                }

                override fun onFailure(request: PutObjectRequest?, clientEx: com.alibaba.sdk.android.oss.ClientException?, serviceEx: com.alibaba.sdk.android.oss.ServiceException?) {
                    val error = clientEx ?: serviceEx ?: Exception("Unknown OSS Error")
                    Log.e("OSS_DEBUG", ">>> UPLOAD FAILED: ${error.message}")
                    onFailure(error)
                }
            })
        } catch (e: Exception) {
            Log.e("OSS_DEBUG", ">>> EXCEPTION: ${e.message}")
            onFailure(e)
        }
    }
}
