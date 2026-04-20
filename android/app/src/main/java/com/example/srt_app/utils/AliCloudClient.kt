package com.example.srt_app.utils

import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

/**
 * 阿里云专用网络客户端 - 强制日志监控版
 */
class AliCloudClient(
    private var baseUrl: String = "https://526bf211ee5c45b6b67162a3782c157b-cn-hangzhou.alicloudapi.com",
    private var accessToken: String = ""
) {
    private val gson = Gson()
    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    fun updateToken(token: String) {
        this.accessToken = token
    }

    suspend fun <T> request(
        method: String,
        path: String,
        body: Any? = null,
        responseType: Class<T>
    ): T? = withContext(Dispatchers.IO) {
        val url = if (path.startsWith("http")) path else "$baseUrl$path"
        
        // 【核心监控】强制打印请求详情
        Log.e("!!!_DEBUG_!!!", ">>> [NETWORK_START] $method to $url")
        Log.e("!!!_DEBUG_!!!", ">>> [PAYLOAD] ${gson.toJson(body)}")

        val requestBuilder = Request.Builder()
            .url(url)
            .addHeader("Accept", "application/json") // 明确告诉网关我们需要 JSON
            .addHeader("Content-Type", "application/json") 
            .apply {
                if (accessToken.isNotEmpty()) {
                    addHeader("Authorization", "Bearer $accessToken")
                }
            }

        // 强制使用不含 charset 的原始 JSON 字符串
        val jsonString = if (body != null) gson.toJson(body) else "{}"
        val mediaType = "application/json".toMediaType()
        val jsonBody = jsonString.toRequestBody(mediaType)

        when (method.uppercase()) {
            "POST" -> requestBuilder.post(jsonBody)
            "PUT" -> requestBuilder.put(jsonBody)
            "DELETE" -> requestBuilder.delete(jsonBody)
            else -> requestBuilder.get()
        }

        return@withContext try {
            val response = client.newCall(requestBuilder.build()).execute()
            val responseBody = response.body?.string()

            // 【核心监控】强制打印响应详情
            Log.e("!!!_DEBUG_!!!", ">>> [NETWORK_RECEIVE] Code: ${response.code}")
            Log.e("!!!_DEBUG_!!!", ">>> [RESPONSE_BODY] $responseBody")

            if (response.isSuccessful && responseBody != null) {
                gson.fromJson(responseBody, responseType)
            } else {
                Log.e("!!!_DEBUG_!!!", ">>> [NETWORK_ERROR] HTTP ${response.code}")
                null
            }
        } catch (e: Exception) {
            // 【核心监控】强制打印异常详情
            Log.e("!!!_DEBUG_!!!", ">>> [NETWORK_EXCEPTION] ${e.message}")
            e.printStackTrace()
            null
        }
    }
}
