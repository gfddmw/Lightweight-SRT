package com.example.srt_app.data

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.JsonObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.UUID

import java.util.concurrent.TimeUnit

class UserRepository(
    private val context: Context, 
    private val userDao: UserDao,
    private val settingsManager: com.example.srt_app.utils.SettingsManager
) {
    private val envId = "srt-app-3gw2nml5a0b41a6f"
    private val baseUrl = "https://$envId.api.tcloudbasegateway.com/auth/v1"

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .writeTimeout(20, TimeUnit.SECONDS)
        .addInterceptor { chain ->
            val original = chain.request()
            val requestBuilder = original.newBuilder()
                .addHeader("x-device-id", deviceId)
                .addHeader("X-CloudBase-Env", envId)

            // 从 SettingsManager 中尝试同步读取 Token (由于 DataStore 是异步的，这里使用 runBlocking 获取)
            // 提示：拦截器运行在 OkHttp 的子线程中，runBlocking 在这里是安全的
            val token = kotlinx.coroutines.runBlocking {
                settingsManager.getAccessTokenSync()
            }

            if (token.isNotEmpty()) {
                requestBuilder.header("Authorization", "Bearer $token")
                Log.d("UserRepository", "Interceptor: Adding Authorization header")
            }

            chain.proceed(requestBuilder.build())
        }
        .build()
    
    private val gson = Gson()
    
    private val deviceId: String by lazy {
        val prefs = context.getSharedPreferences("auth_prefs", Context.MODE_PRIVATE)
        var id = prefs.getString("device_id", null)
        if (id == null) {
            id = "android_device_${UUID.randomUUID().toString().take(8)}"
            prefs.edit().putString("device_id", id).apply()
        }
        id
    }

    data class AuthResult(
        val user: User,
        val accessToken: String,
        val refreshToken: String
    )

    /**
     * 1. 内部辅助方法：验证验证码并获取 verification_token
     */
    private suspend fun verifyCode(verificationId: String, verificationCode: String): Result<String> = withContext(Dispatchers.IO) {
        try {
            Log.d("UserRepository", "Verifying code: $verificationCode for id: $verificationId")
            val json = JsonObject().apply {
                addProperty("verification_id", verificationId)
                addProperty("verification_code", verificationCode)
            }
            
            val requestBody = json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
            val request = Request.Builder()
                .url("$baseUrl/verification/verify")
                .post(requestBody)
                .addHeader("x-device-id", deviceId)
                .addHeader("X-CloudBase-Env", envId)
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""
            Log.d("UserRepository", "Verify Response (${response.code}): $responseBody")

            if (response.isSuccessful) {
                val jsonResponse = gson.fromJson(responseBody, JsonObject::class.java)
                val token = jsonResponse.get("verification_token")?.asString ?: ""
                if (token.isNotEmpty()) Result.success(token) else Result.failure(Exception("No verification_token in response"))
            } else {
                val jsonResponse = try { gson.fromJson(responseBody, JsonObject::class.java) } catch (e: Exception) { null }
                val message = jsonResponse?.get("error_description")?.asString 
                           ?: jsonResponse?.get("message")?.asString 
                           ?: "Verification failed (${response.code})"
                Result.failure(Exception(message))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * 2. 注册 (逻辑：先 verify 获取 token，再 signup)
     */
    suspend fun register(user: User, verificationCode: String, verificationId: String): Result<AuthResult> = withContext(Dispatchers.IO) {
        try {
            // 第一步：验证验证码
            val verifyResult = verifyCode(verificationId, verificationCode)
            if (verifyResult.isFailure) return@withContext Result.failure(verifyResult.exceptionOrNull()!!)
            
            val verificationToken = verifyResult.getOrThrow()

            // 第二步：正式注册
            Log.d("UserRepository", "Finalizing signup for: ${user.email} with token: $verificationToken")
            val json = JsonObject().apply {
                addProperty("username", user.username) 
                addProperty("email", user.email)
                addProperty("password", user.password)
                addProperty("verification_token", verificationToken)
            }
            
            val requestBody = json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
            val request = Request.Builder()
                .url("$baseUrl/signup")
                .post(requestBody)
                .addHeader("x-device-id", deviceId)
                .addHeader("X-CloudBase-Env", envId)
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""
            Log.d("UserRepository", "Signup Response (${response.code}): $responseBody")
            
            val jsonResponse = try { gson.fromJson(responseBody, JsonObject::class.java) } catch (e: Exception) { null }
            val data = if (jsonResponse?.has("data") == true) {
                val dataElement = jsonResponse.get("data")
                if (dataElement.isJsonObject) dataElement.asJsonObject else null
            } else jsonResponse

            if (response.isSuccessful && data != null && (data.has("uid") || data.has("access_token"))) {
                val accessToken = data.get("access_token")?.asString ?: ""
                val refreshToken = data.get("refresh_token")?.asString ?: ""
                Result.success(AuthResult(user, accessToken, refreshToken))
            } else {
                val message = jsonResponse?.get("message")?.asString ?: "Registration failed (${response.code})"
                Result.failure(Exception(message))
            }
        } catch (e: Exception) {
            Log.e("UserRepository", "Register Error", e)
            Result.failure(e)
        }
    }

    /**
     * 3. 用户名密码登录
     */
    suspend fun login(email: String, password: String): Result<AuthResult> = withContext(Dispatchers.IO) {
        try {
            Log.d("UserRepository", "Logging in: $email")
            val json = JsonObject().apply {
                addProperty("username", email)
                addProperty("password", password)
            }
            
            val requestBody = json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
            val request = Request.Builder()
                .url("$baseUrl/signin")
                .post(requestBody)
                .addHeader("x-device-id", deviceId)
                .addHeader("X-CloudBase-Env", envId)
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""
            Log.d("UserRepository", "Login Response (${response.code}): $responseBody")
            val jsonResponse = try { gson.fromJson(responseBody, JsonObject::class.java) } catch (e: Exception) { null }
            val data = if (jsonResponse?.has("data") == true) {
                val dataElement = jsonResponse.get("data")
                if (dataElement.isJsonObject) dataElement.asJsonObject else null
            } else jsonResponse

            if (response.isSuccessful && data?.has("access_token") == true) {
                val accessToken = data.get("access_token").asString
                val refreshToken = data.get("refresh_token").asString
                // 不再硬编码 "User"，使用邮箱名作为临时展示名
                val tempName = email.substringBefore("@")
                val user = User(id = 0, username = tempName, email = email, password = "")
                Result.success(AuthResult(user, accessToken, refreshToken))
            } else {
                val message = jsonResponse?.get("message")?.asString ?: "Login failed (${response.code})"
                Result.failure(Exception(message))
            }
        } catch (e: Exception) {
            Log.e("UserRepository", "Login Error", e)
            Result.failure(e)
        }
    }

    /**
     * 4. 发送邮箱验证码 (返回 verification_id)
     */
    suspend fun sendVerificationCode(email: String): Result<String> = withContext(Dispatchers.IO) {
        try {
            Log.d("UserRepository", "Sending verification code to: $email")
            val json = JsonObject().apply {
                addProperty("email", email)
                addProperty("target", "ANY") // 'ANY' matches most use cases (login or registration)
            }
            
            val requestBody = json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
            val request = Request.Builder()
                .url("$baseUrl/verification") 
                .post(requestBody)
                .addHeader("x-device-id", deviceId)
                .addHeader("X-CloudBase-Env", envId)
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""
            Log.d("UserRepository", "SendCode Response (${response.code}): $responseBody")

            if (response.isSuccessful) {
                val jsonResponse = gson.fromJson(responseBody, JsonObject::class.java)
                val dataElement = if (jsonResponse.has("data")) jsonResponse.get("data") else jsonResponse
                
                val id = when {
                    dataElement.isJsonObject -> {
                        val dataObj = dataElement.asJsonObject
                        dataObj.get("verification_id")?.asString 
                            ?: dataObj.get("verificationId")?.asString
                            ?: ""
                    }
                    else -> ""
                }
                
                if (id.isNotEmpty()) Result.success(id) else Result.failure(Exception("No verification_id in response"))
            } else {
                val jsonResponse = try { gson.fromJson(responseBody, JsonObject::class.java) } catch (e: Exception) { null }
                val message = jsonResponse?.get("message")?.asString ?: "Failed to send code (${response.code})"
                Result.failure(Exception(message))
            }
        } catch (e: Exception) {
            Log.e("UserRepository", "SendCode Error", e)
            Result.failure(e)
        }
    }

    /**
     * 5. 邮箱验证码登录 (逻辑：先 verify 再 signin)
     */
    suspend fun loginWithCode(email: String, code: String, verificationId: String): Result<AuthResult> = withContext(Dispatchers.IO) {
        try {
            // 第一步：验证验证码
            val verifyResult = verifyCode(verificationId, code)
            if (verifyResult.isFailure) return@withContext Result.failure(verifyResult.exceptionOrNull()!!)
            
            val verificationToken = verifyResult.getOrThrow()

            // 第二步：使用 token 登录
            Log.d("UserRepository", "Finalizing signin-with-code with token: $verificationToken")
            val json = JsonObject().apply {
                addProperty("email", email)
                addProperty("verification_token", verificationToken)
            }
            
            val requestBody = json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
            val request = Request.Builder()
                .url("$baseUrl/signin") // 通常也是 signin 接口
                .post(requestBody)
                .addHeader("x-device-id", deviceId)
                .addHeader("X-CloudBase-Env", envId)
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""
            Log.d("UserRepository", "SigninWithToken Response (${response.code}): $responseBody")
            val jsonResponse = try { gson.fromJson(responseBody, JsonObject::class.java) } catch (e: Exception) { null }
            val data = if (jsonResponse?.has("data") == true) {
                val dataElement = jsonResponse.get("data")
                if (dataElement.isJsonObject) dataElement.asJsonObject else null
            } else jsonResponse

            if (response.isSuccessful && data?.has("access_token") == true) {
                val accessToken = data.get("access_token").asString
                val refreshToken = data.get("refresh_token").asString
                // 不再硬编码 "User"，使用邮箱名作为临时展示名
                val tempName = email.substringBefore("@")
                val user = User(id = 0, username = tempName, email = email, password = "")
                Result.success(AuthResult(user, accessToken, refreshToken))
            } else {
                val message = jsonResponse?.get("message")?.asString ?: "Login failed (${response.code})"
                Result.failure(Exception(message))
            }
        } catch (e: Exception) {
            Log.e("UserRepository", "LoginWithCode Error", e)
            Result.failure(e)
        }
    }

    /**
     * 6. 更新用户信息 (同步到云端)
     */
    suspend fun updateProfile(accessToken: String, nickname: String, description: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            Log.d("UserRepository", "Updating profile: nickname=$nickname, description=$description")
            val json = JsonObject().apply {
                addProperty("nickname", nickname)
                addProperty("description", description)
            }
            
            val requestBody = json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
            val request = Request.Builder()
                .url("$baseUrl/user/basic/edit")
                .post(requestBody)
                .addHeader("Authorization", "Bearer $accessToken")
                .addHeader("x-device-id", deviceId)
                .addHeader("X-CloudBase-Env", envId)
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""
            Log.d("UserRepository", "UpdateProfile Response (${response.code}): $responseBody")

            if (response.isSuccessful) {
                Result.success(Unit)
            } else {
                val jsonResponse = try { gson.fromJson(responseBody, JsonObject::class.java) } catch (e: Exception) { null }
                val message = jsonResponse?.get("message")?.asString ?: "Profile update failed (${response.code})"
                Result.failure(Exception(message))
            }
        } catch (e: Exception) {
            Log.e("UserRepository", "UpdateProfile Error", e)
            Result.failure(e)
        }
    }

    /**
     * 7. 获取用户信息 (从云端拉取)
     */
    suspend fun getUserProfile(accessToken: String): Result<JsonObject> = withContext(Dispatchers.IO) {
        try {
            Log.d("UserRepository", "Fetching user profile from cloud (POST): $baseUrl/user/info")
            // 某些环境下 GET 会返回 501，尝试使用 POST (即使没有 body)
            val emptyBody = "{}".toRequestBody("application/json; charset=utf-8".toMediaType())
            val request = Request.Builder()
                .url("$baseUrl/user/info")
                .post(emptyBody)
                .addHeader("Authorization", "Bearer $accessToken")
                .addHeader("x-device-id", deviceId)
                .addHeader("X-CloudBase-Env", envId)
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""
            Log.d("UserRepository", "GetUserProfile Response (${response.code}): $responseBody")

            if (response.isSuccessful) {
                val jsonResponse = gson.fromJson(responseBody, JsonObject::class.java)
                val data = if (jsonResponse.has("data") && jsonResponse.get("data").isJsonObject) {
                    jsonResponse.getAsJsonObject("data")
                } else {
                    jsonResponse
                }
                Result.success(data)
            } else {
                val jsonResponse = try { gson.fromJson(responseBody, JsonObject::class.java) } catch (e: Exception) { null }
                val message = jsonResponse?.get("message")?.asString ?: "Failed to fetch profile (${response.code})"
                Result.failure(Exception(message))
            }
        } catch (e: Exception) {
            Log.e("UserRepository", "GetUserProfile Error", e)
            Result.failure(e)
        }
    }

    fun logout() {}
    fun getCurrentUser(): User? = null
}
