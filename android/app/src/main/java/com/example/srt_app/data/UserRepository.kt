package com.example.srt_app.data

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import com.example.srt_app.utils.TokenManager
import com.example.srt_app.utils.SettingsManager
import com.example.srt_app.utils.AliCloudClient
import com.google.gson.annotations.SerializedName

class UserRepository(
    private val context: Context, 
    private val userDao: UserDao,
    private val settingsManager: SettingsManager,
    private val tokenManager: TokenManager
) {
    private val aliClient = AliCloudClient()

    private val _userProfile = MutableStateFlow<UserProfile>(UserProfile())
    val userProfile: StateFlow<UserProfile> = _userProfile.asStateFlow()

    init {
        Log.i("SRT_DEBUG", "UserRepository: Switching to AliCloud Backend.")
    }

    /**
     * 实现阿里云登录逻辑
     */
    suspend fun login(email: String, password: String): Result<AuthData> = withContext(Dispatchers.IO) {
        try {
            val loginData = mapOf("username" to email, "password" to password)
            val response = aliClient.request(
                method = "POST",
                path = "/login",
                body = loginData,
                responseType = AuthData::class.java
            )

            if (response != null) {
                tokenManager.saveTokens(response.accessToken, response.refreshToken)
                aliClient.updateToken(response.accessToken)
                
                // 保存登录状态及用户名，以便下次同步
                settingsManager.setLoggedIn(true, response.user.username)
                
                // 立即去云端拉取最完整的 Profile (包含头像)
                fetchUserProfile(response.user.username)
                
                Result.success(response)
            } else {
                Result.failure(Exception("Login failed on AliCloud"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * 从云端拉取最新的用户信息
     */
    suspend fun fetchUserProfile(username: String = ""): Result<UserProfile> = withContext(Dispatchers.IO) {
        try {
            // 确定目标用户名
            var targetUsername = username
            if (targetUsername.isEmpty()) {
                // 如果参数为空，先尝试从 DataStore 读取已保存的用户名
                val prefs = settingsManager.settingsFlow.first()
                targetUsername = prefs.savedUsername
            }
            
            if (targetUsername.isEmpty() || targetUsername == "Guest") {
                return@withContext Result.failure(Exception("No user logged in"))
            }

            Log.d("SRT_DEBUG", ">>> Fetching Profile for: $targetUsername")

            val response = aliClient.request(
                method = "GET",
                path = "/get_profile?username=$targetUsername",
                responseType = Map::class.java
            )

            if (response != null && (response["status"] == "success" || response["statusCode"] == 200.0)) {
                // 处理复杂的 Map 类型转换
                val profileMap = response["profile"] as? Map<*, *>
                val newProfile = UserProfile(
                    nickname = profileMap?.get("nickname") as? String ?: targetUsername,
                    description = profileMap?.get("description") as? String ?: "Offline User",
                    avatarUrl = profileMap?.get("avatarUrl") as? String ?: ""
                )
                
                Log.i("SRT_DEBUG", ">>> PROFILE FETCHED: Nickname=${newProfile.nickname}, Avatar=${newProfile.avatarUrl}")
                
                _userProfile.value = newProfile
                Result.success(newProfile)
            } else {
                Result.failure(Exception("Profile not found on cloud"))
            }
        } catch (e: Exception) {
            Log.e("SRT_DEBUG", ">>> Fetch Profile Error: ${e.message}")
            Result.failure(e)
        }
    }

    suspend fun performStartupSync() {
        val token = tokenManager.getAccessToken()
        if (token.isNotEmpty()) {
            aliClient.updateToken(token)
            // 冷启动时同步一次云端资料
            fetchUserProfile()
        }
    }

    /**
     * 发送验证码 (阿里云适配)
     */
    suspend fun sendVerificationCode(email: String): Result<String> = withContext(Dispatchers.IO) {
        try {
            val response = aliClient.request(
                method = "POST",
                path = "/send_code",
                body = mapOf("email" to email),
                responseType = Map::class.java
            )
            val id = response?.get("verification_id") as? String ?: ""
            if (id.isNotEmpty()) Result.success(id) else Result.failure(Exception("Send failed"))
        } catch (e: Exception) { Result.failure(e) }
    }

    /**
     * 验证码登录 (阿里云适配)
     */
    suspend fun loginWithCode(email: String, code: String, id: String): Result<AuthData> = withContext(Dispatchers.IO) {
        try {
            val response = aliClient.request(
                method = "POST",
                path = "/login_with_code",
                body = mapOf("email" to email, "code" to code, "verification_id" to id),
                responseType = AuthData::class.java
            )
            if (response != null) {
                tokenManager.saveTokens(response.accessToken, response.refreshToken)
                settingsManager.setLoggedIn(true, response.user.username)
                fetchUserProfile(response.user.username)
                Result.success(response)
            } else Result.failure(Exception("Login failed"))
        } catch (e: Exception) { Result.failure(e) }
    }

    /**
     * 注册 (阿里云适配)
     */
    suspend fun register(user: User, code: String, id: String): Result<AuthData> = withContext(Dispatchers.IO) {
        try {
            val response = aliClient.request(
                method = "POST",
                path = "/register",
                body = mapOf(
                    "username" to user.username,
                    "password" to user.password,
                    "email" to user.email,
                    "code" to code,
                    "verification_id" to id
                ),
                responseType = AuthData::class.java
            )
            if (response != null) {
                tokenManager.saveTokens(response.accessToken, response.refreshToken)
                settingsManager.setLoggedIn(true, response.user.username)
                _userProfile.value = UserProfile(nickname = response.user.username)
                Result.success(response)
            } else Result.failure(Exception("Register failed"))
        } catch (e: Exception) { Result.failure(e) }
    }

    /**
     * 修改阿里云上的用户信息
     */
    suspend fun updateUserProfile(nickname: String, description: String, avatarUrl: String = ""): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            // 注意：这里由于 _userProfile.value.nickname 在云端同步后已经是真实昵称了
            // 为了安全，我们需要使用保存在 DataStore 中的唯一 username 作为主键
            val prefs = settingsManager.settingsFlow.first()
            val currentUsername = prefs.savedUsername
            
            val updateData = mutableMapOf(
                "username" to currentUsername,
                "nickname" to nickname, 
                "description" to description
            )
            if (avatarUrl.isNotEmpty()) {
                updateData["avatarUrl"] = avatarUrl
            }
            
            val response = aliClient.request(
                method = "POST",
                path = "/update_profile?username=$currentUsername",
                body = updateData,
                responseType = Map::class.java
            )

            if (response != null && (response["status"] == "success" || response["statusCode"] == 200.0)) {
                _userProfile.value = _userProfile.value.copy(
                    nickname = nickname,
                    description = description,
                    avatarUrl = if (avatarUrl.isNotEmpty()) avatarUrl else _userProfile.value.avatarUrl
                )
                Result.success(Unit)
            } else {
                Result.failure(Exception("Update failed"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun logout() {
        tokenManager.clear()
        settingsManager.logout()
        _userProfile.value = UserProfile()
    }

    /**
     * 获取 OSS 上传所需的 STS 临时凭证
     */
    suspend fun fetchSTSToken(): Result<Map<String, String>> = withContext(Dispatchers.IO) {
        try {
            val response = aliClient.request(
                method = "GET",
                path = "/get_sts_token",
                responseType = Map::class.java
            )
            if (response != null) {
                @Suppress("UNCHECKED_CAST")
                Result.success(response as Map<String, String>)
            } else {
                Result.failure(Exception("Failed to get STS token"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

data class UserProfile(
    val nickname: String = "Guest",
    val description: String = "Offline User",
    val avatarUrl: String = "",
    val totalTranslations: Int = 0,
    val accuracy: Float = 0.0f,
    val expertLevel: Int = 0
)

data class AuthData(
    @SerializedName("user") val user: User,
    @SerializedName("access_token") val accessToken: String,
    @SerializedName("refresh_token") val refreshToken: String
)
