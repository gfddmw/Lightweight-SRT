package com.example.srt_app.utils

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import java.io.IOException

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

class SettingsManager(context: Context) {

    private val dataStore = context.dataStore

    object PreferencesKeys {
        val APP_LANGUAGE = stringPreferencesKey("app_language")
        val SIGN_LANGUAGE_STANDARD = stringPreferencesKey("sign_language_standard")
        val OUTPUT_LANGUAGE = stringPreferencesKey("output_language")
        val SHOW_SKELETON = booleanPreferencesKey("show_skeleton")
        val CONFIDENCE_THRESHOLD = floatPreferencesKey("confidence_threshold")
        val AUTO_FOCUS = booleanPreferencesKey("auto_focus")
        val TEXT_SIZE = stringPreferencesKey("text_size")
        val DISPLAY_DURATION = stringPreferencesKey("display_duration")
        val FLASH_ON_TRANSLATION = booleanPreferencesKey("flash_on_translation")
        val VIBRATION = booleanPreferencesKey("vibration")
        
        // User Profile Stats
        val USER_NAME = stringPreferencesKey("user_name")
        val USER_ROLE = stringPreferencesKey("user_role")
        val TOTAL_TRANSLATIONS = intPreferencesKey("total_translations")
        val ACCURACY = floatPreferencesKey("accuracy")
        val STREAK_DAYS = intPreferencesKey("streak_days")
        val EXPERT_LEVEL = intPreferencesKey("expert_level")
        val HELPED_COUNT = intPreferencesKey("helped_count")
        val FASTEST_DELAY = floatPreferencesKey("fastest_delay")
        
        // Auth State
        val IS_LOGGED_IN = booleanPreferencesKey("is_logged_in")
        val AUTH_TOKEN = stringPreferencesKey("auth_token")
        val ACCESS_TOKEN = stringPreferencesKey("access_token")
        val REFRESH_TOKEN = stringPreferencesKey("refresh_token")
    }

    val settingsFlow: Flow<UserSettings> = dataStore.data
        .catch { exception ->
            if (exception is IOException) {
                emit(emptyPreferences())
            } else {
                throw exception
            }
        }.map { preferences ->
            UserSettings(
                appLanguage = preferences[PreferencesKeys.APP_LANGUAGE] ?: "en",
                signLanguageStandard = preferences[PreferencesKeys.SIGN_LANGUAGE_STANDARD] ?: "ASL",
                outputLanguage = preferences[PreferencesKeys.OUTPUT_LANGUAGE] ?: "en",
                showSkeleton = preferences[PreferencesKeys.SHOW_SKELETON] ?: true,
                confidenceThreshold = preferences[PreferencesKeys.CONFIDENCE_THRESHOLD] ?: 0.5f,
                autoFocus = preferences[PreferencesKeys.AUTO_FOCUS] ?: true,
                textSize = preferences[PreferencesKeys.TEXT_SIZE] ?: "AA",
                displayDuration = preferences[PreferencesKeys.DISPLAY_DURATION] ?: "5s",
                flashOnTranslation = preferences[PreferencesKeys.FLASH_ON_TRANSLATION] ?: false,
                vibration = preferences[PreferencesKeys.VIBRATION] ?: true,
                
                // Profile
                userName = preferences[PreferencesKeys.USER_NAME] ?: "Guest",
                userRole = preferences[PreferencesKeys.USER_ROLE] ?: "New User",
                totalTranslations = preferences[PreferencesKeys.TOTAL_TRANSLATIONS] ?: 0,
                accuracy = preferences[PreferencesKeys.ACCURACY] ?: 0.0f,
                streakDays = preferences[PreferencesKeys.STREAK_DAYS] ?: 0,
                expertLevel = preferences[PreferencesKeys.EXPERT_LEVEL] ?: 0,
                helpedCount = preferences[PreferencesKeys.HELPED_COUNT] ?: 0,
                fastestDelay = preferences[PreferencesKeys.FASTEST_DELAY] ?: 0.0f,

                // Auth
                isLoggedIn = preferences[PreferencesKeys.IS_LOGGED_IN] ?: false,
                authToken = preferences[PreferencesKeys.AUTH_TOKEN] ?: "",
                accessToken = preferences[PreferencesKeys.ACCESS_TOKEN] ?: "",
                refreshToken = preferences[PreferencesKeys.REFRESH_TOKEN] ?: ""
            )
        }

    suspend fun setLoggedIn(loggedIn: Boolean, token: String = "", name: String = "", role: String = "", accessToken: String = "", refreshToken: String = "") {
        dataStore.edit { 
            it[PreferencesKeys.IS_LOGGED_IN] = loggedIn
            it[PreferencesKeys.AUTH_TOKEN] = token
            it[PreferencesKeys.ACCESS_TOKEN] = accessToken
            it[PreferencesKeys.REFRESH_TOKEN] = refreshToken
            if (name.isNotEmpty()) it[PreferencesKeys.USER_NAME] = name
            if (role.isNotEmpty()) it[PreferencesKeys.USER_ROLE] = role
        }
    }

    suspend fun updateTokens(accessToken: String, refreshToken: String) {
        dataStore.edit {
            it[PreferencesKeys.ACCESS_TOKEN] = accessToken
            it[PreferencesKeys.REFRESH_TOKEN] = refreshToken
        }
    }

    /**
     * 同步获取 AccessToken (供拦截器使用)
     */
    suspend fun getAccessTokenSync(): String {
        return dataStore.data.first()[PreferencesKeys.ACCESS_TOKEN] ?: ""
    }

    suspend fun logout() {
        dataStore.edit { 
            it[PreferencesKeys.IS_LOGGED_IN] = false
            it[PreferencesKeys.AUTH_TOKEN] = ""
        }
    }

    suspend fun updateAppLanguage(language: String) {
        dataStore.edit { it[PreferencesKeys.APP_LANGUAGE] = language }
    }

    suspend fun updateSignLanguageStandard(standard: String) {
        dataStore.edit { it[PreferencesKeys.SIGN_LANGUAGE_STANDARD] = standard }
    }

    suspend fun updateOutputLanguage(language: String) {
        dataStore.edit { it[PreferencesKeys.OUTPUT_LANGUAGE] = language }
    }

    suspend fun updateShowSkeleton(show: Boolean) {
        dataStore.edit { it[PreferencesKeys.SHOW_SKELETON] = show }
    }

    suspend fun updateConfidenceThreshold(threshold: Float) {
        dataStore.edit { it[PreferencesKeys.CONFIDENCE_THRESHOLD] = threshold }
    }

    suspend fun updateAutoFocus(enabled: Boolean) {
        dataStore.edit { it[PreferencesKeys.AUTO_FOCUS] = enabled }
    }

    suspend fun updateTextSize(size: String) {
        dataStore.edit { it[PreferencesKeys.TEXT_SIZE] = size }
    }

    suspend fun updateDisplayDuration(duration: String) {
        dataStore.edit { it[PreferencesKeys.DISPLAY_DURATION] = duration }
    }

    suspend fun updateFlashOnTranslation(enabled: Boolean) {
        dataStore.edit { it[PreferencesKeys.FLASH_ON_TRANSLATION] = enabled }
    }

    suspend fun updateVibration(enabled: Boolean) {
        dataStore.edit { it[PreferencesKeys.VIBRATION] = enabled }
    }

    // Profile Updates
    suspend fun updateUserName(name: String) {
        dataStore.edit { it[PreferencesKeys.USER_NAME] = name }
    }

    suspend fun updateUserRole(role: String) {
        dataStore.edit { it[PreferencesKeys.USER_ROLE] = role }
    }

    suspend fun updateStats(translations: Int, accuracy: Float) {
        dataStore.edit { 
            it[PreferencesKeys.TOTAL_TRANSLATIONS] = translations
            it[PreferencesKeys.ACCURACY] = accuracy
        }
    }

    suspend fun resetSettings() {
        dataStore.edit { it.clear() }
    }
}

data class UserSettings(
    val appLanguage: String,
    val signLanguageStandard: String,
    val outputLanguage: String,
    val showSkeleton: Boolean,
    val confidenceThreshold: Float,
    val autoFocus: Boolean,
    val textSize: String,
    val displayDuration: String,
    val flashOnTranslation: Boolean,
    val vibration: Boolean,
    
    // Profile Data
    val userName: String,
    val userRole: String,
    val totalTranslations: Int,
    val accuracy: Float,
    val streakDays: Int,
    val expertLevel: Int,
    val helpedCount: Int,
    val fastestDelay: Float,
    
    // Auth State
    val isLoggedIn: Boolean,
    val authToken: String,
    val accessToken: String,
    val refreshToken: String
)
