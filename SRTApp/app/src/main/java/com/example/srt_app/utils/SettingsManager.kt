package com.example.srt_app.utils

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.*
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
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
                vibration = preferences[PreferencesKeys.VIBRATION] ?: true
            )
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
    val vibration: Boolean
)
