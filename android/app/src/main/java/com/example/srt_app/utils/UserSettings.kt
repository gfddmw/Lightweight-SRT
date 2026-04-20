package com.example.srt_app.utils

/**
 * UI DTO combining local preferences, user profile, and auth state.
 */
data class UserSettings(
    // Local Preferences
    val appLanguage: String = "en",
    val signLanguageStandard: String = "ASL",
    val outputLanguage: String = "en",
    val showSkeleton: Boolean = true,
    val confidenceThreshold: Float = 0.5f,
    val autoFocus: Boolean = true,
    val textSize: String = "AA",
    val displayDuration: String = "5s",
    val flashOnTranslation: Boolean = false,
    val vibration: Boolean = true,

    // User Profile Stats
    val userName: String = "Guest",
    val userRole: String = "New User",
    val totalTranslations: Int = 0,
    val accuracy: Float = 0.0f,
    val streakDays: String = "0",
    val expertLevel: String = "0",
    val helpedCount: String = "0",
    val fastestDelay: String = "0s",

    // Auth State
    val isLoggedIn: Boolean = false,
    val authToken: String = "",
    val accessToken: String = "",
    val refreshToken: String = ""
)
