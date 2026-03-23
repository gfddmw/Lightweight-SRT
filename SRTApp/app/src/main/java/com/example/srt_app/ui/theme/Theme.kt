package com.example.srt_app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val DarkColorScheme = darkColorScheme(
    primary = PrimaryBlue,
    secondary = GlowingBlue,
    background = CharcoalBackground,
    surface = SurfaceDark,
    onPrimary = OnPrimaryDark,
    onSecondary = CharcoalBackground,
    onBackground = PureWhite,
    onSurface = PureWhite,
    error = ErrorRed
)

private val LightColorScheme = lightColorScheme(
    primary = PrimaryBlue,
    secondary = GlowingBlue,
    background = CharcoalBackground, // Live Transcribe typically stays dark
    surface = SurfaceDark,
    onPrimary = OnPrimaryDark,
    onSecondary = CharcoalBackground,
    onBackground = PureWhite,
    onSurface = PureWhite,
    error = ErrorRed
)

@Composable
fun SRTAppTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
