package com.example.srt_app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val DarkColorScheme = darkColorScheme(
    primary = PrimaryColor,
    secondary = SecondaryColor,
    tertiary = TertiaryColor,
    background = SurfaceDim,
    surface = SurfaceContainer,
    surfaceVariant = SurfaceVariant,
    onPrimary = OnPrimaryFixed,
    onSecondary = ColorOnWarm,
    onTertiary = PureWhite,
    onBackground = OnBackground,
    onSurface = OnSurface,
    onSurfaceVariant = OnSurfaceVariant,
    outline = OutlineVariant,
    error = DangerColor
)

private val LightColorScheme = lightColorScheme(
    primary = PrimaryColor,
    secondary = SecondaryColor,
    tertiary = TertiaryColor,
    background = SurfaceDim, // Camera recognition remains dark for visual focus.
    surface = SurfaceContainer,
    surfaceVariant = SurfaceVariant,
    onPrimary = OnPrimaryFixed,
    onSecondary = ColorOnWarm,
    onTertiary = PureWhite,
    onBackground = OnBackground,
    onSurface = OnSurface,
    onSurfaceVariant = OnSurfaceVariant,
    outline = OutlineVariant,
    error = DangerColor
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
