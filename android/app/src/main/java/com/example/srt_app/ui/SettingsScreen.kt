package com.example.srt_app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.srt_app.R
import com.example.srt_app.ui.theme.*
import com.example.srt_app.utils.UserSettings

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    onNavigateToProfile: () -> Unit,
    onSave: (UserSettings) -> Unit,
    onReset: () -> Unit = {},
    // Current actual settings
    initialSettings: UserSettings
) {
    // Local state for editing
    var appLanguage by remember { mutableStateOf(initialSettings.appLanguage) }
    var signLanguage by remember { mutableStateOf(initialSettings.signLanguageStandard) }
    var outputLanguage by remember { mutableStateOf(initialSettings.outputLanguage) }
    var showSkeleton by remember { mutableStateOf(initialSettings.showSkeleton) }
    var confidenceThreshold by remember { mutableFloatStateOf(initialSettings.confidenceThreshold) }
    var autoFocus by remember { mutableStateOf(initialSettings.autoFocus) }
    var textSize by remember { mutableStateOf(initialSettings.textSize) }
    var displayDuration by remember { mutableStateOf(initialSettings.displayDuration) }
    var flashOnTranslation by remember { mutableStateOf(initialSettings.flashOnTranslation) }
    var vibration by remember { mutableStateOf(initialSettings.vibration) }
    var showUserGuideDialog by remember { mutableStateOf(false) }
    var showResetDialog by remember { mutableStateOf(false) }
    var showUpgradeDialog by remember { mutableStateOf(false) }

    fun currentSettings(): UserSettings {
        return UserSettings(
            appLanguage = appLanguage,
            signLanguageStandard = signLanguage,
            outputLanguage = outputLanguage,
            showSkeleton = showSkeleton,
            confidenceThreshold = confidenceThreshold,
            autoFocus = autoFocus,
            textSize = textSize,
            displayDuration = displayDuration,
            flashOnTranslation = flashOnTranslation,
            vibration = vibration,

            userName = initialSettings.userName,
            userRole = initialSettings.userRole,
            totalTranslations = initialSettings.totalTranslations,
            accuracy = initialSettings.accuracy,
            streakDays = initialSettings.streakDays,
            expertLevel = initialSettings.expertLevel,
            helpedCount = initialSettings.helpedCount,
            fastestDelay = initialSettings.fastestDelay,

            isLoggedIn = initialSettings.isLoggedIn,
            authToken = initialSettings.authToken,
            accessToken = initialSettings.accessToken,
            refreshToken = initialSettings.refreshToken
        )
    }

    fun resetLocalSettings() {
        appLanguage = "en"
        signLanguage = "ASL"
        outputLanguage = "en"
        showSkeleton = true
        confidenceThreshold = 0.5f
        autoFocus = true
        textSize = "AA"
        displayDuration = "5s"
        flashOnTranslation = false
        vibration = true
        onReset()
        showResetDialog = false
    }

    Scaffold(
        topBar = {
            Surface(
                color = Color(0xFF0E0E0E).copy(alpha = 0.8f),
                modifier = Modifier.fillMaxWidth()
            ) {
                TopAppBar(
                    title = { 
                        Text(
                            stringResource(R.string.settings_title), 
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                            color = Color.White 
                        ) 
                    },
                    navigationIcon = {
                        IconButton(onClick = onBack) {
                            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = PrimaryColor)
                        }
                    },
                    actions = {
                        IconButton(onClick = {
                            onSave(currentSettings())
                        }) {
                            Icon(Icons.Default.Save, contentDescription = stringResource(R.string.save), tint = PrimaryColor)
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.Transparent)
                )
            }
        },
        containerColor = Color(0xFF0E0E0E),
        bottomBar = {
            // Reusing the same SenseBottomNavBar but for the settings context (simplified)
            SenseBottomNavBar(
                modifier = Modifier,
                selectedTab = -1,
                onNavigateToProfile = onNavigateToProfile,
                onNavigateToTranslator = onBack
            )
        }
    ) { paddingValues ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            item { Spacer(modifier = Modifier.height(8.dp)) }

            // 1. Camera & Tracking Section
            item {
                SettingsSection(title = stringResource(R.string.header_camera_tracking)) {
                    // Confidence Threshold (Sensitivity)
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.Speed, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(20.dp))
                                Spacer(modifier = Modifier.width(16.dp))
                                Text(stringResource(R.string.confidence_threshold), color = Color.White, fontWeight = FontWeight.Medium)
                            }
                            Text(String.format("%.1f%%", confidenceThreshold * 100), color = PrimaryColor, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                        }
                        Spacer(modifier = Modifier.height(12.dp))
                        Slider(
                            value = confidenceThreshold,
                            onValueChange = { confidenceThreshold = it },
                            valueRange = 0.1f..0.9f,
                            colors = SliderDefaults.colors(
                                thumbColor = PrimaryColor,
                                activeTrackColor = PrimaryColor,
                                inactiveTrackColor = SurfaceContainerHighest
                            )
                        )
                    }

                    // Show Skeleton
                    SettingItem(
                        icon = Icons.Default.Visibility,
                        title = stringResource(R.string.show_skeleton_overlay),
                        subtitle = stringResource(R.string.show_skeleton_subtitle),
                        onClick = { showSkeleton = !showSkeleton },
                        trailing = {
                            Switch(
                                checked = showSkeleton,
                                onCheckedChange = { showSkeleton = it },
                                colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = PrimaryColor)
                            )
                        }
                    )

                    // Auto-focus
                    SettingItem(
                        icon = Icons.Default.CenterFocusStrong,
                        title = stringResource(R.string.auto_focus),
                        onClick = { autoFocus = !autoFocus },
                        trailing = {
                            Switch(
                                checked = autoFocus, 
                                onCheckedChange = { autoFocus = it },
                                colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = PrimaryColor)
                            )
                        }
                    )
                }
            }

            // 2. Translation Section
            item {
                SettingsSection(title = stringResource(R.string.header_translation)) {
                    // App Language
                    SettingItem(
                        icon = Icons.Default.Language,
                        title = stringResource(R.string.app_language),
                        subtitle = if (appLanguage == "en") stringResource(R.string.lang_english) else stringResource(R.string.lang_chinese),
                        onClick = { appLanguage = if (appLanguage == "en") "zh" else "en" },
                        trailing = {
                            IconButton(onClick = {
                                appLanguage = if (appLanguage == "en") "zh" else "en"
                            }) {
                                Icon(Icons.Default.SwapHoriz, contentDescription = null, tint = OnSurfaceVariant)
                            }
                        }
                    )

                    // Standard selection (simple toggle for demo)
                    SettingItem(
                        icon = Icons.Default.Translate,
                        title = stringResource(R.string.sign_language_standard),
                        subtitle = if (signLanguage == "ASL") stringResource(R.string.std_asl) else stringResource(R.string.std_csl),
                        onClick = { signLanguage = if (signLanguage == "ASL") "CSL" else "ASL" },
                        trailing = {
                            IconButton(onClick = {
                                signLanguage = if (signLanguage == "ASL") "CSL" else "ASL"
                            }) {
                                Icon(Icons.Default.SwapHoriz, contentDescription = null, tint = OnSurfaceVariant)
                            }
                        }
                    )
                    
                    SettingItem(
                        icon = Icons.Default.Translate,
                        title = stringResource(R.string.output_language),
                        subtitle = if (outputLanguage == "en") stringResource(R.string.lang_english) else stringResource(R.string.lang_chinese),
                        onClick = { outputLanguage = if (outputLanguage == "en") "zh" else "en" },
                        trailing = {
                            IconButton(onClick = {
                                outputLanguage = if (outputLanguage == "en") "zh" else "en"
                            }) {
                                Icon(Icons.Default.SwapHoriz, contentDescription = null, tint = OnSurfaceVariant)
                            }
                        }
                    )

                    // Text Size Toggle
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.FormatSize, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(20.dp))
                            Spacer(modifier = Modifier.width(16.dp))
                            Text(stringResource(R.string.text_size), color = Color.White, fontWeight = FontWeight.Medium)
                        }
                        
                        Row(
                            modifier = Modifier
                                .background(SurfaceContainerHighest, RoundedCornerShape(8.dp))
                                .padding(4.dp)
                        ) {
                            TextSizeButton("A", active = textSize == "A") { textSize = "A" }
                            TextSizeButton("AA", active = textSize == "AA") { textSize = "AA" }
                            TextSizeButton("AAA", active = textSize == "AAA") { textSize = "AAA" }
                        }
                    }

                    // Display Duration
                    SettingItem(
                        icon = Icons.Default.Timer,
                        title = stringResource(R.string.display_duration),
                        subtitle = stringResource(R.string.display_duration_subtitle),
                        onClick = {
                            displayDuration = when(displayDuration) {
                                "3s" -> "5s"
                                "5s" -> "8s"
                                else -> "3s"
                            }
                        },
                        trailing = {
                            IconButton(onClick = {
                                displayDuration = when(displayDuration) {
                                    "3s" -> "5s"
                                    "5s" -> "8s"
                                    else -> "3s"
                                }
                            }) {
                                Text(displayDuration, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                            }
                        }
                    )
                }
            }

            // 3. Notifications Section
            item {
                SettingsSection(title = stringResource(R.string.header_notifications)) {
                    SettingItem(
                        icon = Icons.Default.FlashlightOn,
                        title = stringResource(R.string.flash_on_translation),
                        onClick = { flashOnTranslation = !flashOnTranslation },
                        trailing = {
                            Switch(
                                checked = flashOnTranslation, 
                                onCheckedChange = { flashOnTranslation = it },
                                colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = PrimaryColor)
                            )
                        }
                    )
                    SettingItem(
                        icon = Icons.Default.Vibration,
                        title = stringResource(R.string.vibration),
                        onClick = { vibration = !vibration },
                        trailing = {
                            Switch(
                                checked = vibration, 
                                onCheckedChange = { vibration = it },
                                colors = SwitchDefaults.colors(checkedThumbColor = Color.White, checkedTrackColor = PrimaryColor)
                            )
                        }
                    )
                }
            }

            // 4. About & Help
            item {
                SettingsSection(title = stringResource(R.string.header_about_help)) {
                    SettingItem(
                        icon = Icons.Default.HelpOutline,
                        title = stringResource(R.string.user_guide),
                        onClick = { showUserGuideDialog = true },
                        trailing = { Icon(Icons.AutoMirrored.Filled.OpenInNew, contentDescription = null, tint = OnSurfaceVariant) }
                    )
                    SettingItem(
                        icon = Icons.Default.Info,
                        title = stringResource(R.string.version),
                        trailing = { Text("v1.0.0-stable", color = OnSurfaceVariant, fontSize = 14.sp) }
                    )
                    
                    // Danger Zone
                    HorizontalDivider(
                        modifier = Modifier.padding(horizontal = 16.dp),
                        thickness = 1.dp,
                        color = Color.White.copy(alpha = 0.05f)
                    )
                    
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { showResetDialog = true }
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.Logout, contentDescription = null, tint = Color(0xFFFF716C), modifier = Modifier.size(20.dp))
                        Spacer(modifier = Modifier.width(16.dp))
                        Text(stringResource(R.string.reset_settings), color = Color(0xFFFF716C), fontWeight = FontWeight.Medium)
                    }
                }
            }

            // 5. Promotional Card
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(18.dp))
                        .background(SurfaceContainerHigh)
                        .padding(24.dp)
                ) {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Text(
                            stringResource(R.string.pro_tracking_title), 
                            style = MaterialTheme.typography.titleLarge, 
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            stringResource(R.string.pro_tracking_desc),
                            color = OnSurfaceVariant,
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.fillMaxWidth(0.7f)
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = { showUpgradeDialog = true },
                            colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text(stringResource(R.string.upgrade_now).uppercase(), fontWeight = FontWeight.Bold, fontSize = 12.sp)
                        }
                    }

                    Icon(
                        Icons.Default.WorkspacePremium, 
                        contentDescription = null, 
                        tint = PrimaryColor.copy(alpha = 0.1f),
                        modifier = Modifier
                            .align(Alignment.CenterEnd)
                            .size(80.dp)
                    )
                }
            }

            item { Spacer(modifier = Modifier.height(32.dp)) }
        }
    }

    if (showUserGuideDialog) {
        AlertDialog(
            onDismissRequest = { showUserGuideDialog = false },
            title = { Text(stringResource(R.string.user_guide_title)) },
            text = { Text(stringResource(R.string.user_guide_body), color = OnSurfaceVariant) },
            confirmButton = {
                TextButton(onClick = { showUserGuideDialog = false }) {
                    Text(stringResource(R.string.ok))
                }
            },
            containerColor = SurfaceContainer,
            titleContentColor = Color.White,
            textContentColor = OnSurfaceVariant
        )
    }

    if (showResetDialog) {
        AlertDialog(
            onDismissRequest = { showResetDialog = false },
            title = { Text(stringResource(R.string.reset_settings_title)) },
            text = { Text(stringResource(R.string.reset_settings_desc), color = OnSurfaceVariant) },
            confirmButton = {
                TextButton(onClick = { resetLocalSettings() }) {
                    Text(stringResource(R.string.reset), color = DangerColor)
                }
            },
            dismissButton = {
                TextButton(onClick = { showResetDialog = false }) {
                    Text(stringResource(R.string.cancel))
                }
            },
            containerColor = SurfaceContainer,
            titleContentColor = Color.White,
            textContentColor = OnSurfaceVariant
        )
    }

    if (showUpgradeDialog) {
        AlertDialog(
            onDismissRequest = { showUpgradeDialog = false },
            title = { Text(stringResource(R.string.upgrade_title)) },
            text = { Text(stringResource(R.string.upgrade_desc), color = OnSurfaceVariant) },
            confirmButton = {
                TextButton(onClick = { showUpgradeDialog = false }) {
                    Text(stringResource(R.string.ok))
                }
            },
            containerColor = SurfaceContainer,
            titleContentColor = Color.White,
            textContentColor = OnSurfaceVariant
        )
    }
}

@Composable
fun SettingsSection(title: String, content: @Composable ColumnScope.() -> Unit) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = title.uppercase(),
            style = MaterialTheme.typography.labelSmall.copy(
                fontWeight = FontWeight.Bold,
                letterSpacing = 2.sp
            ),
            color = OnSurfaceVariant,
            modifier = Modifier.padding(start = 8.dp, bottom = 12.dp)
        )
        Surface(
            color = SurfaceContainer,
            shape = RoundedCornerShape(24.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(content = content)
        }
    }
}

@Composable
fun SettingItem(
    icon: ImageVector,
    title: String,
    subtitle: String? = null,
    onClick: (() -> Unit)? = null,
    trailing: @Composable () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(enabled = onClick != null) { onClick?.invoke() }
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(20.dp))
        Spacer(modifier = Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(title, color = Color.White, fontWeight = FontWeight.Medium, fontSize = 16.sp)
            if (subtitle != null) {
                Text(subtitle, color = OnSurfaceVariant, fontSize = 12.sp)
            }
        }
        trailing()
    }
}

@Composable
fun TextSizeButton(label: String, active: Boolean, onClick: () -> Unit) {
    Surface(
        color = if (active) PrimaryColor else Color.Transparent,
        shape = RoundedCornerShape(8.dp),
        modifier = Modifier
            .width(48.dp)
            .height(32.dp)
            .clickable { onClick() }
    ) {
        Box(contentAlignment = Alignment.Center) {
            Text(
                text = label,
                color = if (active) OnPrimaryFixed else Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 12.sp
            )
        }
    }
}

