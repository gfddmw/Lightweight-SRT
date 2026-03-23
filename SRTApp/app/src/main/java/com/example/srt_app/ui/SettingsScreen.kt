package com.example.srt_app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.srt_app.R
import com.example.srt_app.ui.theme.CharcoalBackground
import com.example.srt_app.ui.theme.GrayText
import com.example.srt_app.ui.theme.PureWhite

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    currentSignLanguage: String,
    onSignLanguageChange: (String) -> Unit,
    currentOutputLanguage: String,
    onOutputLanguageChange: (String) -> Unit,
    showSkeleton: Boolean,
    onShowSkeletonChange: (Boolean) -> Unit,
    confidenceThreshold: Float,
    onConfidenceThresholdChange: (Float) -> Unit,
    currentAppLanguage: String,
    onAppLanguageChange: (String) -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.settings_title), color = PureWhite) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.back), tint = PureWhite)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = CharcoalBackground)
            )
        },
        containerColor = CharcoalBackground
    ) { paddingValues ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
        ) {
            item { SettingHeader(stringResource(R.string.header_general)) }
            item {
                SelectionSettingItem(
                    title = stringResource(R.string.app_language),
                    subtitle = currentAppLanguage,
                    icon = Icons.Default.Language,
                    options = listOf(stringResource(R.string.lang_english), stringResource(R.string.lang_simplified_chinese)),
                    onSelected = onAppLanguageChange
                )
            }

            item { HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp), color = GrayText.copy(alpha = 0.2f)) }

            item { SettingHeader(stringResource(R.string.header_recognition)) }
            item {
                SelectionSettingItem(
                    title = stringResource(R.string.sign_language_standard),
                    subtitle = currentSignLanguage,
                    icon = Icons.Default.SignLanguage,
                    options = listOf(
                        stringResource(R.string.std_asl),
                        stringResource(R.string.std_csl),
                        stringResource(R.string.std_bsl),
                        stringResource(R.string.std_isl)
                    ),
                    onSelected = onSignLanguageChange
                )
            }
            item {
                SelectionSettingItem(
                    title = stringResource(R.string.output_language),
                    subtitle = currentOutputLanguage,
                    icon = Icons.Default.Translate,
                    options = listOf(
                        stringResource(R.string.lang_english),
                        stringResource(R.string.lang_chinese),
                        stringResource(R.string.lang_spanish),
                        stringResource(R.string.lang_french)
                    ),
                    onSelected = onOutputLanguageChange
                )
            }

            item { HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp), color = GrayText.copy(alpha = 0.2f)) }

            item { SettingHeader(stringResource(R.string.header_visuals)) }
            item {
                SwitchSettingItem(
                    title = stringResource(R.string.show_skeleton_overlay),
                    subtitle = stringResource(R.string.show_skeleton_subtitle),
                    icon = Icons.Default.GraphicEq,
                    checked = showSkeleton,
                    onCheckedChange = onShowSkeletonChange
                )
            }
            item {
                SliderSettingItem(
                    title = stringResource(R.string.confidence_threshold),
                    subtitle = stringResource(R.string.confidence_subtitle),
                    icon = Icons.Default.Speed,
                    value = confidenceThreshold,
                    onValueChange = onConfidenceThresholdChange
                )
            }

            item { HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp), color = GrayText.copy(alpha = 0.2f)) }

            item { SettingHeader(stringResource(R.string.header_about)) }
            item {
                InfoSettingItem(
                    title = stringResource(R.string.version),
                    subtitle = "1.0.0-alpha01",
                    icon = Icons.Default.Info
                )
            }
        }
    }
}

@Composable
fun SettingHeader(text: String) {
    Text(
        text = text,
        color = MaterialTheme.colorScheme.primary,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.Bold,
        modifier = Modifier.padding(vertical = 8.dp)
    )
}

@Composable
fun SelectionSettingItem(
    title: String,
    subtitle: String,
    icon: ImageVector,
    options: List<String>,
    onSelected: (String) -> Unit
) {
    var showDialog by remember { mutableStateOf(false) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { showDialog = true }
            .padding(vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = GrayText, modifier = Modifier.size(24.dp))
        Spacer(modifier = Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(title, color = PureWhite, style = MaterialTheme.typography.bodyLarge)
            Text(subtitle, color = GrayText, style = MaterialTheme.typography.bodyMedium)
        }
        Icon(Icons.Default.ChevronRight, contentDescription = null, tint = GrayText)
    }

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(title) },
            text = {
                Column {
                    options.forEach { option ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    onSelected(option)
                                    showDialog = false
                                }
                                .padding(vertical = 12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(selected = (option == subtitle), onClick = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(option)
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showDialog = false }) { Text(stringResource(R.string.cancel)) }
            }
        )
    }
}

@Composable
fun SwitchSettingItem(
    title: String,
    subtitle: String,
    icon: ImageVector,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = GrayText, modifier = Modifier.size(24.dp))
        Spacer(modifier = Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(title, color = PureWhite, style = MaterialTheme.typography.bodyLarge)
            Text(subtitle, color = GrayText, style = MaterialTheme.typography.bodyMedium)
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
fun SliderSettingItem(
    title: String,
    subtitle: String,
    icon: ImageVector,
    value: Float,
    onValueChange: (Float) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, contentDescription = null, tint = GrayText, modifier = Modifier.size(24.dp))
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text(title, color = PureWhite, style = MaterialTheme.typography.bodyLarge)
                Text(subtitle, color = GrayText, style = MaterialTheme.typography.bodyMedium)
            }
        }
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(top = 8.dp)
        ) {
            Slider(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "${(value * 100).toInt()}%",
                color = PureWhite,
                modifier = Modifier.width(48.dp).padding(start = 8.dp),
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Composable
fun InfoSettingItem(
    title: String,
    subtitle: String,
    icon: ImageVector
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = GrayText, modifier = Modifier.size(24.dp))
        Spacer(modifier = Modifier.width(16.dp))
        Column {
            Text(title, color = PureWhite, style = MaterialTheme.typography.bodyLarge)
            Text(subtitle, color = GrayText, style = MaterialTheme.typography.bodyMedium)
        }
    }
}
