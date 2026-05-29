package com.example.srt_app.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.runtime.*
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.srt_app.R
import com.example.srt_app.ui.theme.*

@Composable
fun SenseBottomNavBar(
    modifier: Modifier = Modifier,
    selectedTab: Int = 1, // 0: Profile, 1: Translator
    onNavigateToProfile: () -> Unit,
    onNavigateToTranslator: () -> Unit
) {
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(topStart = 20.dp, topEnd = 20.dp)),
        color = SurfaceContainer.copy(alpha = 0.98f),
        border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.24f)),
        shadowElevation = 18.dp
    ) {
        Row(
            modifier = Modifier
                .navigationBarsPadding()
                .padding(horizontal = 20.dp, vertical = 10.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            NavItem(
                modifier = Modifier.weight(1f),
                icon = Icons.Default.Person,
                label = stringResource(R.string.profile),
                isActive = selectedTab == 0,
                onClick = onNavigateToProfile
            )
            NavItem(
                modifier = Modifier.weight(1f),
                icon = Icons.Default.Translate,
                label = stringResource(R.string.translator),
                isActive = selectedTab == 1,
                onClick = onNavigateToTranslator
            )
        }
    }
}

@Composable
fun NavItem(
    modifier: Modifier = Modifier,
    icon: ImageVector,
    label: String,
    isActive: Boolean,
    onClick: () -> Unit
) {
    Surface(
        modifier = Modifier
            .then(modifier)
            .heightIn(min = 56.dp)
            .clip(RoundedCornerShape(16.dp))
            .clickable { onClick() }
            .padding(2.dp),
        color = if (isActive) PrimaryColor else Color.Transparent,
        shape = RoundedCornerShape(14.dp),
        border = if (isActive) null else BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.18f))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = label,
                tint = if (isActive) OnPrimaryFixed else OnSurfaceVariant,
                modifier = Modifier.size(22.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium.copy(
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (isActive) OnPrimaryFixed else OnSurfaceVariant
                )
            )
        }
    }
}

@Composable
fun SenseInputField(
    value: String, 
    onValueChange: (String) -> Unit, 
    label: String, 
    icon: ImageVector,
    modifier: Modifier = Modifier,
    visualTransformation: VisualTransformation = VisualTransformation.None,
    placeholder: String = "",
    errorText: String? = null
) {
    var isFocused by remember { mutableStateOf(false) }
    val borderColor = when {
        errorText != null -> DangerColor
        isFocused -> PrimaryColor
        else -> OutlineVariant.copy(alpha = 0.35f)
    }
    val borderWidth = if (isFocused || errorText != null) 1.5.dp else 1.dp

    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium.copy(
                fontWeight = FontWeight.Bold,
                color = if (errorText != null) DangerColor else OnSurfaceVariant
            ),
            modifier = Modifier.padding(start = 4.dp)
        )
        Surface(
            color = SurfaceContainerLow,
            shape = RoundedCornerShape(14.dp),
            border = BorderStroke(borderWidth, borderColor)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 56.dp)
                    .padding(horizontal = 14.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Icon(
                    imageVector = icon, 
                    contentDescription = null, 
                    tint = when {
                        errorText != null -> DangerColor
                        isFocused -> PrimaryColor
                        else -> OnSurfaceVariant
                    }, 
                    modifier = Modifier.size(20.dp)
                )
                TextField(
                    value = value, 
                    onValueChange = onValueChange, 
                    modifier = Modifier
                        .weight(1f)
                        .onFocusChanged { isFocused = it.isFocused },
                    visualTransformation = visualTransformation,
                    placeholder = {
                        if (placeholder.isNotEmpty()) {
                            Text(placeholder, color = OnSurfaceVariant.copy(alpha = 0.58f))
                        }
                    },
                    singleLine = true,
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = Color.Transparent,
                        unfocusedContainerColor = Color.Transparent,
                        disabledContainerColor = Color.Transparent,
                        focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        disabledIndicatorColor = Color.Transparent,
                        cursorColor = PrimaryColor,
                        focusedTextColor = OnSurface,
                        unfocusedTextColor = OnSurface
                    )
                )
            }
        }
        if (errorText != null) {
            Text(
                text = errorText,
                color = DangerColor,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(start = 4.dp, top = 2.dp)
            )
        }
    }
}

@Composable
fun SensePasswordField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    modifier: Modifier = Modifier,
    placeholder: String = "",
    errorText: String? = null
) {
    var isPasswordVisible by remember { mutableStateOf(false) }
    var isFocused by remember { mutableStateOf(false) }
    val borderColor = when {
        errorText != null -> DangerColor
        isFocused -> PrimaryColor
        else -> OutlineVariant.copy(alpha = 0.35f)
    }
    val borderWidth = if (isFocused || errorText != null) 1.5.dp else 1.dp

    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium.copy(
                fontWeight = FontWeight.Bold,
                color = if (errorText != null) DangerColor else OnSurfaceVariant
            ),
            modifier = Modifier.padding(start = 4.dp)
        )
        Surface(
            color = SurfaceContainerLow,
            shape = RoundedCornerShape(14.dp),
            border = BorderStroke(borderWidth, borderColor)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 56.dp)
                    .padding(horizontal = 14.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Lock, 
                    contentDescription = null, 
                    tint = when {
                        errorText != null -> DangerColor
                        isFocused -> PrimaryColor
                        else -> OnSurfaceVariant
                    }, 
                    modifier = Modifier.size(20.dp)
                )
                TextField(
                    value = value,
                    onValueChange = onValueChange,
                    modifier = Modifier
                        .weight(1f)
                        .onFocusChanged { isFocused = it.isFocused },
                    visualTransformation = if (isPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                    placeholder = {
                        if (placeholder.isNotEmpty()) {
                            Text(placeholder, color = OnSurfaceVariant.copy(alpha = 0.58f))
                        }
                    },
                    singleLine = true,
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = Color.Transparent,
                        unfocusedContainerColor = Color.Transparent,
                        disabledContainerColor = Color.Transparent,
                        focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        disabledIndicatorColor = Color.Transparent,
                        cursorColor = PrimaryColor,
                        focusedTextColor = OnSurface,
                        unfocusedTextColor = OnSurface
                    )
                )
                IconButton(onClick = { isPasswordVisible = !isPasswordVisible }) {
                    Icon(
                        imageVector = if (isPasswordVisible) Icons.Default.VisibilityOff else Icons.Default.Visibility,
                        contentDescription = if (isPasswordVisible) "Hide password" else "Show password",
                        tint = OnSurfaceVariant
                    )
                }
            }
        }
        if (errorText != null) {
            Text(
                text = errorText,
                color = DangerColor,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(start = 4.dp, top = 2.dp)
            )
        }
    }
}

@Composable
fun SenseVerificationCodeField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    onGetCodeClick: () -> Unit,
    isGetCodeEnabled: Boolean,
    modifier: Modifier = Modifier,
    placeholder: String = "",
    errorText: String? = null
) {
    var isFocused by remember { mutableStateOf(false) }
    val borderColor = when {
        errorText != null -> DangerColor
        isFocused -> PrimaryColor
        else -> OutlineVariant.copy(alpha = 0.35f)
    }
    val borderWidth = if (isFocused || errorText != null) 1.5.dp else 1.dp

    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium.copy(
                fontWeight = FontWeight.Bold,
                color = if (errorText != null) DangerColor else OnSurfaceVariant
            ),
            modifier = Modifier.padding(start = 4.dp)
        )
        Surface(
            color = SurfaceContainerLow,
            shape = RoundedCornerShape(14.dp),
            border = BorderStroke(borderWidth, borderColor)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 56.dp)
                    .padding(horizontal = 14.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.VpnKey, 
                    contentDescription = null, 
                    tint = when {
                        errorText != null -> DangerColor
                        isFocused -> PrimaryColor
                        else -> OnSurfaceVariant
                    }, 
                    modifier = Modifier.size(20.dp)
                )
                TextField(
                    value = value,
                    onValueChange = onValueChange,
                    modifier = Modifier
                        .weight(1f)
                        .onFocusChanged { isFocused = it.isFocused },
                    placeholder = {
                        if (placeholder.isNotEmpty()) {
                            Text(placeholder, color = OnSurfaceVariant.copy(alpha = 0.58f))
                        }
                    },
                    singleLine = true,
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = Color.Transparent,
                        unfocusedContainerColor = Color.Transparent,
                        disabledContainerColor = Color.Transparent,
                        focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        disabledIndicatorColor = Color.Transparent,
                        cursorColor = PrimaryColor,
                        focusedTextColor = OnSurface,
                        unfocusedTextColor = OnSurface
                    )
                )
                TextButton(
                    onClick = onGetCodeClick,
                    enabled = isGetCodeEnabled
                ) {
                    Text(
                        text = stringResource(R.string.get_code),
                        color = if (isGetCodeEnabled) PrimaryColor else OnSurfaceVariant.copy(alpha = 0.5f),
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
        if (errorText != null) {
            Text(
                text = errorText,
                color = DangerColor,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(start = 4.dp, top = 2.dp)
            )
        }
    }
}
