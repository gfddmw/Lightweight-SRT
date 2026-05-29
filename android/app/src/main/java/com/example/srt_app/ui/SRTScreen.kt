package com.example.srt_app.ui

import androidx.camera.view.PreviewView
import androidx.compose.animation.core.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.srt_app.R
import com.example.srt_app.ui.theme.*
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

@Composable
fun SRTScreen(
    translationResult: String,
    onStartCamera: (PreviewView) -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateToProfile: () -> Unit,
    onCapture: () -> Unit = {},
    onFlipCamera: () -> Unit = {},
    onTypeBackSend: (String) -> Unit = {},
    showSkeleton: Boolean = true,
    handLandmarks: HandLandmarkerResult? = null,
    signLanguage: String = "ASL",
    outputLanguage: String = "en",
    textSize: String = "AA",
    flashOnTranslation: Boolean = false
) {
    val signShort = signLanguage
    val outputShort = if (outputLanguage == "en") "EN" else "ZH"
    val hasHand = handLandmarks?.landmarks()?.isNotEmpty() == true
    var replyText by remember { mutableStateOf("") }
    val flashAlpha = remember { Animatable(0f) }

    LaunchedEffect(translationResult, flashOnTranslation) {
        if (flashOnTranslation && translationResult.isNotBlank()) {
            flashAlpha.snapTo(0.36f)
            flashAlpha.animateTo(0f, animationSpec = tween(durationMillis = 460))
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(SurfaceDim)
    ) {
        CameraPreview(onStartCamera)

        if (showSkeleton && handLandmarks != null) {
            SkeletonOverlay(handLandmarks)
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            SurfaceDim.copy(alpha = 0.78f),
                            Color.Transparent,
                            SurfaceDim.copy(alpha = 0.92f)
                        )
                    )
                )
        )

        if (flashAlpha.value > 0f) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.White.copy(alpha = flashAlpha.value))
            )
        }

        SenseTopBar(
            onSettingsClick = onNavigateToSettings,
            modifier = Modifier.align(Alignment.TopCenter)
        )

        Column(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .statusBarsPadding()
                .padding(top = 76.dp, end = 18.dp),
            horizontalAlignment = Alignment.End,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            TrackingStatusChip(isActive = hasHand)
            LanguagePairChip(signShort = signShort, outputShort = outputShort)
        }

        Column(
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            CameraControlButton(
                icon = Icons.Default.CameraAlt,
                contentDescription = stringResource(R.string.capture_photo),
                onClick = onCapture
            )
            CameraControlButton(
                icon = Icons.Default.FlipCameraIos,
                contentDescription = stringResource(R.string.flip_camera),
                onClick = onFlipCamera
            )
        }

        TranslationPanel(
            translationResult = translationResult,
            replyText = replyText,
            onReplyTextChange = { replyText = it },
            textSize = textSize,
            onReplySend = {
                if (replyText.isNotBlank()) {
                    onTypeBackSend(replyText.trim())
                    replyText = ""
                }
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(horizontal = 16.dp)
                .padding(bottom = 96.dp)
        )

        SenseBottomNavBar(
            modifier = Modifier.align(Alignment.BottomCenter),
            selectedTab = 1,
            onNavigateToProfile = onNavigateToProfile,
            onNavigateToTranslator = { /* Already here */ }
        )
    }
}

@Composable
fun SenseTopBar(onSettingsClick: () -> Unit, modifier: Modifier = Modifier) {
    Surface(
        color = SurfaceDim.copy(alpha = 0.72f),
        border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.18f)),
        modifier = modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .statusBarsPadding()
                .padding(horizontal = 18.dp, vertical = 12.dp)
                .fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Surface(color = PrimaryColor, shape = RoundedCornerShape(8.dp), modifier = Modifier.size(34.dp)) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(Icons.Default.Translate, contentDescription = null, tint = OnPrimaryFixed, modifier = Modifier.size(20.dp))
                    }
                }
                Column {
                    Text(
                        text = stringResource(R.string.app_name),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold),
                        color = OnSurface
                    )
                    Text(
                        text = stringResource(R.string.camera_hint),
                        style = MaterialTheme.typography.labelSmall,
                        color = OnSurfaceVariant
                    )
                }
            }
            IconButton(onClick = onSettingsClick) {
                Icon(Icons.Default.Settings, contentDescription = stringResource(R.string.open_settings), tint = PrimaryColor)
            }
        }
    }
}

@Composable
fun TranslationPanel(
    translationResult: String,
    replyText: String,
    onReplyTextChange: (String) -> Unit,
    textSize: String,
    onReplySend: () -> Unit,
    modifier: Modifier = Modifier
) {
    val normalSize = when (textSize) {
        "A" -> 24.sp
        "AAA" -> 36.sp
        else -> 30.sp
    }
    val compactSize = when (textSize) {
        "A" -> 21.sp
        "AAA" -> 30.sp
        else -> 24.sp
    }
    val lineHeight = when (textSize) {
        "A" -> 30.sp
        "AAA" -> 42.sp
        else -> 34.sp
    }

    var isFocused by remember { mutableStateOf(false) }
    val borderColor = if (isFocused) PrimaryColor else OutlineVariant.copy(alpha = 0.3f)
    val borderWidth = if (isFocused) 1.5.dp else 1.dp

    Surface(
        modifier = modifier.fillMaxWidth(),
        color = SurfaceContainer.copy(alpha = 0.96f),
        shape = RoundedCornerShape(22.dp),
        border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.28f)),
        shadowElevation = 18.dp
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 18.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Icon(Icons.Default.Translate, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(18.dp))
                Text(
                    text = stringResource(R.string.live_translation),
                    style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                    color = PrimaryColor
                )
            }

            Text(
                text = if (translationResult.isBlank()) stringResource(R.string.start_signing) else translationResult,
                style = MaterialTheme.typography.headlineMedium.copy(
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = if (translationResult.length > 28) compactSize else normalSize,
                    lineHeight = lineHeight
                ),
                color = if (translationResult.isBlank()) OnSurfaceVariant else OnSurface,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.fillMaxWidth()
            )

            Surface(
                color = SurfaceContainerLow,
                shape = RoundedCornerShape(16.dp),
                border = BorderStroke(borderWidth, borderColor)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 54.dp)
                        .padding(start = 12.dp, end = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Default.Keyboard, 
                        contentDescription = null, 
                        tint = if (isFocused) PrimaryColor else SecondaryColor, 
                        modifier = Modifier.size(20.dp)
                    )
                    TextField(
                        value = replyText,
                        onValueChange = onReplyTextChange,
                        modifier = Modifier
                            .weight(1f)
                            .onFocusChanged { isFocused = it.isFocused },
                        placeholder = { Text(stringResource(R.string.reply_placeholder), color = OnSurfaceVariant.copy(alpha = 0.62f)) },
                        singleLine = true,
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = Color.Transparent,
                            unfocusedContainerColor = Color.Transparent,
                            focusedIndicatorColor = Color.Transparent,
                            unfocusedIndicatorColor = Color.Transparent,
                            cursorColor = PrimaryColor,
                            focusedTextColor = OnSurface,
                            unfocusedTextColor = OnSurface
                        )
                    )
                    val sendInteraction = remember { MutableInteractionSource() }
                    val sendPressed by sendInteraction.collectIsPressedAsState()
                    val sendScale by animateFloatAsState(if (sendPressed) 0.9f else 1f, label = "sendScale")
                    IconButton(
                        onClick = onReplySend,
                        enabled = replyText.isNotBlank(),
                        modifier = Modifier.scale(sendScale),
                        interactionSource = sendInteraction
                    ) {
                        Icon(
                            Icons.Default.Send,
                            contentDescription = stringResource(R.string.send_reply),
                            tint = if (replyText.isNotBlank()) PrimaryColor else OnSurfaceVariant.copy(alpha = 0.45f)
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun TrackingStatusChip(isActive: Boolean) {
    Surface(
        color = if (isActive) SuccessColor.copy(alpha = 0.18f) else SurfaceContainer.copy(alpha = 0.82f),
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(1.dp, if (isActive) SuccessColor.copy(alpha = 0.42f) else OutlineVariant.copy(alpha = 0.3f))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 7.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(7.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(if (isActive) SuccessColor else WarningColor, CircleShape)
            )
            Text(
                text = if (isActive) stringResource(R.string.recognition_ready) else stringResource(R.string.recognition_waiting),
                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                color = if (isActive) SuccessColor else OnSurfaceVariant
            )
        }
    }
}

@Composable
fun LanguagePairChip(signShort: String, outputShort: String) {
    Surface(
        color = SurfaceContainer.copy(alpha = 0.82f),
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.26f))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 7.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(signShort, color = PrimaryColor, fontWeight = FontWeight.Bold, fontSize = 12.sp)
            Icon(Icons.Default.ArrowForward, contentDescription = null, tint = OnSurfaceVariant, modifier = Modifier.size(12.dp))
            Text(outputShort, color = OnSurface, fontWeight = FontWeight.Bold, fontSize = 12.sp)
        }
    }
}

@Composable
fun CameraControlButton(icon: ImageVector, contentDescription: String, onClick: () -> Unit) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(if (isPressed) 0.94f else 1f, label = "scale")

    Surface(
        color = SurfaceContainer.copy(alpha = 0.74f),
        shape = CircleShape,
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.18f)),
        shadowElevation = 8.dp,
        modifier = Modifier
            .size(50.dp)
            .scale(scale)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onClick
            )
    ) {
        Box(contentAlignment = Alignment.Center) {
            Icon(icon, contentDescription = contentDescription, tint = OnSurface, modifier = Modifier.size(23.dp))
        }
    }
}

@Composable
fun MetadataBadge(text: String, isGlow: Boolean) {
    val infiniteTransition = rememberInfiniteTransition(label = "badge")
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 0.9f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glow"
    )

    Surface(
        color = PrimaryColor.copy(alpha = 0.1f),
        shape = RoundedCornerShape(8.dp),
        border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.15f)),
        modifier = Modifier.alpha(if (isGlow) glowAlpha else 1f)
    ) {
        Text(
            text = text.uppercase(),
            style = MaterialTheme.typography.labelSmall.copy(
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp
            ),
            color = if (isGlow) OnSurfaceVariant else PrimaryColor,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
        )
    }
}

@Composable
fun FloatingMicButton() {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(if (isPressed) 0.94f else 1f, label = "scale")

    Surface(
        color = PrimaryColor,
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .size(56.dp)
            .scale(scale)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = { /* Mic logic */ }
            ),
        shadowElevation = 8.dp
    ) {
        Box(contentAlignment = Alignment.Center) {
            Icon(Icons.Default.Mic, contentDescription = stringResource(R.string.voice_mic), tint = OnPrimaryFixed)
        }
    }
}

@Composable
fun CameraPreview(onStartCamera: (PreviewView) -> Unit) {
    AndroidView(
        factory = { context ->
            PreviewView(context).apply {
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                onStartCamera(this)
            }
        },
        modifier = Modifier.fillMaxSize()
    )
}

@Composable
fun SkeletonOverlay(handLandmarks: HandLandmarkerResult) {
    Canvas(
        modifier = Modifier.fillMaxSize()
    ) {
        val width = size.width
        val height = size.height

        handLandmarks.landmarks().forEach { handLandmark ->
            // Connections
            HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
                val start = handLandmark[connection.start()]
                val end = handLandmark[connection.end()]

                drawLine(
                    color = SecondaryColor.copy(alpha = 0.7f),
                    start = Offset(start.x() * width, start.y() * height),
                    end = Offset(end.x() * width, end.y() * height),
                    strokeWidth = 2.dp.toPx()
                )
            }

            // Joints with glow
            handLandmark.forEach { landmark ->
                drawCircle(
                    color = PrimaryColor,
                    radius = 3.dp.toPx(),
                    center = Offset(landmark.x() * width, landmark.y() * height)
                )
            }
        }
    }
}
