package com.example.srt_app.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FlipCameraIos
import androidx.compose.material.icons.filled.Keyboard
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Translate
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.srt_app.R
import com.example.srt_app.ui.theme.CharcoalBackground
import com.example.srt_app.ui.theme.GlowingBlue
import com.example.srt_app.ui.theme.GrayText
import com.example.srt_app.ui.theme.PureWhite
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

@Composable
fun SRTScreen(
    translationResult: String,
    onStartCamera: (PreviewView) -> Unit,
    onNavigateToSettings: () -> Unit,
    onCapture: () -> Unit = {},
    onFlipCamera: () -> Unit = {},
    onTypeBackSend: (String) -> Unit = {},
    showSkeleton: Boolean = true,
    handLandmarks: HandLandmarkerResult? = null,
    signLanguage: String = "ASL (American)",
    outputLanguage: String = "English"
) {
    // Extract short name for display (e.g., "ASL (American)" -> "ASL")
    val signShort = signLanguage.split(" ").first()
    val outputShort = outputLanguage.split(" ").first()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(CharcoalBackground)
    ) {
        // 1. Top 70% - Camera Feed & Skeleton Overlay
        Box(
            modifier = Modifier
                .weight(0.7f)
                .fillMaxWidth()
        ) {
            CameraPreview(onStartCamera)

            // Subtle Glowing Skeleton Overlay
            if (showSkeleton && handLandmarks != null) {
                SkeletonOverlay(handLandmarks)
            }

            // Shutter & Flip Buttons
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 24.dp)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Spacer to balance the layout
                Spacer(modifier = Modifier.size(48.dp))

                IconButton(
                    onClick = onCapture,
                    modifier = Modifier
                        .size(72.dp)
                        .background(Color.White.copy(alpha = 0.3f), shape = MaterialTheme.shapes.extraLarge)
                ) {
                    Box(
                        modifier = Modifier
                            .size(60.dp)
                            .background(Color.White, shape = MaterialTheme.shapes.extraLarge)
                    )
                }

                IconButton(
                    onClick = onFlipCamera,
                    modifier = Modifier
                        .padding(start = 16.dp)
                        .size(48.dp)
                        .background(Color.Black.copy(alpha = 0.4f), shape = CircleShape)
                ) {
                    Icon(
                        Icons.Default.FlipCameraIos,
                        contentDescription = stringResource(R.string.flip_camera),
                        tint = Color.White
                    )
                }
            }
            // Top Status Label
            Surface(
                color = Color.Black.copy(alpha = 0.4f),
                modifier = Modifier.align(Alignment.TopCenter).padding(top = 40.dp),
                shape = MaterialTheme.shapes.extraLarge
            ) {
                Text(
                    text = stringResource(R.string.sign_tracking_active),
                    color = GlowingBlue,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp),
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }

        // 2. Bottom 30% - Translation Area
        Column(
            modifier = Modifier
                .weight(0.3f)
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Translated Text
            Text(
                text = if (translationResult.isEmpty()) stringResource(R.string.start_signing) else translationResult,
                color = PureWhite,
                style = MaterialTheme.typography.headlineMedium.copy(
                    fontWeight = FontWeight.Medium,
                    fontSize = 32.sp,
                    lineHeight = 40.sp
                ),
                modifier = Modifier.fillMaxWidth()
            )

            // Bottom Navigation Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = onNavigateToSettings) {
                    Icon(Icons.Default.Settings, contentDescription = stringResource(R.string.settings), tint = GrayText)
                }

                Button(
                    onClick = onNavigateToSettings,
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent),
                    contentPadding = PaddingValues(0.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Translate, contentDescription = stringResource(R.string.language), tint = PrimaryBlue, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("$signShort ${stringResource(R.string.to)} $outputShort", color = PrimaryBlue, fontWeight = FontWeight.SemiBold)
                    }
                }

                var showTypeBackDialog by remember { mutableStateOf(false) }
                IconButton(onClick = { showTypeBackDialog = true }) {
                    Icon(Icons.Default.Keyboard, contentDescription = stringResource(R.string.type_back), tint = GrayText)
                }

                if (showTypeBackDialog) {
                    TypeBackDialog(
                        onDismiss = { showTypeBackDialog = false },
                        onSend = { text ->
                            onTypeBackSend(text)
                            showTypeBackDialog = false
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun TypeBackDialog(onDismiss: () -> Unit, onSend: (String) -> Unit) {
    var text by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.type_back)) },
        text = {
            TextField(
                value = text,
                onValueChange = { text = it },
                placeholder = { Text(stringResource(R.string.type_something)) },
                modifier = Modifier.fillMaxWidth()
            )
        },
        confirmButton = {
            TextButton(onClick = {
                onSend(text)
                onDismiss()
            }) {
                Text(stringResource(R.string.send))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.cancel))
            }
        }
    )
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
        modifier = Modifier
            .fillMaxSize()
    ) {
        val width = size.width
        val height = size.height

        handLandmarks.landmarks().forEach { handLandmark ->
            // Draw connections
            HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
                val start = handLandmark[connection.start()]
                val end = handLandmark[connection.end()]

                drawLine(
                    color = GlowingBlue,
                    start = Offset(start.x() * width, start.y() * height),
                    end = Offset(end.x() * width, end.y() * height),
                    strokeWidth = 3.dp.toPx()
                )
            }

            // Draw joints
            handLandmark.forEach { landmark ->
                drawCircle(
                    color = Color.White,
                    radius = 4.dp.toPx(),
                    center = Offset(landmark.x() * width, landmark.y() * height)
                )
            }
        }
    }
}

private val PrimaryBlue = Color(0xFF8AB4F8)
