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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
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
    outputLanguage: String = "en"
) {
    val signShort = signLanguage
    val outputShort = if (outputLanguage == "en") "EN" else "ZH"

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(SurfaceDim)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            // 1. Camera Section (approx 80% height)
            Box(
                modifier = Modifier
                    .weight(0.8f)
                    .fillMaxWidth()
            ) {
                CameraPreview(onStartCamera)

                // Skeleton Overlay
                if (showSkeleton && handLandmarks != null) {
                    SkeletonOverlay(handLandmarks)
                }

                // Video Gradient Overlay (Dark bottom)
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(
                            brush = Brush.verticalGradient(
                                colors = listOf(Color.Transparent, SurfaceDim.copy(alpha = 0.8f)),
                                startY = 400f
                            )
                        )
                )

                // Technical Metadata Badges (Top Right)
                Column(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(top = 100.dp, end = 24.dp),
                    horizontalAlignment = Alignment.End,
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    MetadataBadge(text = stringResource(R.string.sign_tracking_active), isGlow = true)
                    
                    // Quick Language Switcher Badge (Visual only)
                    Surface(
                        color = Color.Black.copy(alpha = 0.4f),
                        shape = RoundedCornerShape(12.dp),
                        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.1f))
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Text(signShort, color = PrimaryColor, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                            Icon(Icons.Default.ArrowForward, contentDescription = null, tint = OnSurfaceVariant, modifier = Modifier.size(12.dp))
                            Text(outputShort, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                        }
                    }
                }
                
                // Camera controls
                Column(
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(bottom = 16.dp, end = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    IconButton(
                        onClick = onCapture,
                        modifier = Modifier
                            .background(Color.Black.copy(alpha = 0.3f), CircleShape)
                    ) {
                        Icon(Icons.Default.CameraAlt, contentDescription = "Capture", tint = Color.White)
                    }
                    
                    IconButton(
                        onClick = onFlipCamera,
                        modifier = Modifier
                            .background(Color.Black.copy(alpha = 0.3f), CircleShape)
                    ) {
                        Icon(Icons.Default.FlipCameraIos, contentDescription = stringResource(R.string.flip_camera), tint = Color.White)
                    }
                }
            }

            // 2. Translation Section (approx 20% height)
            Box(
                modifier = Modifier
                    .weight(0.2f)
                    .fillMaxWidth()
                    .background(SurfaceContainerLow)
                    .padding(horizontal = 24.dp, vertical = 16.dp)
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = stringResource(R.string.live_translation).uppercase(),
                            style = MaterialTheme.typography.labelSmall,
                            color = PrimaryColor,
                            letterSpacing = 2.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Box(modifier = Modifier.fillMaxWidth()) {
                        Text(
                            text = if (translationResult.isEmpty()) stringResource(R.string.start_signing) else translationResult,
                            style = MaterialTheme.typography.headlineLarge.copy(
                                fontWeight = FontWeight.Bold,
                                fontSize = 32.sp,
                                lineHeight = 40.sp
                            ),
                            color = OnSurface,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }

                    Spacer(modifier = Modifier.weight(1f))

                    // Quick Actions Footer (Placeholder for real-time status)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = stringResource(R.string.sign_tracking_active),
                            style = MaterialTheme.typography.bodySmall,
                            color = OnSurfaceVariant.copy(alpha = 0.4f),
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
            
            // Padding for Bottom Nav
            Spacer(modifier = Modifier.height(80.dp))
        }

        // 3. Floating Top Bar (Floating on top of everything)
        SenseTopBar(
            onSettingsClick = onNavigateToSettings,
            modifier = Modifier.align(Alignment.TopCenter)
        )

        // 4. Bottom Navigation Bar
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
        color = SurfaceDim.copy(alpha = 0.8f), // Semi-transparent background like HTML
        modifier = modifier.fillMaxWidth()
    ) {
        Box(
            modifier = Modifier
                .statusBarsPadding()
                .padding(horizontal = 24.dp, vertical = 16.dp)
                .fillMaxWidth(),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = stringResource(R.string.app_name).uppercase(),
                style = MaterialTheme.typography.titleLarge.copy(
                    fontWeight = FontWeight.Black,
                    letterSpacing = 4.sp,
                    fontSize = 20.sp
                ),
                color = PrimaryColor
            )
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
    Surface(
        color = PrimaryColor,
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .size(56.dp)
            .clickable { /* Mic logic */ },
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
