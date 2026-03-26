package com.example.srt_app.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Translate
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
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
            .clip(RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)),
        color = SurfaceContainer,
        border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.15f)),
        shadowElevation = 16.dp
    ) {
        Row(
            modifier = Modifier
                .navigationBarsPadding()
                .padding(horizontal = 24.dp, vertical = 8.dp)
                .fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceAround,
            verticalAlignment = Alignment.CenterVertically
        ) {
            NavItem(
                icon = Icons.Default.Person,
                label = stringResource(R.string.profile),
                isActive = selectedTab == 0,
                onClick = onNavigateToProfile
            )
            NavItem(
                icon = Icons.Default.Translate,
                label = stringResource(R.string.translator),
                isActive = selectedTab == 1,
                onClick = onNavigateToTranslator
            )
        }
    }
}

@Composable
fun NavItem(icon: ImageVector, label: String, isActive: Boolean, onClick: () -> Unit) {
    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .background(if (isActive) PrimaryColor.copy(alpha = 0.1f) else Color.Transparent)
            .clickable { onClick() }
            .padding(horizontal = 12.dp, vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = if (isActive) PrimaryColor else OnSurfaceVariant,
            modifier = Modifier.size(26.dp)
        )
        Spacer(modifier = Modifier.height(2.dp))
        @Suppress("DEPRECATION")
        Text(
            text = label.uppercase(),
            style = MaterialTheme.typography.labelSmall.copy(
                fontSize = 11.sp,
                fontWeight = FontWeight.ExtraBold,
                letterSpacing = 0.5.sp,
                color = if (isActive) PrimaryColor else Color.White
            )
        )
    }
}
