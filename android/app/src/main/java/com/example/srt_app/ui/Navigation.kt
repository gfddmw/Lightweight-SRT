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
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.runtime.Composable
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

@Composable
fun SenseInputField(
    value: String, 
    onValueChange: (String) -> Unit, 
    label: String, 
    icon: ImageVector,
    visualTransformation: VisualTransformation = VisualTransformation.None
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(text = label.uppercase(), style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 1.sp, color = OnSurfaceVariant.copy(alpha = 0.7f)), modifier = Modifier.padding(start = 4.dp))
        Surface(color = SurfaceDim, shape = RoundedCornerShape(16.dp), border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.3f))) {
            Row(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Icon(icon, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(20.dp))
                BasicTextField(
                    value = value, 
                    onValueChange = onValueChange, 
                    textStyle = MaterialTheme.typography.bodyLarge.copy(color = Color.White), 
                    cursorBrush = SolidColor(PrimaryColor), 
                    modifier = Modifier.weight(1f),
                    visualTransformation = visualTransformation
                )
            }
        }
    }
}
