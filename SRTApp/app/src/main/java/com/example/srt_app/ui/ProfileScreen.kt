package com.example.srt_app.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.example.srt_app.R
import com.example.srt_app.ui.theme.*

@Composable
fun ProfileScreen(
    onNavigateToTranslator: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateToHistory: () -> Unit,
    onLogout: () -> Unit = {}
) {
    // UI States for Dialogs
    var showEditDialog by remember { mutableStateOf(false) }
    var showSecurityDialog by remember { mutableStateOf(false) }
    var showPrivacyDialog by remember { mutableStateOf(false) }
    var showAchievementDialog by remember { mutableStateOf(false) }
    var selectedBadge by remember { mutableStateOf("") }

    // User Data (Local state for demo)
    var userName by remember { mutableStateOf("Alex Johnson") }
    var userRole by remember { mutableStateOf("Elite Interpreter") }

    Scaffold(
        topBar = {
            ProfileTopBar()
        },
        containerColor = Color(0xFF0E0E0E),
        bottomBar = {
            SenseBottomNavBar(
                selectedTab = 0,
                onNavigateToProfile = {}, // Current
                onNavigateToTranslator = onNavigateToTranslator
            )
        }
    ) { paddingValues ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 24.dp),
            verticalArrangement = Arrangement.spacedBy(32.dp)
        ) {
            item { Spacer(modifier = Modifier.height(16.dp)) }

            // 1. Profile Hero Section
            item {
                HeroSection(
                    name = userName,
                    role = if (userRole == "Elite Interpreter") stringResource(R.string.elite_interpreter) else userRole,
                    joinedDate = stringResource(R.string.joined_date)
                )
            }

            // 2. Achievements Section
            item {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    SectionHeader(stringResource(R.string.header_achievements))
                    Row(
                        modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        AchievementBadge(stringResource(R.string.badge_streak), stringResource(R.string.badge_days, "7"), Icons.Default.Whatshot, PrimaryColor) {
                            selectedBadge = "Streak Expert"
                            showAchievementDialog = true
                        }
                        AchievementBadge(stringResource(R.string.badge_expert), stringResource(R.string.badge_level, "12"), Icons.Default.Psychology, SecondaryColor) {
                            selectedBadge = "Knowledge Master"
                            showAchievementDialog = true
                        }
                        AchievementBadge(stringResource(R.string.badge_helper), stringResource(R.string.badge_sent, "50"), Icons.Default.Handshake, TertiaryColor) {
                            selectedBadge = "Community Helper"
                            showAchievementDialog = true
                        }
                        AchievementBadge(stringResource(R.string.badge_fast), stringResource(R.string.badge_delay, "2s"), Icons.Default.Bolt, Color.White) {
                            selectedBadge = "Speed Demon"
                            showAchievementDialog = true
                        }
                    }
                }
            }

            // 3. Stats & Progress
            item {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    SectionHeader(stringResource(R.string.header_performance))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        ProgressStatCard(
                            modifier = Modifier.weight(1f),
                            label = stringResource(R.string.total_translations),
                            value = "12.4k",
                            progress = 0.85f,
                            icon = Icons.Default.Translate,
                            color = PrimaryColor
                        )
                        ProgressStatCard(
                            modifier = Modifier.weight(1f),
                            label = stringResource(R.string.accuracy),
                            value = "98.4%",
                            progress = 0.98f,
                            icon = Icons.Default.Verified,
                            color = SecondaryColor
                        )
                    }
                }
            }

            // 4. Account Controls
            item {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SectionHeader(stringResource(R.string.header_general))
                    
                    AccountItem(
                        icon = Icons.Default.Settings,
                        label = stringResource(R.string.settings),
                        onClick = onNavigateToSettings
                    )
                    AccountItem(
                        icon = Icons.Default.History,
                        label = stringResource(R.string.history),
                        onClick = onNavigateToHistory
                    )
                    AccountItem(
                        icon = Icons.Default.EditNote,
                        label = stringResource(R.string.edit_profile),
                        onClick = { showEditDialog = true }
                    )
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    SectionHeader(stringResource(R.string.header_legal))
                    
                    AccountItem(
                        icon = Icons.Default.Security,
                        label = stringResource(R.string.security),
                        onClick = { showSecurityDialog = true }
                    )
                    AccountItem(
                        icon = Icons.Default.Policy,
                        label = stringResource(R.string.privacy_policy),
                        onClick = { showPrivacyDialog = true }
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    LogoutItem(onClick = onLogout)
                }
            }

            // 5. Subscription Card
            item {
                SubscriptionCard()
            }

            item { Spacer(modifier = Modifier.height(32.dp)) }
        }
    }

    // Dialogs Implementation
    if (showEditDialog) {
        var tempName by remember { mutableStateOf(userName) }
        var tempRole by remember { mutableStateOf(userRole) }

        AlertDialog(
            onDismissRequest = { showEditDialog = false },
            title = { Text(stringResource(R.string.edit_profile)) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    @Suppress("DEPRECATION")
                    OutlinedTextField(
                        value = tempName,
                        onValueChange = { tempName = it },
                        label = { Text(stringResource(R.string.language)) }, 
                        modifier = Modifier.fillMaxWidth()
                    )
                    @Suppress("DEPRECATION")
                    OutlinedTextField(
                        value = tempRole,
                        onValueChange = { tempRole = it },
                        label = { Text(stringResource(R.string.elite_interpreter)) },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(onClick = {
                    userName = tempName
                    userRole = tempRole
                    showEditDialog = false
                }) {
                    Text(stringResource(R.string.save))
                }
            },
            dismissButton = {
                TextButton(onClick = { showEditDialog = false }) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }

    if (showSecurityDialog) {
        AlertDialog(
            onDismissRequest = { showSecurityDialog = false },
            title = { Text(stringResource(R.string.security)) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(stringResource(R.string.pro_tracking_desc), style = MaterialTheme.typography.bodyMedium)
                    @Suppress("DEPRECATION")
                    Text("• Biometric Lock", color = PrimaryColor)
                }
            },
            confirmButton = {
                Button(onClick = { showSecurityDialog = false }) {
                    Text(stringResource(R.string.save))
                }
            }
        )
    }

    if (showPrivacyDialog) {
        AlertDialog(
            onDismissRequest = { showPrivacyDialog = false },
            title = { Text(stringResource(R.string.privacy_policy)) },
            text = {
                Box(modifier = Modifier.heightIn(max = 300.dp).verticalScroll(rememberScrollState())) {
                    Text(
                        stringResource(R.string.subscription_desc),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            },
            confirmButton = {
                Button(onClick = { showPrivacyDialog = false }) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }

    if (showAchievementDialog) {
        AlertDialog(
            onDismissRequest = { showAchievementDialog = false },
            title = { Text(selectedBadge) },
            text = {
                Text(stringResource(R.string.pro_tracking_title))
            },
            confirmButton = {
                Button(onClick = { showAchievementDialog = false }) {
                    Text(stringResource(R.string.save))
                }
            }
        )
    }
}

@Composable
fun ProfileTopBar() {
    Row(
        modifier = Modifier
            .statusBarsPadding()
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            Icons.Default.MenuOpen, 
            contentDescription = null, 
            tint = PrimaryColor,
            modifier = Modifier.size(28.dp).clickable { }
        )
        
        Text(
            text = "LUMINARY",
            style = MaterialTheme.typography.titleLarge.copy(
                fontWeight = FontWeight.Black,
                letterSpacing = 4.sp,
                fontSize = 20.sp
            ),
            color = Color.White
        )
        
        Surface(
            modifier = Modifier.size(40.dp),
            shape = CircleShape,
            border = BorderStroke(1.dp, PrimaryColor.copy(alpha = 0.2f)),
            color = Color.Transparent
        ) {
            AsyncImage(
                model = "https://lh3.googleusercontent.com/aida-public/AB6AXuAlliXUV5egDcGjBFcd33T2P-NIcYs7GcXLwzsMSssTRysHYteVJocGbuXdx6Md5VLJVkRAl9Qp53ua_7_GOTwF_5I2TzzBgGTsjl1X0-7TCxOLiYwALZWroHIGjalX0bb69EBZRDq22joFFKaSRO2I5y9WThvA-7Xq2OEX78OJv3XnTdCuqMtq7FEmKNMwRlhS4k-_-EMZcrZApKKKh8nDS3CeJO6_33oXENBFZOa3iKvRwmfHZGVyDb9dBoUGQxkGH2PY_T6Ys9k",
                contentDescription = null,
                contentScale = ContentScale.Crop
            )
        }
    }
}

@Composable
fun HeroSection(name: String, role: String, joinedDate: String) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(contentAlignment = Alignment.BottomEnd) {
            // Glow effect
            Box(
                modifier = Modifier
                    .size(128.dp)
                    .background(
                        Brush.radialGradient(
                            colors = listOf(PrimaryColor.copy(alpha = 0.25f), Color.Transparent)
                        ),
                        CircleShape
                    )
            )
            
            Surface(
                modifier = Modifier
                    .size(128.dp)
                    .padding(4.dp),
                shape = CircleShape,
                border = BorderStroke(4.dp, Color(0xFF0E0E0E))
            ) {
                AsyncImage(
                    model = "https://lh3.googleusercontent.com/aida-public/AB6AXuBjGgESx_1GrfHRNU7AK1bz949LQP9QpTghMnLvRbthu0w2YtISanjUQtNGAtkmAZi_ImVYB2KFIClzF0ADT-sCg4Gs26aT-8bqQLwACG8S7BFwl34JyIHBHft3EnrMYa0W0AR75xyZcefC4gCGp36ymdMjBuQSrMYykvhUgdPreWScKgMBeGFCIa8i46_Cqhn1dcr8YSwrwUd8QKGVwvGPjPQXMHvzxpPqAXwgfvvLnB5ko67V2XYjjjVWpy5s4OZPdKkKOBAaWww",
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.clip(CircleShape)
                )
            }
            
            Surface(
                color = PrimaryColor,
                shape = CircleShape,
                modifier = Modifier.offset(x = (-4).dp, y = (-4).dp)
            ) {
                Text(
                    "PRO",
                    color = OnPrimaryFixed,
                    fontWeight = FontWeight.Black,
                    fontSize = 10.sp,
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                    letterSpacing = 1.sp
                )
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Text(name, color = Color.White, style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.ExtraBold)
        Text("$role • $joinedDate", color = OnSurfaceVariant, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium)
    }
}

@Composable
fun AchievementBadge(label: String, value: String, icon: ImageVector, color: Color, onClick: () -> Unit) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(if (isPressed) 0.95f else 1f, label = "scale")

    Surface(
        color = Color(0xFF1A1A1A),
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.dp, if (isPressed) color.copy(alpha = 0.5f) else Color.White.copy(alpha = 0.05f)),
        modifier = Modifier
            .scale(scale)
            .clickable(
                interactionSource = interactionSource,
                indication = LocalIndication.current
            ) { onClick() }
    ) {
        Column(
            modifier = Modifier.padding(16.dp).width(80.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(24.dp))
            Spacer(modifier = Modifier.height(8.dp))
            Text(label, color = OnSurfaceVariant, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            Text(value, color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Black)
        }
    }
}

@Composable
fun ProgressStatCard(modifier: Modifier, label: String, value: String, progress: Float, icon: ImageVector, color: Color) {
    Surface(
        modifier = modifier,
        color = Color(0xFF131313),
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.05f))
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(18.dp))
                Text(value, color = color, fontWeight = FontWeight.Black, fontSize = 14.sp)
            }
            Spacer(modifier = Modifier.height(12.dp))
            Text(label.uppercase(), color = OnSurfaceVariant, fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
            Spacer(modifier = Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth().height(4.dp).clip(CircleShape),
                color = color,
                trackColor = Color.White.copy(alpha = 0.05f)
            )
        }
    }
}

@Composable
fun AccountItem(icon: ImageVector, label: String, onClick: () -> Unit = {}) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val bgColor = if (isPressed) SurfaceContainerHighest else Color.Transparent

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = bgColor,
        shape = RoundedCornerShape(16.dp)
    ) {
        Row(
            modifier = Modifier
                .clickable(
                    interactionSource = interactionSource,
                    indication = LocalIndication.current
                ) { onClick() }
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Surface(
                modifier = Modifier.size(40.dp),
                color = if (isPressed) PrimaryColor.copy(alpha = 0.2f) else SurfaceContainerHigh,
                shape = RoundedCornerShape(12.dp)
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(icon, contentDescription = null, tint = if (isPressed) PrimaryColor else OnSurfaceVariant, modifier = Modifier.size(20.dp))
                }
            }
            Spacer(modifier = Modifier.width(16.dp))
            Text(
                label, 
                color = if (isPressed) PrimaryColor else Color.White, 
                fontWeight = if (isPressed) FontWeight.Bold else FontWeight.SemiBold, 
                modifier = Modifier.weight(1f)
            )
            Icon(
                Icons.Default.ChevronRight, 
                contentDescription = null, 
                tint = if (isPressed) PrimaryColor else OnSurfaceVariant
            )
        }
    }
}

@Composable
fun LogoutItem(onClick: () -> Unit) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .clickable(
                interactionSource = interactionSource,
                indication = LocalIndication.current
            ) { onClick() }
            .background(if (isPressed) Color(0xFFFF716C).copy(alpha = 0.15f) else Color(0xFFFF716C).copy(alpha = 0.05f))
            .padding(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Surface(
            modifier = Modifier.size(40.dp),
            color = Color(0xFFFF716C).copy(alpha = 0.2f),
            shape = RoundedCornerShape(12.dp)
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(Icons.Default.Logout, contentDescription = null, tint = Color(0xFFFF716C), modifier = Modifier.size(20.dp))
            }
        }
        Spacer(modifier = Modifier.width(16.dp))
        Text(stringResource(R.string.logout), color = Color(0xFFFF716C), fontWeight = FontWeight.Bold)
    }
}

@Composable
fun SubscriptionCard() {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(if (isPressed) 0.98f else 1f, label = "scale")

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .scale(scale)
            .clip(RoundedCornerShape(24.dp))
            .background(
                Brush.linearGradient(
                    colors = if (isPressed) 
                        listOf(PrimaryColor.copy(alpha = 0.2f), PrimaryColor.copy(alpha = 0.05f))
                    else 
                        listOf(PrimaryColor.copy(alpha = 0.1f), Color.Transparent)
                )
            )
            .border(1.dp, PrimaryColor.copy(alpha = if (isPressed) 0.3f else 0.1f), RoundedCornerShape(24.dp))
            .clickable(
                interactionSource = interactionSource,
                indication = null // Custom scale effect
            ) { /* Navigate */ }
            .padding(24.dp)
    ) {
        Icon(
            Icons.Default.AutoAwesome,
            contentDescription = null,
            tint = PrimaryColor.copy(alpha = if (isPressed) 0.2f else 0.1f),
            modifier = Modifier.size(96.dp).align(Alignment.TopEnd).offset(x = 16.dp, y = (-16).dp)
        )
        
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            @Suppress("DEPRECATION")
            Text(stringResource(R.string.subscription_title), color = Color.White, fontWeight = FontWeight.Bold, fontSize = 20.sp)
            @Suppress("DEPRECATION")
            Text(
                stringResource(R.string.subscription_desc),
                color = OnSurfaceVariant,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.fillMaxWidth(0.8f)
            )
            Button(
                onClick = { },
                colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.padding(top = 8.dp)
            ) {
                Text(stringResource(R.string.manage_subscription).uppercase(), fontWeight = FontWeight.Bold, fontSize = 12.sp)
            }
        }
    }
}

@Composable
fun SectionHeader(title: String) {
    Text(
        text = title.uppercase(),
        style = MaterialTheme.typography.labelSmall.copy(
            fontWeight = FontWeight.Bold,
            letterSpacing = 2.sp
        ),
        color = OnSurfaceVariant,
        modifier = Modifier.padding(start = 8.dp, bottom = 8.dp)
    )
}
