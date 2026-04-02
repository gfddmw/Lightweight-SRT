package com.example.srt_app.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
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
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.example.srt_app.R
import com.example.srt_app.ui.theme.*
import kotlinx.coroutines.launch

@Composable
fun ProfileScreen(
    onNavigateToTranslator: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onLogout: () -> Unit = {},
    userSettings: com.example.srt_app.utils.UserSettings,
    onUpdateProfile: (String, String) -> Unit = { _, _ -> }
) {
    // Drawer States
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    // UI States for Dialogs
    var showEditDialog by remember { mutableStateOf(false) }
    var showSecurityDialog by remember { mutableStateOf(false) }
    var showPrivacyDialog by remember { mutableStateOf(false) }
    var showAchievementDialog by remember { mutableStateOf(false) }
    var selectedBadge by remember { mutableStateOf("") }

    // User Data
    val userName = userSettings.userName
    val userRole = userSettings.userRole

    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = true,
        drawerContent = {
            ModalDrawerSheet(
                drawerContainerColor = SurfaceContainer,
                drawerContentColor = OnSurface,
                modifier = Modifier.width(300.dp).fillMaxHeight(),
                drawerShape = RoundedCornerShape(topEnd = 24.dp, bottomEnd = 24.dp)
            ) {
                ProfileDrawerContent(
                    userName = userName,
                    onSettingsClick = {
                        scope.launch { drawerState.close() }
                        onNavigateToSettings()
                    },
                    onEditProfileClick = {
                        scope.launch { drawerState.close() }
                        showEditDialog = true
                    },
                    onSecurityClick = {
                        scope.launch { drawerState.close() }
                        showSecurityDialog = true
                    },
                    onPrivacyClick = {
                        scope.launch { drawerState.close() }
                        showPrivacyDialog = true
                    },
                    onLogoutClick = {
                        scope.launch { drawerState.close() }
                        onLogout()
                    }
                )
            }
        }
    ) {
        Scaffold(
            topBar = {
                ProfileTopBar(onMenuClick = { scope.launch { drawerState.open() } })
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
                            AchievementBadge(stringResource(R.string.badge_streak), stringResource(R.string.badge_days, userSettings.streakDays.toString()), Icons.Default.Whatshot, PrimaryColor) {
                                selectedBadge = "Streak Expert"
                                showAchievementDialog = true
                            }
                            AchievementBadge(stringResource(R.string.badge_expert), stringResource(R.string.badge_level, userSettings.expertLevel.toString()), Icons.Default.Psychology, SecondaryColor) {
                                selectedBadge = "Knowledge Master"
                                showAchievementDialog = true
                            }
                            AchievementBadge(stringResource(R.string.badge_helper), stringResource(R.string.badge_sent, userSettings.helpedCount.toString()), Icons.Default.Handshake, TertiaryColor) {
                                selectedBadge = "Community Helper"
                                showAchievementDialog = true
                            }
                            AchievementBadge(stringResource(R.string.badge_fast), stringResource(R.string.badge_delay, "${userSettings.fastestDelay}s"), Icons.Default.Bolt, Color.White) {
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
                                value = if (userSettings.totalTranslations >= 1000) "%.1fk".format(userSettings.totalTranslations / 1000.0) else userSettings.totalTranslations.toString(),
                                progress = (userSettings.totalTranslations / 1000f).coerceIn(0f, 1f), // Normalized to 1k target for demo
                                icon = Icons.Default.Translate,
                                color = PrimaryColor
                            )
                            ProgressStatCard(
                                modifier = Modifier.weight(1f),
                                label = stringResource(R.string.accuracy),
                                value = "%.1f%%".format(userSettings.accuracy * 100),
                                progress = userSettings.accuracy,
                                icon = Icons.Default.Verified,
                                color = SecondaryColor
                            )
                        }
                    }
                }

                // 4. Subscription Card (Promoted here now as it's more visible)
                item {
                    SubscriptionCard()
                }

                item { Spacer(modifier = Modifier.height(32.dp)) }
            }
        }
    }

    // Dialogs Implementation (Edit, Security, Privacy, Achievement)
    if (showEditDialog) {
        var tempName by remember { mutableStateOf(userSettings.userName) }
        var tempRole by remember { mutableStateOf(userSettings.userRole) }

        androidx.compose.ui.window.Dialog(onDismissRequest = { showEditDialog = false }) {
            Surface(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                shape = RoundedCornerShape(28.dp),
                color = SurfaceContainer,
                border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.2f)),
                shadowElevation = 24.dp
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    Text(
                        text = stringResource(R.string.edit_profile).uppercase(),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp, color = PrimaryColor)
                    )

                    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        SenseInputField(value = tempName, onValueChange = { tempName = it }, label = stringResource(R.string.language), icon = Icons.Default.Person)
                        SenseInputField(value = tempRole, onValueChange = { tempRole = it }, label = stringResource(R.string.elite_interpreter), icon = Icons.Default.Stars)
                    }

                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        TextButton(onClick = { showEditDialog = false }, modifier = Modifier.weight(1f)) {
                            Text(stringResource(R.string.cancel), color = OnSurfaceVariant)
                        }
                        Button(
                            onClick = { onUpdateProfile(tempName, tempRole); showEditDialog = false },
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Text(stringResource(R.string.save), color = OnPrimaryFixed, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }
    }

    if (showSecurityDialog) {
        androidx.compose.ui.window.Dialog(onDismissRequest = { showSecurityDialog = false }) {
            Surface(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                shape = RoundedCornerShape(28.dp),
                color = SurfaceContainer,
                border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.2f))
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    Text(
                        text = stringResource(R.string.security).uppercase(),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp, color = PrimaryColor)
                    )
                    
                    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text(stringResource(R.string.pro_tracking_desc), style = MaterialTheme.typography.bodyMedium, color = OnSurfaceVariant)
                        Surface(
                            color = SurfaceDim,
                            shape = RoundedCornerShape(16.dp),
                            border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.1f))
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(16.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                    Icon(Icons.Default.Fingerprint, contentDescription = null, tint = PrimaryColor)
                                    Text("Biometric Lock", color = Color.White, fontWeight = FontWeight.SemiBold)
                                }
                                Switch(checked = true, onCheckedChange = {}, colors = SwitchDefaults.colors(checkedThumbColor = PrimaryColor))
                            }
                        }
                    }

                    Button(
                        onClick = { showSecurityDialog = false },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Text(stringResource(R.string.save), color = OnPrimaryFixed, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }

    if (showPrivacyDialog) {
        androidx.compose.ui.window.Dialog(onDismissRequest = { showPrivacyDialog = false }) {
            Surface(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                shape = RoundedCornerShape(28.dp),
                color = SurfaceContainer,
                border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.2f))
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    Text(
                        text = stringResource(R.string.privacy_policy).uppercase(),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp, color = PrimaryColor)
                    )
                    
                    Box(modifier = Modifier.heightIn(max = 240.dp).verticalScroll(rememberScrollState())) {
                        Text(stringResource(R.string.subscription_desc), style = MaterialTheme.typography.bodySmall, color = OnSurfaceVariant, lineHeight = 20.sp)
                    }

                    Button(
                        onClick = { showPrivacyDialog = false },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Text(stringResource(R.string.cancel), color = OnPrimaryFixed, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }

    if (showAchievementDialog) {
        androidx.compose.ui.window.Dialog(onDismissRequest = { showAchievementDialog = false }) {
            Surface(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                shape = RoundedCornerShape(28.dp),
                color = SurfaceContainer,
                border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.2f))
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    Icon(Icons.Default.WorkspacePremium, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(64.dp))
                    Text(text = selectedBadge.uppercase(), style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp, color = PrimaryColor))
                    Text(stringResource(R.string.pro_tracking_title), style = MaterialTheme.typography.bodyMedium, color = OnSurfaceVariant, textAlign = TextAlign.Center)
                    Button(onClick = { showAchievementDialog = false }, modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor), shape = RoundedCornerShape(16.dp)) {
                        Text(stringResource(R.string.save), color = OnPrimaryFixed, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

@Composable
fun ProfileDrawerContent(
    userName: String,
    onSettingsClick: () -> Unit,
    onEditProfileClick: () -> Unit,
    onSecurityClick: () -> Unit,
    onPrivacyClick: () -> Unit,
    onLogoutClick: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp)
    ) {
        // Drawer Header
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(bottom = 32.dp, top = 16.dp)
        ) {
            Surface(
                modifier = Modifier.size(48.dp),
                shape = CircleShape,
                color = PrimaryColor.copy(alpha = 0.1f),
                border = BorderStroke(1.dp, PrimaryColor.copy(alpha = 0.2f))
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(Icons.Default.Person, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(24.dp))
                }
            }
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text("LUMINARY", style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp), color = PrimaryColor)
                Text(userName, style = MaterialTheme.typography.bodyMedium, color = Color.White)
            }
        }

        Divider(color = Color.White.copy(alpha = 0.05f), modifier = Modifier.padding(bottom = 24.dp))

        // Navigation Items
        Text(stringResource(R.string.header_general).uppercase(), style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 2.sp), color = OnSurfaceVariant, modifier = Modifier.padding(bottom = 16.dp))
        
        AccountItem(icon = Icons.Default.Settings, label = stringResource(R.string.settings), onClick = onSettingsClick)
        AccountItem(icon = Icons.Default.EditNote, label = stringResource(R.string.edit_profile), onClick = onEditProfileClick)

        Spacer(modifier = Modifier.height(32.dp))

        Text(stringResource(R.string.header_legal).uppercase(), style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 2.sp), color = OnSurfaceVariant, modifier = Modifier.padding(bottom = 16.dp))
        
        AccountItem(icon = Icons.Default.Security, label = stringResource(R.string.security), onClick = onSecurityClick)
        AccountItem(icon = Icons.Default.Policy, label = stringResource(R.string.privacy_policy), onClick = onPrivacyClick)

        Spacer(modifier = Modifier.weight(1f))

        LogoutItem(onClick = onLogoutClick)
    }
}

@Composable
fun ProfileTopBar(onMenuClick: () -> Unit) {
    Row(
        modifier = Modifier.statusBarsPadding().fillMaxWidth().padding(horizontal = 24.dp, vertical = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            Icons.Default.MenuOpen, 
            contentDescription = "Menu", 
            tint = PrimaryColor,
            modifier = Modifier.size(28.dp).clickable { onMenuClick() }
        )
        
        Text(
            text = "LUMINARY",
            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Black, letterSpacing = 4.sp, fontSize = 20.sp),
            color = Color.White
        )
        
        Surface(modifier = Modifier.size(40.dp), shape = CircleShape, border = BorderStroke(1.dp, PrimaryColor.copy(alpha = 0.2f)), color = Color.Transparent) {
            AsyncImage(model = "https://lh3.googleusercontent.com/aida-public/AB6AXuAlliXUV5egDcGjBFcd33T2P-NIcYs7GcXLwzsMSssTRysHYteVJocGbuXdx6Md5VLJVkRAl9Qp53ua_7_GOTwF_5I2TzzBgGTsjl1X0-7TCxOLiYwALZWroHIGjalX0bb69EBZRDq22joFFKaSRO2I5y9WThvA-7Xq2OEX78OJv3XnTdCuqMtq7FEmKNMwRlhS4k-_-EMZcrZApKKKh8nDS3CeJO6_33oXENBFZOa3iKvRwmfHZGVyDb9dBoUGQxkGH2PY_T6Ys9k", contentDescription = null, contentScale = ContentScale.Crop)
        }
    }
}

@Composable
fun HeroSection(name: String, role: String, joinedDate: String) {
    Column(modifier = Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
        Box(contentAlignment = Alignment.BottomEnd) {
            Box(modifier = Modifier.size(128.dp).background(Brush.radialGradient(colors = listOf(PrimaryColor.copy(alpha = 0.25f), Color.Transparent)), CircleShape))
            Surface(modifier = Modifier.size(128.dp).padding(4.dp), shape = CircleShape, border = BorderStroke(4.dp, Color(0xFF0E0E0E))) {
                AsyncImage(model = "https://lh3.googleusercontent.com/aida-public/AB6AXuBjGgESx_1GrfHRNU7AK1bz949LQP9QpTghMnLvRbthu0w2YtISanjUQtNGAtkmAZi_ImVYB2KFIClzF0ADT-sCg4Gs26aT-8bqQLwACG8S7BFwl34JyIHBHft3EnrMYa0W0AR75xyZcefC4gCGp36ymdMjBuQSrMYykvhUgdPreWScKgMBeGFCIa8i46_Cqhn1dcr8YSwrwUd8QKGVwvGPjPQXMHvzxpPqAXwgfvvLnB5ko67V2XYjjjVWpy5s4OZPdKkKOBAaWww", contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.clip(CircleShape))
            }
            Surface(color = PrimaryColor, shape = CircleShape, modifier = Modifier.offset(x = (-4).dp, y = (-4).dp)) {
                Text("PRO", color = OnPrimaryFixed, fontWeight = FontWeight.Black, fontSize = 10.sp, modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp), letterSpacing = 1.sp)
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
                indication = null,
                onClick = onClick
            )
    ) {
        Column(modifier = Modifier.padding(16.dp).width(80.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(24.dp))
            Spacer(modifier = Modifier.height(8.dp))
            Text(label, color = OnSurfaceVariant, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            Text(value, color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Black)
        }
    }
}

@Composable
fun ProgressStatCard(modifier: Modifier, label: String, value: String, progress: Float, icon: ImageVector, color: Color) {
    Surface(modifier = modifier, color = Color(0xFF131313), shape = RoundedCornerShape(20.dp), border = BorderStroke(1.dp, Color.White.copy(alpha = 0.05f))) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(18.dp))
                Text(value, color = color, fontWeight = FontWeight.Black, fontSize = 14.sp)
            }
            Spacer(modifier = Modifier.height(12.dp))
            Text(label.uppercase(), color = OnSurfaceVariant, fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
            Spacer(modifier = Modifier.height(8.dp))
            LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().height(4.dp).clip(CircleShape), color = color, trackColor = Color.White.copy(alpha = 0.05f))
        }
    }
}

@Composable
fun AccountItem(icon: ImageVector, label: String, onClick: () -> Unit = {}) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val bgColor = if (isPressed) SurfaceContainerHighest else Color.Transparent
    Surface(modifier = Modifier.fillMaxWidth(), color = bgColor, shape = RoundedCornerShape(16.dp)) {
        Row(
            modifier = Modifier
                .clickable(
                    interactionSource = interactionSource, 
                    indication = null,
                    onClick = onClick
                )
                .padding(12.dp), 
            verticalAlignment = Alignment.CenterVertically
        ) {
            Surface(modifier = Modifier.size(40.dp), color = if (isPressed) PrimaryColor.copy(alpha = 0.2f) else SurfaceContainerHigh, shape = RoundedCornerShape(12.dp)) {
                Box(contentAlignment = Alignment.Center) { Icon(icon, contentDescription = null, tint = if (isPressed) PrimaryColor else OnSurfaceVariant, modifier = Modifier.size(20.dp)) }
            }
            Spacer(modifier = Modifier.width(16.dp))
            Text(label, color = if (isPressed) PrimaryColor else Color.White, fontWeight = if (isPressed) FontWeight.Bold else FontWeight.SemiBold, modifier = Modifier.weight(1f))
            Icon(Icons.Default.ChevronRight, contentDescription = null, tint = if (isPressed) PrimaryColor else OnSurfaceVariant)
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
                indication = null,
                onClick = onClick
            )
            .background(if (isPressed) Color(0xFFFF716C).copy(alpha = 0.15f) else Color(0xFFFF716C).copy(alpha = 0.05f))
            .padding(12.dp), 
        verticalAlignment = Alignment.CenterVertically
    ) {
        Surface(modifier = Modifier.size(40.dp), color = Color(0xFFFF716C).copy(alpha = 0.2f), shape = RoundedCornerShape(12.dp)) {
            Box(contentAlignment = Alignment.Center) { Icon(Icons.Default.Logout, contentDescription = null, tint = Color(0xFFFF716C), modifier = Modifier.size(20.dp)) }
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
            .background(Brush.linearGradient(colors = if (isPressed) listOf(PrimaryColor.copy(alpha = 0.2f), PrimaryColor.copy(alpha = 0.05f)) else listOf(PrimaryColor.copy(alpha = 0.1f), Color.Transparent)))
            .border(1.dp, PrimaryColor.copy(alpha = if (isPressed) 0.3f else 0.1f), RoundedCornerShape(24.dp))
            .clickable(
                interactionSource = interactionSource, 
                indication = null,
                onClick = {} 
            )
            .padding(24.dp)
    ) {
        Icon(Icons.Default.AutoAwesome, contentDescription = null, tint = PrimaryColor.copy(alpha = if (isPressed) 0.2f else 0.1f), modifier = Modifier.size(96.dp).align(Alignment.TopEnd).offset(x = 16.dp, y = (-16).dp))
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            @Suppress("DEPRECATION")
            Text(stringResource(R.string.subscription_title), color = Color.White, fontWeight = FontWeight.Bold, fontSize = 20.sp)
            @Suppress("DEPRECATION")
            Text(stringResource(R.string.subscription_desc), color = OnSurfaceVariant, style = MaterialTheme.typography.bodySmall, modifier = Modifier.fillMaxWidth(0.8f))
            Button(onClick = { }, colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor), shape = RoundedCornerShape(12.dp), modifier = Modifier.padding(top = 8.dp)) {
                Text(stringResource(R.string.manage_subscription).uppercase(), fontWeight = FontWeight.Bold, fontSize = 12.sp)
            }
        }
    }
}

@Composable
fun SectionHeader(title: String) {
    Text(text = title.uppercase(), style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 2.sp), color = OnSurfaceVariant, modifier = Modifier.padding(start = 8.dp, bottom = 8.dp))
}
