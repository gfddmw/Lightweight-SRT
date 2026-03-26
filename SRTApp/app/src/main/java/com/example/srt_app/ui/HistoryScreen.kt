package com.example.srt_app.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.example.srt_app.R
import com.example.srt_app.ui.theme.*

@Composable
fun HistoryScreen(
    onNavigateToTranslator: () -> Unit,
    onNavigateToProfile: () -> Unit,
    onClearAll: () -> Unit = {}
) {
    var searchQuery by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            ProfileTopBar() 
        },
        containerColor = Color(0xFF0E0E0E),
        bottomBar = {
            SenseBottomNavBar(
                selectedTab = 0, // Highlight profile tab since it's a sub-page
                onNavigateToProfile = onNavigateToProfile,
                onNavigateToTranslator = onNavigateToTranslator
            )
        }
    ) { paddingValues ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 24.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp)
        ) {
            // 1. Search Section
            item {
                Spacer(modifier = Modifier.height(16.dp))
                SearchField(
                    query = searchQuery,
                    onQueryChange = { searchQuery = it }
                )
            }

            // 2. Section Header
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        stringResource(R.string.recent_translations).uppercase(),
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 2.sp
                        ),
                        color = OnSurfaceVariant
                    )
                    Text(
                        stringResource(R.string.clear_all).uppercase(),
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        ),
                        color = PrimaryColor,
                        modifier = Modifier.clickable { onClearAll() }
                    )
                }
            }

            // 3. Highlight Highlight Card
            item {
                HighlightHistoryCard(
                    languagePair = "ASL TO ENGLISH",
                    time = "2 MINS AGO",
                    result = "\"Where is the nearest medical center?\"",
                    precision = "98.4%",
                    thumbnailUrl = "https://lh3.googleusercontent.com/aida-public/AB6AXuAEtnBH-RYAu9pGrt1ZP539_U6WrQni4h7_WKLo10LWeZEEDFssTufcl6drEp_zLRtqNfWYN-VUIseGN7BJjVapudgw2AFswKxR_MPM8TBW8g0OW7wBkLVAvwaEI-zAFSR0dyM9lsZ1hWzQmx0vAA-2KowAWVPdJHAMMbMNdxF_VEP4UikvV2wjMH1JKmNTgMLZsKHeL7J5s8GdbncS_EyaQ0m8_nlXvQI3g-UgH-kNnptEqBYql5WLaSu6rYWP0OOu3TOYtnFDF5s"
                )
            }

            // 4. Secondary Cards Grid
            item {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    SecondaryHistoryCard(
                        modifier = Modifier.weight(1f),
                        languagePair = "BSL TO ENGLISH",
                        time = "Yesterday",
                        result = "Thank you for your assistance today, I really appreciate it.",
                        isGlass = true
                    )
                    SecondaryHistoryCard(
                        modifier = Modifier.weight(1f),
                        languagePair = "ASL TO ENGLISH",
                        time = "Oct 24",
                        result = "Meeting at 5 PM near the transit station.",
                        isGlass = false
                    )
                }
            }

            // 5. Subtle Data Rows
            item {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    HistoryListItem(
                        text = "How much for the bus ticket?",
                        subtitle = "ASL • Oct 23, 14:20"
                    )
                    HistoryListItem(
                        text = "I need some water please.",
                        subtitle = "ASL • Oct 22, 09:15"
                    )
                }
            }

            item { Spacer(modifier = Modifier.height(32.dp)) }
        }
    }
}

@Composable
fun SearchField(query: String, onQueryChange: (String) -> Unit) {
    TextField(
        value = query,
        onValueChange = onQueryChange,
        modifier = Modifier.fillMaxWidth(),
        placeholder = { 
            Text(
                stringResource(R.string.search_translations), 
                color = OnSurfaceVariant.copy(alpha = 0.5f),
                fontSize = 16.sp
            ) 
        },
        leadingIcon = { Icon(Icons.Default.Search, contentDescription = null, tint = OnSurfaceVariant) },
        colors = TextFieldDefaults.colors(
            focusedContainerColor = SurfaceContainerLow,
            unfocusedContainerColor = SurfaceContainerLow,
            focusedIndicatorColor = PrimaryColor,
            unfocusedIndicatorColor = Color.White.copy(alpha = 0.1f),
            cursorColor = PrimaryColor
        ),
        singleLine = true
    )
}

@Composable
fun HighlightHistoryCard(
    languagePair: String,
    time: String,
    result: String,
    precision: String,
    thumbnailUrl: String
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = SurfaceContainerLow,
        shape = RoundedCornerShape(24.dp),
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.05f))
    ) {
        Column(modifier = Modifier.padding(24.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Surface(
                        color = PrimaryColor.copy(alpha = 0.1f),
                        shape = CircleShape
                    ) {
                        Text(
                            languagePair,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
                            color = PrimaryColor,
                            fontSize = 10.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                    }
                    Text(time, color = OnSurfaceVariant, fontSize = 10.sp, fontWeight = FontWeight.Medium, letterSpacing = 1.sp)
                }
                Icon(Icons.Default.History, contentDescription = null, tint = OnSurfaceVariant, modifier = Modifier.size(20.dp))
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            Text(
                result,
                color = Color.White,
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                lineHeight = 36.sp,
                letterSpacing = (-0.5).sp
            )
            
            Spacer(modifier = Modifier.height(32.dp))
            
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Surface(
                    modifier = Modifier.size(48.dp),
                    shape = RoundedCornerShape(12.dp),
                    color = Color.Black.copy(alpha = 0.4f)
                ) {
                    AsyncImage(
                        model = thumbnailUrl,
                        contentDescription = null,
                        contentScale = ContentScale.Crop
                    )
                }
                Column {
                    Text(
                        stringResource(R.string.confidence_score).uppercase(),
                        color = OnSurfaceVariant,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp
                    )
                    Text("$precision ${stringResource(R.string.precision)}", color = PrimaryColor, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                }
            }
        }
    }
}

@Composable
fun SecondaryHistoryCard(
    modifier: Modifier,
    languagePair: String,
    time: String,
    result: String,
    isGlass: Boolean
) {
    Surface(
        modifier = modifier.heightIn(min = 160.dp),
        color = if (isGlass) PrimaryColor.copy(alpha = 0.08f) else SurfaceContainerLow,
        shape = RoundedCornerShape(24.dp),
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.05f))
    ) {
        Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.SpaceBetween) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    languagePair,
                    color = if (isGlass) SecondaryColor else PrimaryColor,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
                Text(time, color = OnSurfaceVariant, fontSize = 10.sp, fontWeight = FontWeight.Medium)
            }
            
            Text(
                result,
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                lineHeight = 22.sp,
                modifier = Modifier.padding(top = 16.dp)
            )
        }
    }
}

@Composable
fun HistoryListItem(text: String, subtitle: String) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = Color(0xFF000000).copy(alpha = 0.2f),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, Color.White.copy(alpha = 0.03f))
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(text, color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Medium)
                Text(subtitle.uppercase(), color = OnSurfaceVariant, fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp, modifier = Modifier.padding(top = 4.dp))
            }
            Icon(Icons.Default.ChevronRight, contentDescription = null, tint = OnSurfaceVariant, modifier = Modifier.size(16.dp))
        }
    }
}
