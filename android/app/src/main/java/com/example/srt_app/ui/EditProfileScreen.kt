package com.example.srt_app.ui

import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.example.srt_app.R
import com.example.srt_app.data.UserRepository
import com.example.srt_app.ui.theme.*
import com.example.srt_app.utils.OSSManager
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EditProfileScreen(
    repository: UserRepository,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val ossManager = remember { OSSManager(context) }
    
    val userProfile by repository.userProfile.collectAsState()
    
    var nickname by remember { mutableStateOf(userProfile.nickname) }
    var description by remember { mutableStateOf(userProfile.description) }
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var isUploading by remember { mutableStateOf(false) }

    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        selectedImageUri = uri
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.edit_profile).uppercase(), style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = SurfaceDim,
                    titleContentColor = Color.White,
                    navigationIconContentColor = PrimaryColor
                )
            )
        },
        containerColor = SurfaceDim
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(24.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(32.dp)
        ) {
            // 1. Avatar Selection
            Box(
                contentAlignment = Alignment.BottomEnd,
                modifier = Modifier.clickable { imagePickerLauncher.launch("image/*") }
            ) {
                Surface(
                    modifier = Modifier.size(120.dp),
                    shape = CircleShape,
                    color = SurfaceContainerHigh,
                    border = BorderStroke(2.dp, PrimaryColor.copy(alpha = 0.5f))
                ) {
                    if (selectedImageUri != null) {
                        AsyncImage(
                            model = selectedImageUri,
                            contentDescription = null,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.clip(CircleShape)
                        )
                    } else if (userProfile.avatarUrl.isNotEmpty()) {
                        AsyncImage(
                            model = userProfile.avatarUrl,
                            contentDescription = null,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.clip(CircleShape)
                        )
                    } else {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(Icons.Default.AddAPhoto, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(40.dp))
                        }
                    }
                }
                
                Surface(
                    color = PrimaryColor,
                    shape = CircleShape,
                    modifier = Modifier.size(32.dp).offset(x = (-4).dp, y = (-4).dp)
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(Icons.Default.Edit, contentDescription = null, tint = OnPrimaryFixed, modifier = Modifier.size(16.dp))
                    }
                }
            }

            // 2. Input Fields
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                SenseInputField(
                    value = nickname,
                    onValueChange = { nickname = it },
                    label = stringResource(R.string.username),
                    icon = Icons.Default.Person
                )

                SenseInputField(
                    value = description,
                    onValueChange = { description = it },
                    label = stringResource(R.string.bio),
                    icon = Icons.Default.Description
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            // 3. Save Button
            Button(
                onClick = {
                    android.util.Log.e("EDIT_PROFILE", ">>> Save Button Clicked | nickname=$nickname")
                    scope.launch {
                        isUploading = true
                        if (selectedImageUri != null) {
                            android.util.Log.e("EDIT_PROFILE", ">>> Starting image upload for URI: $selectedImageUri")
                            // 直接上传图片
                            ossManager.uploadImage(
                                uri = selectedImageUri!!,
                                onSuccess = { url ->
                                    android.util.Log.e("EDIT_PROFILE", ">>> Image uploaded successfully: $url")
                                    scope.launch {
                                        // 更新资料
                                        repository.updateUserProfile(nickname, description, url)
                                        isUploading = false
                                        Toast.makeText(context, context.getString(R.string.profile_updated_avatar), Toast.LENGTH_SHORT).show()
                                        onBack()
                                    }
                                },
                                onFailure = { ex ->
                                    android.util.Log.e("EDIT_PROFILE", ">>> Image upload FAILED: ${ex.message}")
                                    isUploading = false
                                    Toast.makeText(context, context.getString(R.string.upload_failed, ex.message), Toast.LENGTH_SHORT).show()
                                }
                            )
                        } else {
                            android.util.Log.e("EDIT_PROFILE", ">>> No new image selected, updating text only")
                            // 只更新文字资料
                            repository.updateUserProfile(nickname, description)
                            isUploading = false
                            Toast.makeText(context, context.getString(R.string.profile_updated), Toast.LENGTH_SHORT).show()
                            onBack()
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth().height(56.dp),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor),
                enabled = !isUploading
            ) {
                if (isUploading) {
                    CircularProgressIndicator(color = OnPrimaryFixed, modifier = Modifier.size(24.dp))
                } else {
                    Text(stringResource(R.string.save_changes).uppercase(), fontWeight = FontWeight.Black, letterSpacing = 1.sp)
                }
            }
        }
    }
}
