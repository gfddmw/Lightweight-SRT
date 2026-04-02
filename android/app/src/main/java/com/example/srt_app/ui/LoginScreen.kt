package com.example.srt_app.ui

import androidx.compose.animation.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.srt_app.R
import com.example.srt_app.ui.theme.*

import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.srt_app.data.AppDatabase
import com.example.srt_app.data.UserRepository
import com.example.srt_app.ui.viewmodel.AuthState
import com.example.srt_app.ui.viewmodel.AuthViewModel
import com.example.srt_app.ui.viewmodel.AuthViewModelFactory
import android.widget.Toast

enum class LoginMode {
    Password, VerificationCode
}

@Composable
fun LoginScreen(
    onLoginSuccess: (String, String, String, String, String) -> Unit,
    onNavigateToRegister: () -> Unit
) {
    val context = LocalContext.current
    val database = AppDatabase.getDatabase(context)
    val settingsManager = remember { com.example.srt_app.utils.SettingsManager(context) }
    val repository = remember { UserRepository(context, database.userDao(), settingsManager) }
    val factory = remember { AuthViewModelFactory(repository) }
    val viewModel: AuthViewModel = viewModel(factory = factory)

    val authState by viewModel.authState.collectAsState()

    var loginMode by remember { mutableStateOf(LoginMode.Password) }
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var verificationCode by remember { mutableStateOf("") }
    var verificationToken by remember { mutableStateOf("") } // 新增：保存获取到的 Token
    var isPasswordVisible by remember { mutableStateOf(false) }

    LaunchedEffect(authState) {
        when (authState) {
            is AuthState.Success -> {
                val successState = authState as AuthState.Success
                onLoginSuccess(
                    successState.accessToken, 
                    successState.user.username, 
                    "User",
                    successState.accessToken,
                    successState.refreshToken
                )
                viewModel.resetState()
            }
            is AuthState.Error -> {
                Toast.makeText(context, (authState as AuthState.Error).message, Toast.LENGTH_SHORT).show()
                viewModel.resetState()
            }
            is AuthState.CodeSent -> {
                verificationToken = (authState as AuthState.CodeSent).token // 捕获并保存 Token
                Toast.makeText(context, "Verification code sent to your email", Toast.LENGTH_SHORT).show()
            }
            else -> {}
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(SurfaceDim)
    ) {
        // Decorative background glow
        Box(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .height(400.dp)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(PrimaryColor.copy(alpha = 0.15f), Color.Transparent)
                    )
                )
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Logo / Title
            Text(
                text = "LUMINARY",
                style = MaterialTheme.typography.displaySmall.copy(
                    fontWeight = FontWeight.Black,
                    letterSpacing = 8.sp
                ),
                color = Color.White
            )
            Text(
                text = stringResource(R.string.auth_system).uppercase(),
                style = MaterialTheme.typography.labelSmall.copy(
                    letterSpacing = 2.sp,
                    fontWeight = FontWeight.Bold
                ),
                color = PrimaryColor
            )

            Spacer(modifier = Modifier.height(64.dp))

            // Input Fields
            SenseInputField(
                value = email,
                onValueChange = { email = it },
                label = stringResource(R.string.email_address),
                icon = Icons.Default.Email
            )

            Spacer(modifier = Modifier.height(20.dp))

            // Dynamic Input Fields based on LoginMode
            if (loginMode == LoginMode.Password) {
                // Password Field with Visibility Toggle
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = stringResource(R.string.password).uppercase(),
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp,
                            color = OnSurfaceVariant.copy(alpha = 0.7f)
                        ),
                        modifier = Modifier.padding(start = 4.dp)
                    )
                    Surface(
                        color = Color.Black.copy(alpha = 0.3f),
                        shape = RoundedCornerShape(16.dp),
                        border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.2f))
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(Icons.Default.Lock, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(20.dp))
                            TextField(
                                value = password,
                                onValueChange = { password = it },
                                modifier = Modifier.weight(1f),
                                visualTransformation = if (isPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                                colors = TextFieldDefaults.colors(
                                    focusedContainerColor = Color.Transparent,
                                    unfocusedContainerColor = Color.Transparent,
                                    focusedIndicatorColor = Color.Transparent,
                                    unfocusedIndicatorColor = Color.Transparent,
                                    cursorColor = PrimaryColor,
                                    focusedTextColor = Color.White,
                                    unfocusedTextColor = Color.White
                                ),
                                singleLine = true
                            )
                            IconButton(onClick = { isPasswordVisible = !isPasswordVisible }) {
                                Icon(
                                    if (isPasswordVisible) Icons.Default.VisibilityOff else Icons.Default.Visibility,
                                    contentDescription = null,
                                    tint = OnSurfaceVariant
                                )
                            }
                        }
                    }
                }
            } else {
                // Verification Code Field
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = "VERIFICATION CODE",
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp,
                            color = OnSurfaceVariant.copy(alpha = 0.7f)
                        ),
                        modifier = Modifier.padding(start = 4.dp)
                    )
                    Surface(
                        color = Color.Black.copy(alpha = 0.3f),
                        shape = RoundedCornerShape(16.dp),
                        border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.2f))
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(Icons.Default.VpnKey, contentDescription = null, tint = PrimaryColor, modifier = Modifier.size(20.dp))
                            TextField(
                                value = verificationCode,
                                onValueChange = { verificationCode = it },
                                modifier = Modifier.weight(1f),
                                placeholder = { Text("6-digit code", color = OnSurfaceVariant.copy(alpha = 0.5f)) },
                                colors = TextFieldDefaults.colors(
                                    focusedContainerColor = Color.Transparent,
                                    unfocusedContainerColor = Color.Transparent,
                                    focusedIndicatorColor = Color.Transparent,
                                    unfocusedIndicatorColor = Color.Transparent,
                                    cursorColor = PrimaryColor,
                                    focusedTextColor = Color.White,
                                    unfocusedTextColor = Color.White
                                ),
                                singleLine = true
                            )
                            TextButton(
                                onClick = {
                                    if (email.isNotEmpty()) {
                                        viewModel.sendVerificationCode(email)
                                    } else {
                                        Toast.makeText(context, "Please enter email first", Toast.LENGTH_SHORT).show()
                                    }
                                },
                                enabled = authState !is AuthState.Loading
                            ) {
                                Text("GET CODE", color = PrimaryColor, fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            // Switch Mode Button
            TextButton(onClick = { 
                loginMode = if (loginMode == LoginMode.Password) LoginMode.VerificationCode else LoginMode.Password 
            }) {
                Text(
                    text = if (loginMode == LoginMode.Password) "Use Verification Code" else "Use Password",
                    color = PrimaryColor.copy(alpha = 0.8f),
                    style = MaterialTheme.typography.bodySmall
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Login Button
            Button(
                onClick = {
                    if (email.isNotEmpty()) {
                        if (loginMode == LoginMode.Password) {
                            if (password.isNotEmpty()) {
                                viewModel.login(email, password)
                            } else {
                                Toast.makeText(context, "Please enter password", Toast.LENGTH_SHORT).show()
                            }
                        } else {
                            if (verificationCode.isNotEmpty() && verificationToken.isNotEmpty()) {
                                viewModel.loginWithCode(email, verificationCode, verificationToken)
                            } else if (verificationToken.isEmpty()) {
                                Toast.makeText(context, "Please get verification code first", Toast.LENGTH_SHORT).show()
                            } else {
                                Toast.makeText(context, "Please enter verification code", Toast.LENGTH_SHORT).show()
                            }
                        }
                    } else {
                        Toast.makeText(context, "Please enter email", Toast.LENGTH_SHORT).show()
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor),
                enabled = authState !is AuthState.Loading
            ) {
                if (authState is AuthState.Loading) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), color = OnPrimaryFixed)
                } else {
                    Text(
                        stringResource(R.string.sign_in).uppercase(),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp),
                        color = OnPrimaryFixed
                    )
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Footer Link
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.new_to_luminary), color = OnSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
                TextButton(onClick = onNavigateToRegister) {
                    Text(stringResource(R.string.create_account), color = PrimaryColor, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}
