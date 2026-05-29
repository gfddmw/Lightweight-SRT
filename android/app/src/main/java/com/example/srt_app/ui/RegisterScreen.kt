package com.example.srt_app.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.srt_app.R
import com.example.srt_app.data.AppDatabase
import com.example.srt_app.data.UserRepository
import com.example.srt_app.data.User
import com.example.srt_app.ui.theme.*
import com.example.srt_app.utils.TokenManager
import com.example.srt_app.ui.viewmodel.AuthState
import com.example.srt_app.ui.viewmodel.AuthViewModel
import com.example.srt_app.ui.viewmodel.AuthViewModelFactory
import android.widget.Toast
import android.util.Log

@Composable
fun RegisterScreen(
    repository: UserRepository,
    onRegisterSuccess: () -> Unit,
    onNavigateToBack: () -> Unit
) {
    val context = LocalContext.current
    val factory = remember { AuthViewModelFactory(repository) }
    val viewModel: AuthViewModel = viewModel(factory = factory)

    val authState by viewModel.authState.collectAsState()

    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var username by remember { mutableStateOf("") }
    var verificationCode by remember { mutableStateOf("") }
    var verificationId by remember { mutableStateOf("") }

    // Local Validation Errors
    var usernameError by remember { mutableStateOf<String?>(null) }
    var emailError by remember { mutableStateOf<String?>(null) }
    var passwordError by remember { mutableStateOf<String?>(null) }
    var codeError by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(authState) {
        when (authState) {
            is AuthState.Success -> {
                Log.i("SRT_DEBUG", ">>> Register SUCCESS for user: $username")
                Toast.makeText(context, context.getString(R.string.register_success), Toast.LENGTH_SHORT).show()
                onRegisterSuccess()
                viewModel.resetState()
            }
            is AuthState.Error -> {
                val errorMsg = (authState as AuthState.Error).message
                Log.e("SRT_DEBUG", ">>> Register FAILED: $errorMsg")
                Toast.makeText(context, "注册失败: $errorMsg", Toast.LENGTH_LONG).show()
                viewModel.resetState()
            }
            is AuthState.CodeSent -> {
                verificationId = (authState as AuthState.CodeSent).token
                Log.i("SRT_DEBUG", ">>> Verification code sent, ID: $verificationId")
                Toast.makeText(context, context.getString(R.string.verification_sent), Toast.LENGTH_SHORT).show()
            }
            else -> {}
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(SurfaceDim)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .padding(horizontal = 24.dp, vertical = 28.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                stringResource(R.string.join_luminary),
                style = MaterialTheme.typography.displaySmall.copy(fontWeight = FontWeight.Black),
                color = Color.White
            )
            Text(
                stringResource(R.string.create_secure_account),
                style = MaterialTheme.typography.labelSmall.copy(color = PrimaryColor, fontWeight = FontWeight.Bold)
            )

            Spacer(modifier = Modifier.height(36.dp))

            SenseInputField(
                value = username,
                onValueChange = { 
                    username = it
                    if (usernameError != null) usernameError = null
                },
                label = stringResource(R.string.username),
                icon = Icons.Default.Person,
                errorText = usernameError
            )

            Spacer(modifier = Modifier.height(16.dp))

            SenseInputField(
                value = email,
                onValueChange = { 
                    email = it
                    if (emailError != null) emailError = null
                },
                label = stringResource(R.string.email_address),
                icon = Icons.Default.Email,
                errorText = emailError
            )

            Spacer(modifier = Modifier.height(16.dp))

            SensePasswordField(
                value = password,
                onValueChange = { 
                    password = it
                    if (passwordError != null) passwordError = null
                },
                label = stringResource(R.string.password),
                errorText = passwordError
            )

            Spacer(modifier = Modifier.height(16.dp))

            SenseVerificationCodeField(
                value = verificationCode,
                onValueChange = { 
                    verificationCode = it
                    if (codeError != null) codeError = null
                },
                label = stringResource(R.string.verification_code),
                onGetCodeClick = {
                    if (email.isNotEmpty()) {
                        viewModel.sendVerificationCode(email)
                    } else {
                        emailError = context.getString(R.string.enter_email_first)
                    }
                },
                isGetCodeEnabled = authState !is AuthState.Loading,
                errorText = codeError
            )
            
            Spacer(modifier = Modifier.height(24.dp))

            // Register Button
            val buttonInteraction = remember { MutableInteractionSource() }
            val buttonPressed by buttonInteraction.collectIsPressedAsState()
            val buttonScale by animateFloatAsState(if (buttonPressed) 0.96f else 1f, label = "buttonScale")

            Button(
                onClick = {
                    Log.e("!!!_DEBUG_!!!", ">>> CLICK REGISTER BUTTON | User: $username | Email: $email")
                    
                    usernameError = null
                    emailError = null
                    passwordError = null
                    codeError = null

                    var hasError = false
                    if (username.isEmpty()) {
                        usernameError = context.getString(R.string.fill_required_fields)
                        hasError = true
                    }
                    if (email.isEmpty()) {
                        emailError = context.getString(R.string.enter_email_first)
                        hasError = true
                    }
                    if (password.isEmpty()) {
                        passwordError = context.getString(R.string.enter_password)
                        hasError = true
                    }
                    if (verificationCode.isEmpty()) {
                        codeError = context.getString(R.string.enter_code)
                        hasError = true
                    }
                    if (verificationId.isEmpty()) {
                        emailError = context.getString(R.string.send_code_first)
                        hasError = true
                    }

                    if (!hasError) {
                        viewModel.register(
                            username = username,
                            email = email,
                            password = password,
                            verificationCode = verificationCode,
                            verificationToken = verificationId
                        ) 
                    } else {
                        Log.e("!!!_DEBUG_!!!", ">>> ABORT: Some fields are empty")
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .scale(buttonScale),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor),
                enabled = authState !is AuthState.Loading,
                interactionSource = buttonInteraction
            ) {
                if (authState is AuthState.Loading) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), color = OnPrimaryFixed)
                } else {
                    Text(stringResource(R.string.create_account), fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium, color = OnPrimaryFixed)
                }
            }

            val footerLinkInteraction = remember { MutableInteractionSource() }
            val footerLinkPressed by footerLinkInteraction.collectIsPressedAsState()
            val footerLinkScale by animateFloatAsState(if (footerLinkPressed) 0.95f else 1f, label = "footerLinkScale")

            TextButton(
                onClick = onNavigateToBack,
                modifier = Modifier
                    .padding(top = 16.dp)
                    .scale(footerLinkScale),
                interactionSource = footerLinkInteraction
            ) {
                Text(stringResource(R.string.already_have_account), color = OnSurfaceVariant)
            }
        }
    }
}
