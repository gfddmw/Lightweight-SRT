package com.example.srt_app.ui

import androidx.compose.animation.*
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.srt_app.R
import com.example.srt_app.ui.theme.*
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.srt_app.data.UserRepository
import com.example.srt_app.utils.TokenManager
import com.example.srt_app.ui.viewmodel.AuthState
import com.example.srt_app.ui.viewmodel.AuthViewModel
import com.example.srt_app.ui.viewmodel.AuthViewModelFactory
import android.widget.Toast

enum class LoginMode {
    Password, VerificationCode
}

@Composable
fun LoginScreen(
    repository: UserRepository,
    onLoginSuccess: (String, String, String, String, String) -> Unit,
    onNavigateToRegister: () -> Unit
) {
    val context = LocalContext.current
    val factory = remember { AuthViewModelFactory(repository) }
    val viewModel: AuthViewModel = viewModel(factory = factory)

    val authState by viewModel.authState.collectAsState()

    var loginMode by remember { mutableStateOf(LoginMode.Password) }
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var verificationCode by remember { mutableStateOf("") }
    var verificationToken by remember { mutableStateOf("") }
    
    // Local Validation Errors
    var emailError by remember { mutableStateOf<String?>(null) }
    var passwordError by remember { mutableStateOf<String?>(null) }
    var codeError by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(loginMode) {
        emailError = null
        passwordError = null
        codeError = null
    }

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
                verificationToken = (authState as AuthState.CodeSent).token
                Toast.makeText(context, context.getString(R.string.verification_sent), Toast.LENGTH_SHORT).show()
            }
            else -> {}
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(SurfaceDim)
    ) {
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
                .statusBarsPadding()
                .padding(horizontal = 24.dp, vertical = 28.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = stringResource(R.string.app_name),
                style = MaterialTheme.typography.displaySmall.copy(
                    fontWeight = FontWeight.Black,
                    letterSpacing = 0.sp
                ),
                color = Color.White
            )
            Text(
                text = stringResource(R.string.auth_system),
                style = MaterialTheme.typography.labelSmall.copy(
                    fontWeight = FontWeight.Bold
                ),
                color = PrimaryColor
            )

            Spacer(modifier = Modifier.height(36.dp))

            AuthModeToggle(
                loginMode = loginMode,
                onModeChange = { loginMode = it }
            )

            Spacer(modifier = Modifier.height(24.dp))

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

            Spacer(modifier = Modifier.height(20.dp))

            if (loginMode == LoginMode.Password) {
                SensePasswordField(
                    value = password,
                    onValueChange = { 
                        password = it
                        if (passwordError != null) passwordError = null
                    },
                    label = stringResource(R.string.password),
                    errorText = passwordError
                )

                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    val forgotInteraction = remember { MutableInteractionSource() }
                    val forgotPressed by forgotInteraction.collectIsPressedAsState()
                    val forgotScale by animateFloatAsState(if (forgotPressed) 0.95f else 1f, label = "forgotScale")
                    TextButton(
                        onClick = {
                            if (email.isEmpty()) {
                                emailError = context.getString(R.string.enter_email_first)
                            } else {
                                Toast.makeText(context, context.getString(R.string.forgot_password_hint), Toast.LENGTH_LONG).show()
                            }
                        },
                        modifier = Modifier.scale(forgotScale),
                        interactionSource = forgotInteraction
                    ) {
                        Text(
                            text = stringResource(R.string.forgot_password),
                            color = PrimaryColor,
                            fontWeight = FontWeight.Bold,
                            style = MaterialTheme.typography.labelMedium
                        )
                    }
                }
            } else {
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
            }

            Spacer(modifier = Modifier.height(28.dp))

            val buttonInteraction = remember { MutableInteractionSource() }
            val buttonPressed by buttonInteraction.collectIsPressedAsState()
            val buttonScale by animateFloatAsState(if (buttonPressed) 0.96f else 1f, label = "buttonScale")

            Button(
                onClick = {
                    emailError = null
                    passwordError = null
                    codeError = null

                    if (email.isEmpty()) {
                        emailError = context.getString(R.string.enter_email_first)
                        return@Button
                    }

                    if (loginMode == LoginMode.Password) {
                        if (password.isEmpty()) {
                            passwordError = context.getString(R.string.enter_password)
                        } else {
                            viewModel.login(email, password)
                        }
                    } else {
                        var hasError = false
                        if (verificationToken.isEmpty()) {
                            emailError = context.getString(R.string.send_code_first)
                            hasError = true
                        }
                        if (verificationCode.isEmpty()) {
                            codeError = context.getString(R.string.enter_code)
                            hasError = true
                        }
                        if (!hasError) {
                            viewModel.loginWithCode(email, verificationCode, verificationToken)
                        }
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
                    Text(
                        stringResource(R.string.sign_in).uppercase(),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp),
                        color = OnPrimaryFixed
                    )
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.new_to_luminary), color = OnSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
                val footerLinkInteraction = remember { MutableInteractionSource() }
                val footerLinkPressed by footerLinkInteraction.collectIsPressedAsState()
                val footerLinkScale by animateFloatAsState(if (footerLinkPressed) 0.95f else 1f, label = "footerLinkScale")
                TextButton(
                    onClick = onNavigateToRegister,
                    modifier = Modifier.scale(footerLinkScale),
                    interactionSource = footerLinkInteraction
                ) {
                    Text(stringResource(R.string.create_account), color = PrimaryColor, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

@Composable
fun AuthModeToggle(loginMode: LoginMode, onModeChange: (LoginMode) -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(SurfaceContainerLow, RoundedCornerShape(14.dp))
            .border(1.dp, OutlineVariant.copy(alpha = 0.28f), RoundedCornerShape(14.dp))
            .padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        AuthModeButton(
            label = stringResource(R.string.password),
            active = loginMode == LoginMode.Password,
            modifier = Modifier.weight(1f)
        ) {
            onModeChange(LoginMode.Password)
        }
        AuthModeButton(
            label = stringResource(R.string.verification_code),
            active = loginMode == LoginMode.VerificationCode,
            modifier = Modifier.weight(1f)
        ) {
            onModeChange(LoginMode.VerificationCode)
        }
    }
}

@Composable
fun AuthModeButton(label: String, active: Boolean, modifier: Modifier = Modifier, onClick: () -> Unit) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(if (isPressed) 0.96f else 1f, label = "scale")

    Surface(
        color = if (active) PrimaryColor else Color.Transparent,
        shape = RoundedCornerShape(10.dp),
        modifier = modifier
            .height(42.dp)
            .scale(scale)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onClick
            )
    ) {
        Box(contentAlignment = Alignment.Center) {
            Text(
                text = label,
                color = if (active) OnPrimaryFixed else OnSurfaceVariant,
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.labelMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}
