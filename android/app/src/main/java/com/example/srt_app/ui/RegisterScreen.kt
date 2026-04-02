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

@Composable
fun RegisterScreen(
    onRegisterSuccess: (String, String, String, String, String) -> Unit,
    onNavigateToLogin: () -> Unit
) {
    val context = LocalContext.current
    val database = AppDatabase.getDatabase(context)
    val settingsManager = remember { com.example.srt_app.utils.SettingsManager(context) }
    val repository = remember { UserRepository(context, database.userDao(), settingsManager) }
    val factory = remember { AuthViewModelFactory(repository) }
    val viewModel: AuthViewModel = viewModel(factory = factory)

    val authState by viewModel.authState.collectAsState()

    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var verificationCode by remember { mutableStateOf("") }
    var verificationToken by remember { mutableStateOf("") }

    LaunchedEffect(authState) {
        when (authState) {
            is AuthState.Success -> {
                val successState = authState as AuthState.Success
                onRegisterSuccess(
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
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .height(400.dp)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(Color.Transparent, SecondaryColor.copy(alpha = 0.1f))
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
            Text(
                text = stringResource(R.string.join_luminary).uppercase(),
                style = MaterialTheme.typography.displaySmall.copy(
                    fontWeight = FontWeight.Black,
                    letterSpacing = 4.sp
                ),
                color = Color.White
            )
            Text(
                text = stringResource(R.string.start_journey).uppercase(),
                style = MaterialTheme.typography.labelSmall.copy(
                    letterSpacing = 1.sp,
                    fontWeight = FontWeight.Bold
                ),
                color = SecondaryColor
            )

            Spacer(modifier = Modifier.height(48.dp))

            SenseInputField(
                value = email,
                onValueChange = { email = it },
                label = stringResource(R.string.email_address),
                icon = Icons.Default.Email
            )

            Spacer(modifier = Modifier.height(16.dp))

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
                    color = SurfaceDim,
                    shape = RoundedCornerShape(16.dp),
                    border = BorderStroke(1.dp, OutlineVariant.copy(alpha = 0.3f))
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.VpnKey, contentDescription = null, tint = SecondaryColor, modifier = Modifier.size(20.dp))
                        TextField(
                            value = verificationCode,
                            onValueChange = { verificationCode = it },
                            modifier = Modifier.weight(1f),
                            placeholder = { Text("Code", color = OnSurfaceVariant.copy(alpha = 0.5f)) },
                            colors = TextFieldDefaults.colors(
                                focusedContainerColor = Color.Transparent,
                                unfocusedContainerColor = Color.Transparent,
                                focusedIndicatorColor = Color.Transparent,
                                unfocusedIndicatorColor = Color.Transparent,
                                cursorColor = SecondaryColor,
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
                            Text("GET CODE", color = SecondaryColor, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            SenseInputField(
                value = password,
                onValueChange = { password = it },
                label = stringResource(R.string.password),
                icon = Icons.Default.Lock,
                visualTransformation = PasswordVisualTransformation()
            )

            Spacer(modifier = Modifier.height(48.dp))

            Button(
                onClick = {
                    if (email.isNotEmpty() && password.isNotEmpty() && verificationCode.isNotEmpty() && verificationToken.isNotEmpty()) {
                        viewModel.register("user_default", email, password, verificationCode, verificationToken)
                    } else if (verificationToken.isEmpty()) {
                        Toast.makeText(context, "Please get verification code first", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(context, "Please fill all fields", Toast.LENGTH_SHORT).show()
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(containerColor = SecondaryColor),
                enabled = authState !is AuthState.Loading
            ) {
                if (authState is AuthState.Loading) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), color = Color.Black)
                } else {
                    Text(
                        stringResource(R.string.create_account).uppercase(),
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Black, letterSpacing = 1.sp),
                        color = Color.Black
                    )
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.already_have_account), color = OnSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
                TextButton(onClick = onNavigateToLogin) {
                    Text(stringResource(R.string.sign_in), color = SecondaryColor, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}
