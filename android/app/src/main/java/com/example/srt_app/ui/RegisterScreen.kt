package com.example.srt_app.ui

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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
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

    LaunchedEffect(authState) {
        when (authState) {
            is AuthState.Success -> {
                Log.i("SRT_DEBUG", ">>> Register SUCCESS for user: $username")
                Toast.makeText(context, "注册成功！", Toast.LENGTH_SHORT).show()
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
                Toast.makeText(context, "验证码已发送", Toast.LENGTH_SHORT).show()
            }
            else -> {}
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(SurfaceDim)) {
        Column(
            modifier = Modifier.fillMaxSize().padding(32.dp).verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text("加入 LUMINARY", style = MaterialTheme.typography.displaySmall.copy(fontWeight = FontWeight.Black, letterSpacing = 2.sp), color = Color.White)
            Text("创建您的阿里云账户", style = MaterialTheme.typography.labelSmall.copy(letterSpacing = 1.sp, color = PrimaryColor))

            Spacer(modifier = Modifier.height(48.dp))

            SenseInputField(value = username, onValueChange = { username = it }, label = "用户名", icon = Icons.Default.Person)
            Spacer(modifier = Modifier.height(16.dp))
            SenseInputField(value = email, onValueChange = { email = it }, label = "电子邮箱", icon = Icons.Default.Email)
            Spacer(modifier = Modifier.height(16.dp))
            SenseInputField(value = password, onValueChange = { password = it }, label = "设置密码", icon = Icons.Default.Lock)
            
            Spacer(modifier = Modifier.height(32.dp))

            // Register Button
            Button(
                onClick = {
                    // 按钮点击瞬间立即打印
                    Log.e("!!!_DEBUG_!!!", ">>> CLICK REGISTER BUTTON | User: $username | Email: $email")
                    
                    if (email.isNotEmpty() && password.isNotEmpty() && username.isNotEmpty()) {
                        viewModel.register(
                            username = username,
                            email = email,
                            password = password,
                            verificationCode = verificationCode,
                            verificationToken = verificationId
                        ) 
                    } else {
                        Log.e("!!!_DEBUG_!!!", ">>> ABORT: Some fields are empty")
                        Toast.makeText(context, "请填写所有必填项", Toast.LENGTH_SHORT).show()
                    }
                },
                modifier = Modifier.fillMaxWidth().height(56.dp),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(containerColor = PrimaryColor),
                enabled = authState !is AuthState.Loading
            ) {
                if (authState is AuthState.Loading) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), color = OnPrimaryFixed)
                } else {
                    Text("立即注册", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
                }
            }

            TextButton(onClick = onNavigateToBack, modifier = Modifier.padding(top = 16.dp)) {
                Text("已有账户？点击登录", color = OnSurfaceVariant)
            }
        }
    }
}
