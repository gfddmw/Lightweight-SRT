package com.example.srt_app

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.compose.runtime.collectAsState
import androidx.core.os.LocaleListCompat
import com.example.srt_app.camera.SignFrameAnalyzer
import com.example.srt_app.ml.HandLandmarkerHelper
import com.example.srt_app.ml.LightweightTranslator
import com.example.srt_app.ui.SRTScreen
import com.example.srt_app.ui.theme.SRTAppTheme
import com.example.srt_app.utils.SettingsManager
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

import android.speech.tts.TextToSpeech
import androidx.compose.runtime.LaunchedEffect
import kotlinx.coroutines.flow.collectLatest

class MainActivity : AppCompatActivity(), HandLandmarkerHelper.LandmarkerListener {

    private lateinit var translator: LightweightTranslator
    private lateinit var frameAnalyzer: SignFrameAnalyzer
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var settingsManager: SettingsManager
    private lateinit var userRepository: com.example.srt_app.data.UserRepository
    private var handLandmarkerHelper: HandLandmarkerHelper? = null
    private var tts: TextToSpeech? = null

    private var imageCapture: ImageCapture? = null
    private var lastPreviewView: PreviewView? = null

    // UI States
    private var translationResult by mutableStateOf("")
    private var confidence by mutableFloatStateOf(0.0f)
    private var isProcessing by mutableStateOf(false)
    private var handLandmarks by mutableStateOf<HandLandmarkerResult?>(null)

    // Camera States
    private var lensFacing by mutableStateOf(CameraSelector.LENS_FACING_BACK)

    // Navigation States
    private var currentScreen by mutableStateOf("home")

    // --- Preprocessing Constants & Buffers ---
    private val windowSize = 64
    private val featureDimension = 21 * 3 // 21 landmarks * 3 (xyz)
    private val frameBuffer = java.util.Collections.synchronizedList(mutableListOf<FloatArray>())
    private var lastRawLandmarks: FloatArray? = null
    private val smoothingFactor = 0.3f // Exponential smoothing factor
    // -----------------------------------------

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            // Permission granted
        } else {
            Toast.makeText(this, getString(R.string.camera_permission_denied), Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        settingsManager = SettingsManager(this)
        initTTS()
        try {
            handLandmarkerHelper = HandLandmarkerHelper(
                context = this,
                handLandmarkerHelperListener = this,
                currentDelegate = HandLandmarkerHelper.DELEGATE_CPU // 强制使用 CPU
            )
        } catch (e: Exception) {
            Log.e("MainActivity", "HandLandmarkerHelper init failed", e)
            Toast.makeText(this, "Recognition module initialization failed", Toast.LENGTH_LONG).show()
        }
        initDependencies()
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            val scope = androidx.compose.runtime.rememberCoroutineScope()
            val settings = settingsManager.settingsFlow.collectAsState(initial = null).value

            // 优化：添加防抖处理，只有当用户停止滑动 500ms 后才真正重新初始化识别器
            LaunchedEffect(settings?.confidenceThreshold) {
                settings?.confidenceThreshold?.let { threshold ->
                    kotlinx.coroutines.delay(500) // 等待 500ms
                    handLandmarkerHelper?.let { helper ->
                        helper.minHandDetectionConfidence = threshold
                        helper.minHandTrackingConfidence = threshold
                        helper.minHandPresenceConfidence = threshold
                        // 在后台线程重新初始化，防止阻塞 UI
                        withContext(Dispatchers.IO) {
                            helper.setupHandLandmarker()
                        }
                    }
                }
            }

            // Apply App Language
            LaunchedEffect(settings?.appLanguage) {
                settings?.appLanguage?.let { lang ->
                    val appLocales = LocaleListCompat.forLanguageTags(lang)
                    AppCompatDelegate.setApplicationLocales(appLocales)
                }
            }

            // 全局同步：只要用户已登录且有 Token，就尝试同步云端资料
            LaunchedEffect(settings?.isLoggedIn, settings?.accessToken) {
                if (settings?.isLoggedIn == true && !settings.accessToken.isNullOrEmpty()) {
                    Log.d("MainActivity", ">>> [SYNC] User logged in, starting background sync...")
                    val result = userRepository.getUserProfile(settings.accessToken)
                    result.onSuccess { data ->
                        val nickname = if (data.has("nickname") && !data.get("nickname").isJsonNull) 
                            data.get("nickname").asString else ""
                        val username = if (data.has("username") && !data.get("username").isJsonNull) 
                            data.get("username").asString else ""
                        
                        val finalName = nickname.ifEmpty { username }
                        val description = if (data.has("description") && !data.get("description").isJsonNull) 
                            data.get("description").asString else ""
                            
                        Log.d("MainActivity", ">>> [SYNC] Success! Name: $finalName, Role: $description")
                        
                        if (finalName.isNotEmpty()) settingsManager.updateUserName(finalName)
                        if (description.isNotEmpty()) settingsManager.updateUserRole(description)
                    }.onFailure { error ->
                        Log.e("MainActivity", ">>> [SYNC] Failed to fetch cloud profile", error)
                    }
                } else {
                    Log.d("MainActivity", ">>> [SYNC] Skip sync: LoggedIn=${settings?.isLoggedIn}, HasToken=${!settings?.accessToken.isNullOrEmpty()}")
                }
            }

            SRTAppTheme {
                if (settings != null) {
                    when (currentScreen) {
                        "login" -> {
                            com.example.srt_app.ui.LoginScreen(
                                onLoginSuccess = { token, name, role, access, refresh ->
                                    scope.launch {
                                        settingsManager.setLoggedIn(true, token, name, role, access, refresh)
                                        currentScreen = "profile"
                                    }
                                },
                                onNavigateToRegister = { currentScreen = "register" }
                            )
                        }
                        "register" -> {
                            com.example.srt_app.ui.RegisterScreen(
                                onRegisterSuccess = { token, name, role, access, refresh ->
                                    scope.launch {
                                        settingsManager.setLoggedIn(true, token, name, role, access, refresh)
                                        currentScreen = "profile"
                                    }
                                },
                                onNavigateToLogin = { currentScreen = "login" }
                            )
                        }
                        "home" -> {
                            SRTScreen(
                                translationResult = translationResult,
                                onStartCamera = { previewView ->
                                    lastPreviewView = previewView
                                    startCamera(previewView)
                                },
                                onNavigateToSettings = { currentScreen = "settings" },
                                onNavigateToProfile = {
                                    if (settings.isLoggedIn) {
                                        currentScreen = "profile"
                                    } else {
                                        currentScreen = "login"
                                    }
                                },
                                onCapture = { takePhoto() },
                                onFlipCamera = { toggleCamera() },
                                onTypeBackSend = { text ->
                                    translationResult = text
                                    speak(text)
                                },
                                showSkeleton = settings.showSkeleton,
                                handLandmarks = handLandmarks,
                                signLanguage = settings.signLanguageStandard,
                                outputLanguage = settings.outputLanguage
                            )
                        }
                        "settings" -> {
                            com.example.srt_app.ui.SettingsScreen(
                                onBack = { currentScreen = "profile" },
                                onNavigateToProfile = {
                                    if (settings.isLoggedIn) {
                                        currentScreen = "profile"
                                    } else {
                                        currentScreen = "login"
                                    }
                                },
                                onSave = { newSettings ->
                                    scope.launch {
                                        settingsManager.updateAppLanguage(newSettings.appLanguage)
                                        settingsManager.updateSignLanguageStandard(newSettings.signLanguageStandard)
                                        settingsManager.updateOutputLanguage(newSettings.outputLanguage)
                                        settingsManager.updateShowSkeleton(newSettings.showSkeleton)
                                        settingsManager.updateConfidenceThreshold(newSettings.confidenceThreshold)
                                        settingsManager.updateAutoFocus(newSettings.autoFocus)
                                        settingsManager.updateTextSize(newSettings.textSize)
                                        settingsManager.updateDisplayDuration(newSettings.displayDuration)
                                        settingsManager.updateFlashOnTranslation(newSettings.flashOnTranslation)
                                        settingsManager.updateVibration(newSettings.vibration)
                                        currentScreen = "profile"
                                    }
                                },
                                onReset = {
                                    scope.launch {
                                        settingsManager.resetSettings()
                                    }
                                },
                                initialSettings = settings
                            )
                        }
                        "profile" -> {
                            if (!settings.isLoggedIn) {
                                currentScreen = "login"
                            } else {
                                // 全局同步：只要用户已登录且有 Token，就尝试同步云端资料
                                LaunchedEffect(Unit) {
                                    if (settings.accessToken.isNotEmpty()) {
                                        Log.d("MainActivity", ">>> [SYNC] Starting profile sync...")
                                        val result = userRepository.getUserProfile(settings.accessToken)
                                        result.onSuccess { data ->
                                            // 打印完整的原始数据，方便排查键名
                                            Log.d("MainActivity", ">>> [SYNC] Raw Data from Cloud: $data")
                                            
                                            // 兼容性获取：尝试 nick_name, nickname, nickName
                                            val nickname = when {
                                                data.has("nick_name") && !data.get("nick_name").isJsonNull -> data.get("nick_name").asString
                                                data.has("nickname") && !data.get("nickname").isJsonNull -> data.get("nickname").asString
                                                data.has("nickName") && !data.get("nickName").isJsonNull -> data.get("nickName").asString
                                                else -> ""
                                            }
                                            // 兼容性获取：尝试 username, name
                                            val username = when {
                                                data.has("username") && !data.get("username").isJsonNull -> data.get("username").asString
                                                data.has("name") && !data.get("name").isJsonNull -> data.get("name").asString
                                                else -> ""
                                            }
                                            
                                            val finalName = nickname.ifEmpty { username }
                                            
                                            // 兼容性获取：尝试 description, role, user_role
                                            val description = when {
                                                data.has("description") && !data.get("description").isJsonNull -> data.get("description").asString
                                                data.has("role") && !data.get("role").isJsonNull -> data.get("role").asString
                                                data.has("user_role") && !data.get("user_role").isJsonNull -> data.get("user_role").asString
                                                else -> ""
                                            }

                                            Log.d("MainActivity", ">>> [SYNC] Parsed: Name='$finalName', Role='$description'")

                                            if (finalName.isNotEmpty()) {
                                                settingsManager.updateUserName(finalName)
                                            }
                                            if (description.isNotEmpty()) {
                                                settingsManager.updateUserRole(description)
                                            }
                                        }.onFailure { error ->
                                            Log.e("MainActivity", ">>> [SYNC] Failed", error)
                                        }
                                    }
                                }
                                com.example.srt_app.ui.ProfileScreen(
                                    onNavigateToTranslator = { currentScreen = "home" },
                                    onNavigateToSettings = { currentScreen = "settings" },
                                    onLogout = {
                                        scope.launch {
                                            settingsManager.logout()
                                            currentScreen = "home"
                                        }
                                    },
                                    userSettings = settings,
                                    onUpdateProfile = { name, role ->
                                        scope.launch {
                                            // 1. 更新本地缓存
                                            settingsManager.updateUserName(name)
                                            settingsManager.updateUserRole(role)

                                            // 2. 同步到云端
                                            if (settings.accessToken.isNotEmpty()) {
                                                val result = userRepository.updateProfile(
                                                    accessToken = settings.accessToken,
                                                    nickname = name,
                                                    description = role
                                                )
                                                result.onFailure { error ->
                                                    withContext(Dispatchers.Main) {
                                                        Toast.makeText(this@MainActivity, "同步失败: ${error.message}", Toast.LENGTH_SHORT).show()
                                                    }
                                                }.onSuccess {
                                                    withContext(Dispatchers.Main) {
                                                        Toast.makeText(this@MainActivity, "资料已同步至云端", Toast.LENGTH_SHORT).show()
                                                    }
                                                }
                                            }
                                        }
                                    }
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    private fun initDependencies() {
        translator = LightweightTranslator(this)
        translator.initialize()

        val database = com.example.srt_app.data.AppDatabase.getDatabase(this)
        userRepository = com.example.srt_app.data.UserRepository(this, database.userDao(), settingsManager)

        frameAnalyzer = SignFrameAnalyzer(
            handLandmarkerHelper = handLandmarkerHelper,
            isFrontCamera = { lensFacing == CameraSelector.LENS_FACING_FRONT },
            isExternalProcessing = { isProcessing }
        )
    }

    override fun onResults(resultBundle: HandLandmarkerHelper.ResultBundle) {
        runOnUiThread {
            handLandmarks = resultBundle.results.firstOrNull()

            // 触发翻译逻辑（这里可以加入滑动窗口逻辑）
            if (handLandmarks != null && !isProcessing) {
                processLandmarks(handLandmarks!!)
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        runOnUiThread {
            Toast.makeText(this, error, Toast.LENGTH_SHORT).show()
        }
    }

    private fun processLandmarks(result: HandLandmarkerResult) {
        // 1. 特征提取 (Normalization & Flattening)
        val currentFrameFeatures = extractFeatures(result)

        // 2. 坐标平滑处理 (Exponential Smoothing)
        val smoothedFeatures = smoothFeatures(currentFrameFeatures)

        // 3. 滑动窗口管理
        synchronized(frameBuffer) {
            frameBuffer.add(smoothedFeatures)
            if (frameBuffer.size > windowSize) {
                frameBuffer.removeAt(0)
            }
        }

        // 4. 当缓冲区满时，触发异步翻译逻辑
        if (frameBuffer.size == windowSize && !isProcessing) {
            val inputToModel = synchronized(frameBuffer) {
                // 重新组织数据以匹配 ST-GCN 的输入形状: (N=1, C=3, T=64, V=21, M=1)
                // 顺序: 先所有 X (T*V个), 再所有 Y (T*V个), 最后所有 Z (T*V个)
                val totalSize = 3 * windowSize * 21
                val reshaped = FloatArray(totalSize)
                
                for (c in 0 until 3) {
                    for (t in 0 until windowSize) {
                        for (v in 0 until 21) {
                            val srcIdx = v * 3 + c
                            val dstIdx = c * (windowSize * 21) + t * 21 + v
                            reshaped[dstIdx] = frameBuffer[t][srcIdx]
                        }
                    }
                }
                reshaped
            }

            runTranslation(inputToModel)
        }
    }

    /**
     * 将 MediaPipe 的结果转换为归一化的 FloatArray (63 维: 21点 * 3轴)
     */
    private fun extractFeatures(result: HandLandmarkerResult): FloatArray {
        val features = FloatArray(featureDimension) // 默认全 0 (63维)

        val landmarks = result.landmarks()
        if (landmarks.isNotEmpty()) {
            val firstHand = landmarks[0]
            for (i in 0 until 21) {
                val landmark = firstHand[i]
                val offset = i * 3
                features[offset] = landmark.x()
                features[offset + 1] = landmark.y()
                features[offset + 2] = landmark.z()
            }
        }
        return features
    }

    /**
     * 简单的指数平滑算法，减少手部抖动
     */
    private fun smoothFeatures(current: FloatArray): FloatArray {
        val previous = lastRawLandmarks ?: return current.also { lastRawLandmarks = it }
        val smoothed = FloatArray(featureDimension)

        for (i in 0 until featureDimension) {
            smoothed[i] = current[i] * smoothingFactor + previous[i] * (1f - smoothingFactor)
        }

        lastRawLandmarks = smoothed
        return smoothed
    }

    private fun runTranslation(input: FloatArray) {
        CoroutineScope(Dispatchers.Main).launch {
            isProcessing = true
            try {
                // 调用后端 TFLite 翻译逻辑
                val translation = translator.translate(input)
                if (translation.isNotEmpty()) {
                    translationResult = translation
                    speak(translation)
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Translation error", e)
            } finally {
                isProcessing = false
            }
        }
    }

    private fun startCamera(previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            // Image Capture
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Image Analysis
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, frameAnalyzer)
                }

            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalysis
                )
            } catch (exc: Exception) {
                // Handle binding failure
                Log.e("MainActivity", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun toggleCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        lastPreviewView?.let { startCamera(it) }
    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time stamped name and MediaStore entry.
        val name = SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/SRTApp-Captures")
            }
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(
                contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            )
            .build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    val msg = getString(R.string.photo_save_failed, exc.message)
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val msg = getString(R.string.photo_save_success, output.savedUri.toString())
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d("MainActivity", msg)
                }
            }
        )
    }

    private fun initTTS() {
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.getDefault()
            }
        }
    }

    private fun speak(text: String) {
        if (text.isNotEmpty()) {
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        if (::translator.isInitialized) {
            translator.close()
        }
        handLandmarkerHelper?.clearHandLandmarker()
        tts?.stop()
        tts?.shutdown()
    }
}
