package com.example.srt_app

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.provider.MediaStore
import android.util.Log
import android.view.MotionEvent
import android.widget.Toast
import android.speech.tts.TextToSpeech
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import androidx.core.os.LocaleListCompat
import com.example.srt_app.camera.SignFrameAnalyzer
import com.example.srt_app.ml.HandLandmarkerHelper
import com.example.srt_app.ml.LightweightTranslator
import com.example.srt_app.ui.SRTScreen
import com.example.srt_app.ui.SettingsScreen
import com.example.srt_app.ui.LoginScreen
import com.example.srt_app.ui.ProfileScreen
import com.example.srt_app.ui.EditProfileScreen
import com.example.srt_app.ui.theme.SRTAppTheme
import com.example.srt_app.utils.SettingsManager
import com.example.srt_app.utils.UserSettings
import com.example.srt_app.utils.TokenManager
import com.example.srt_app.data.AppDatabase
import com.example.srt_app.data.UserRepository
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity(), HandLandmarkerHelper.LandmarkerListener {

    private lateinit var translator: LightweightTranslator
    private lateinit var frameAnalyzer: SignFrameAnalyzer
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var settingsManager: SettingsManager
    private lateinit var userRepository: UserRepository // 确保声明
    private var handLandmarkerHelper: HandLandmarkerHelper? = null
    private var tts: TextToSpeech? = null

    private var imageCapture: ImageCapture? = null
    private var lastPreviewView: PreviewView? = null
    private var activeCamera: Camera? = null

    // UI States
    private var translationResult by mutableStateOf("")
    private var isProcessing by mutableStateOf(false)
    private var handLandmarks by mutableStateOf<HandLandmarkerResult?>(null)

    // Camera States
    private var lensFacing by mutableStateOf(CameraSelector.LENS_FACING_BACK)

    // Navigation States
    private var currentScreen by mutableStateOf("home")

    private var lastSpokenWord = ""
    private var lastWristPosition: android.graphics.PointF? = null
    private var handMovingVelocity = 0f
    private val STABILITY_THRESHOLD = 0.02f
    private val RECOGNITION_COOLDOWN = 1500L
    private var lastSpeakTime = 0L
    private var autoFocusEnabled = true
    private var vibrationEnabled = true
    private var displayDurationMillis = 5000L
    
    private enum class RecognitionState { IDLE, DETECTING, STABLE, LOCKED }
    private var currentState = RecognitionState.IDLE

    private val windowSize = 64
    private val featureDimension = 21 * 3 * 2
    private val frameBuffer = Collections.synchronizedList(mutableListOf<FloatArray>())
    private var lastRawLandmarks: FloatArray? = null
    private val smoothingFactor = 0.4f

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (!isGranted) {
            Toast.makeText(this, getString(R.string.camera_permission_denied), Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        settingsManager = SettingsManager(this)
        initTTS()
        initDependencies()
        
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            Log.e("!!!_DEBUG_!!!", ">>> [APP_START] setContent is executing!")
            val scope = rememberCoroutineScope()
            val settings = settingsManager.settingsFlow.collectAsState(initial = null).value

            // 自动冷启动同步：尝试恢复登录状态
            LaunchedEffect(Unit) {
                userRepository.performStartupSync()
            }

            LaunchedEffect(settings?.confidenceThreshold) {
                settings?.confidenceThreshold?.let { threshold ->
                    delay(500)
                    handLandmarkerHelper?.let { helper ->
                        helper.minHandDetectionConfidence = threshold
                        helper.setupHandLandmarker()
                    }
                }
            }

            LaunchedEffect(settings?.appLanguage) {
                settings?.appLanguage?.let { applyAppLocale(it) }
            }

            LaunchedEffect(settings?.outputLanguage) {
                settings?.outputLanguage?.let { updateTtsLanguage(it) }
            }

            LaunchedEffect(settings?.autoFocus) {
                autoFocusEnabled = settings?.autoFocus ?: true
                lastPreviewView?.let { configureAutoFocus(it) }
            }

            LaunchedEffect(settings?.displayDuration) {
                displayDurationMillis = durationToMillis(settings?.displayDuration ?: "5s")
            }

            LaunchedEffect(settings?.vibration) {
                vibrationEnabled = settings?.vibration ?: true
            }

            SRTAppTheme {
                if (settings != null) {
                    when (currentScreen) {
                        "home" -> {
                            SRTScreen(
                                translationResult = translationResult,
                                onStartCamera = { previewView ->
                                    lastPreviewView = previewView
                                    startCamera(previewView)
                                },
                                onNavigateToSettings = { currentScreen = "settings" },
                                onNavigateToProfile = { currentScreen = "profile" },
                                onCapture = { takePhoto() },
                                onFlipCamera = { toggleCamera() },
                                onTypeBackSend = { text ->
                                    translationResult = text
                                    speak(text)
                                    performTranslationFeedback()
                                    scheduleTranslationClear(text)
                                },
                                showSkeleton = settings.showSkeleton,
                                handLandmarks = handLandmarks,
                                signLanguage = settings.signLanguageStandard,
                                outputLanguage = settings.outputLanguage,
                                textSize = settings.textSize,
                                flashOnTranslation = settings.flashOnTranslation
                            )
                        }
                        "settings" -> {
                            val initialUserSettings = UserSettings(
                                appLanguage = settings.appLanguage,
                                signLanguageStandard = settings.signLanguageStandard,
                                outputLanguage = settings.outputLanguage,
                                showSkeleton = settings.showSkeleton,
                                confidenceThreshold = settings.confidenceThreshold,
                                autoFocus = settings.autoFocus,
                                textSize = settings.textSize,
                                displayDuration = settings.displayDuration,
                                flashOnTranslation = settings.flashOnTranslation,
                                vibration = settings.vibration,
                                isLoggedIn = false,
                                userName = "Guest",
                                userRole = "New User"
                            )
                            SettingsScreen(
                                onBack = { currentScreen = "home" },
                                onNavigateToProfile = { currentScreen = "profile" },
                                initialSettings = initialUserSettings,
                                onSave = { updated ->
                                    scope.launch {
                                        settingsManager.updateAppLanguage(updated.appLanguage)
                                        settingsManager.updateSignLanguageStandard(updated.signLanguageStandard)
                                        settingsManager.updateOutputLanguage(updated.outputLanguage)
                                        settingsManager.updateShowSkeleton(updated.showSkeleton)
                                        settingsManager.updateConfidenceThreshold(updated.confidenceThreshold)
                                        settingsManager.updateAutoFocus(updated.autoFocus)
                                        settingsManager.updateTextSize(updated.textSize)
                                        settingsManager.updateDisplayDuration(updated.displayDuration)
                                        settingsManager.updateFlashOnTranslation(updated.flashOnTranslation)
                                        settingsManager.updateVibration(updated.vibration)
                                        currentScreen = "home"
                                    }
                                },
                                onReset = {
                                    scope.launch {
                                        settingsManager.resetSettings()
                                    }
                                }
                            )
                        }
                        "auth" -> {
                            LoginScreen(
                                repository = userRepository, // 传入全局实例
                                onLoginSuccess = { _, _, _, _, _ -> currentScreen = "profile" },
                                onNavigateToRegister = { currentScreen = "register" }
                            )
                        }
                        "register" -> {
                            com.example.srt_app.ui.RegisterScreen(
                                repository = userRepository, // 传入全局实例
                                onRegisterSuccess = { currentScreen = "profile" },
                                onNavigateToBack = { currentScreen = "auth" }
                            )
                        }
                        "profile" -> {
                            val profile by userRepository.userProfile.collectAsState()
                            ProfileScreen(
                                onNavigateToTranslator = { currentScreen = "home" },
                                onNavigateToSettings = { currentScreen = "settings" },
                                onNavigateToLogin = { currentScreen = "auth" },
                                onNavigateToEditProfile = { currentScreen = "edit_profile" },
                                onLogout = { 
                                    scope.launch { userRepository.logout() }
                                    currentScreen = "home" 
                                },
                                userName = profile.nickname,
                                userRole = if (profile.expertLevel > 0) "Elite Interpreter" else "New User",
                                avatarUrl = profile.avatarUrl, // 传递头像 URL
                                totalTranslations = profile.totalTranslations,
                                accuracy = profile.accuracy
                            )
                        }
                        "edit_profile" -> {
                            EditProfileScreen(
                                repository = userRepository,
                                onBack = { currentScreen = "profile" }
                            )
                        }
                    }
                }
            }
        }
    }

    private fun initDependencies() {
        translator = LightweightTranslator(this)
        translator.initialize()

        // 按顺序初始化
        val database = AppDatabase.getDatabase(this)
        val tokenManager = TokenManager(this)
        userRepository = UserRepository(this, database.userDao(), settingsManager, tokenManager)

        try {
            handLandmarkerHelper = HandLandmarkerHelper(
                context = this,
                handLandmarkerHelperListener = this,
                currentDelegate = HandLandmarkerHelper.DELEGATE_CPU
            )
        } catch (e: Exception) {
            Log.e("MainActivity", "HandLandmarker init failed", e)
        }

        frameAnalyzer = SignFrameAnalyzer(
            handLandmarkerHelper = handLandmarkerHelper,
            isFrontCamera = { lensFacing == CameraSelector.LENS_FACING_FRONT },
            isExternalProcessing = { isProcessing }
        )
    }

    override fun onResults(resultBundle: HandLandmarkerHelper.ResultBundle) {
        runOnUiThread {
            val result = resultBundle.results.firstOrNull()
            handLandmarks = result

            if (result != null && result.landmarks().isNotEmpty()) {
                val landmarks = result.landmarks()
                val wrist = landmarks[0][0]
                val currentWristPos = android.graphics.PointF(wrist.x(), wrist.y())
                
                lastWristPosition?.let { lastPos ->
                    val dx = currentWristPos.x - lastPos.x
                    val dy = currentWristPos.y - lastPos.y
                    handMovingVelocity = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat()
                }
                lastWristPosition = currentWristPos

                if (currentState == RecognitionState.IDLE) currentState = RecognitionState.DETECTING
                if (handMovingVelocity < STABILITY_THRESHOLD) {
                    if (currentState == RecognitionState.DETECTING) currentState = RecognitionState.STABLE
                } else {
                    if (currentState == RecognitionState.LOCKED || currentState == RecognitionState.STABLE) currentState = RecognitionState.DETECTING
                }

                if (!isProcessing) processLandmarks(result)
            } else {
                if (currentState != RecognitionState.IDLE) {
                    currentState = RecognitionState.IDLE
                    lastSpokenWord = ""
                    lastWristPosition = null
                    synchronized(frameBuffer) { frameBuffer.clear() }
                    lastRawLandmarks = null
                }
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        runOnUiThread { Toast.makeText(this, error, Toast.LENGTH_SHORT).show() }
    }

    private fun processLandmarks(result: HandLandmarkerResult) {
        val currentFrameFeatures = extractFeatures(result)
        val smoothedFeatures = smoothFeatures(currentFrameFeatures)
        synchronized(frameBuffer) {
            frameBuffer.add(smoothedFeatures)
            if (frameBuffer.size > windowSize) frameBuffer.removeAt(0)
        }

        if (frameBuffer.size == windowSize && !isProcessing && currentState == RecognitionState.STABLE) {
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastSpeakTime < RECOGNITION_COOLDOWN) return
            isProcessing = true
            runTranslation(prepareInput())
        }
    }

    private fun prepareInput(): FloatArray {
        return synchronized(frameBuffer) {
            val reshaped = FloatArray(3 * windowSize * 21 * 2)
            for (c in 0 until 3) {
                for (t in 0 until windowSize) {
                    for (v in 0 until 21) {
                        for (m in 0 until 2) {
                            val srcIdx = (if (m == 0) 0 else 63) + v * 3 + c
                            val dstIdx = c * (windowSize * 21 * 2) + t * (21 * 2) + v * 2 + m
                            reshaped[dstIdx] = frameBuffer[t][srcIdx]
                        }
                    }
                }
            }
            reshaped
        }
    }

    private fun extractFeatures(result: HandLandmarkerResult): FloatArray {
        val features = FloatArray(featureDimension)
        val allLandmarks = result.landmarks()
        val allHandedness = result.handedness()
        if (allLandmarks.isNotEmpty()) {
            for (i in allLandmarks.indices) {
                val hand = allLandmarks[i]
                val isUserRightHand = if (lensFacing == CameraSelector.LENS_FACING_FRONT) allHandedness[i][0].categoryName() == "Left" else allHandedness[i][0].categoryName() == "Right"
                val start = if (isUserRightHand) 0 else 63
                for (j in 0 until 21) {
                    features[start + j * 3] = hand[j].x()
                    features[start + j * 3 + 1] = hand[j].y()
                    features[start + j * 3 + 2] = hand[j].z()
                }
                if (i >= 1) break 
            }
        }
        return features
    }

    private fun smoothFeatures(current: FloatArray): FloatArray {
        val previous = lastRawLandmarks ?: return current.also { lastRawLandmarks = it }
        val smoothed = FloatArray(featureDimension)
        for (i in 0 until featureDimension) smoothed[i] = current[i] * smoothingFactor + previous[i] * (1f - smoothingFactor)
        lastRawLandmarks = smoothed
        return smoothed
    }

    private fun runTranslation(input: FloatArray) {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val translation = translator.translate(input).text
                if (translation.isNotEmpty() && translation != lastSpokenWord) {
                    translationResult = translation
                    speak(translation)
                    performTranslationFeedback()
                    scheduleTranslationClear(translation)
                    lastSpokenWord = translation
                    lastSpeakTime = System.currentTimeMillis()
                    synchronized(frameBuffer) { frameBuffer.clear() }
                    currentState = RecognitionState.LOCKED
                }
            } finally { isProcessing = false }
        }
    }

    private fun startCamera(previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
            val analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build()
                .also { it.setAnalyzer(cameraExecutor) { proxy -> handLandmarkerHelper?.detectLiveStream(proxy, lensFacing == CameraSelector.LENS_FACING_FRONT) } }
            try {
                cameraProvider.unbindAll()
                activeCamera = cameraProvider.bindToLifecycle(this, CameraSelector.Builder().requireLensFacing(lensFacing).build(), preview, imageCapture, analysis)
                configureAutoFocus(previewView)
            } catch (e: Exception) { Log.e("MainActivity", "Camera binding failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun toggleCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
        lastPreviewView?.let { startCamera(it) }
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return
        val name = SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.US).format(System.currentTimeMillis())
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/SRTApp-Captures")
        }
        capture.takePicture(ImageCapture.OutputFileOptions.Builder(contentResolver, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values).build(), ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
            override fun onError(e: ImageCaptureException) { Toast.makeText(baseContext, "Failed: ${e.message}", Toast.LENGTH_SHORT).show() }
            override fun onImageSaved(r: ImageCapture.OutputFileResults) { Toast.makeText(baseContext, "Saved: ${r.savedUri}", Toast.LENGTH_SHORT).show() }
        })
    }

    private fun configureAutoFocus(previewView: PreviewView) {
        if (!autoFocusEnabled || activeCamera == null) {
            previewView.setOnTouchListener(null)
            return
        }

        previewView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_UP) {
                requestFocusAt(previewView, event.x, event.y)
            }
            true
        }

        previewView.post {
            if (autoFocusEnabled && previewView.width > 0 && previewView.height > 0) {
                requestFocusAt(previewView, previewView.width / 2f, previewView.height / 2f)
            }
        }
    }

    private fun requestFocusAt(previewView: PreviewView, x: Float, y: Float) {
        val camera = activeCamera ?: return
        val point = previewView.meteringPointFactory.createPoint(x, y)
        val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF)
            .setAutoCancelDuration(3, TimeUnit.SECONDS)
            .build()
        camera.cameraControl.startFocusAndMetering(action)
    }

    private fun durationToMillis(duration: String): Long {
        return when (duration) {
            "3s" -> 3000L
            "8s" -> 8000L
            else -> 5000L
        }
    }

    private fun scheduleTranslationClear(text: String) {
        val expectedText = text
        CoroutineScope(Dispatchers.Main).launch {
            delay(displayDurationMillis)
            if (translationResult == expectedText) {
                translationResult = ""
            }
        }
    }

    private fun performTranslationFeedback() {
        if (!vibrationEnabled) return
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            getSystemService(VibratorManager::class.java)?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(VIBRATOR_SERVICE) as? Vibrator
        } ?: return

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(55L, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(55L)
        }
    }

    private fun applyAppLocale(language: String) {
        val languageTag = if (language == "zh") "zh" else "en"
        if (AppCompatDelegate.getApplicationLocales().toLanguageTags() != languageTag) {
            AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(languageTag))
        }
    }

    private fun updateTtsLanguage(language: String) {
        val locale = if (language == "zh") Locale.SIMPLIFIED_CHINESE else Locale.ENGLISH
        tts?.setLanguage(locale)
    }

    private fun initTTS() { tts = TextToSpeech(this) { if (it == TextToSpeech.SUCCESS) tts?.setLanguage(Locale.getDefault()) } }
    private fun speak(t: String) { if (t.isNotEmpty()) tts?.speak(t, TextToSpeech.QUEUE_FLUSH, null, "ID") }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        if (::translator.isInitialized) translator.close()
        handLandmarkerHelper?.clearHandLandmarker()
        tts?.stop(); tts?.shutdown()
    }
}
