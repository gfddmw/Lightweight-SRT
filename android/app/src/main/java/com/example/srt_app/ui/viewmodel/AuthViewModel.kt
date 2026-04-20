package com.example.srt_app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.srt_app.data.User
import com.example.srt_app.data.UserRepository
import com.example.srt_app.data.AuthData
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class AuthViewModel(private val repository: UserRepository) : ViewModel() {

    private val _authState = MutableStateFlow<AuthState>(AuthState.Idle)
    val authState: StateFlow<AuthState> = _authState

    fun login(email: String, password: String) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            val trimmedEmail = email.trim()
            val result = repository.login(trimmedEmail, password)
            result.onSuccess { authData: AuthData ->
                _authState.value = AuthState.Success(authData.user, authData.accessToken, authData.refreshToken)
            }.onFailure { error ->
                _authState.value = AuthState.Error(error.message ?: "Login failed")
            }
        }
    }

    fun sendVerificationCode(email: String) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            val trimmedEmail = email.trim()
            val result = repository.sendVerificationCode(trimmedEmail)
            result.onSuccess { token ->
                _authState.value = AuthState.CodeSent(token)
            }.onFailure { error ->
                _authState.value = AuthState.Error(error.message ?: "Failed to send code")
            }
        }
    }

    fun loginWithCode(email: String, code: String, token: String) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            val trimmedEmail = email.trim()
            val result = repository.loginWithCode(trimmedEmail, code, token)
            result.onSuccess { authData: AuthData ->
                _authState.value = AuthState.Success(authData.user, authData.accessToken, authData.refreshToken)
            }.onFailure { error ->
                _authState.value = AuthState.Error(error.message ?: "Login failed")
            }
        }
    }

    fun register(username: String, email: String, password: String, verificationCode: String, verificationToken: String) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            val trimmedEmail = email.trim()
            val user = User(username = username, email = trimmedEmail, password = password)
            val result = repository.register(user, verificationCode, verificationToken)
            result.onSuccess { authData: AuthData ->
                _authState.value = AuthState.Success(authData.user, authData.accessToken, authData.refreshToken)
            }.onFailure { error ->
                _authState.value = AuthState.Error(error.message ?: "Registration failed")
            }
        }
    }

    fun resetState() {
        _authState.value = AuthState.Idle
    }
}

sealed class AuthState {
    object Idle : AuthState()
    object Loading : AuthState()
    data class CodeSent(val token: String) : AuthState()
    data class Success(val user: User, val accessToken: String = "", val refreshToken: String = "") : AuthState()
    data class Error(val message: String) : AuthState()
}
