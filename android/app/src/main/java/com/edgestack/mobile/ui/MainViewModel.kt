package com.edgestack.mobile.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.edgestack.mobile.data.AppSettings
import com.edgestack.mobile.data.ConnectionProbe
import com.edgestack.mobile.data.EdgeStackRepository
import com.edgestack.mobile.data.MobileSnapshot
import com.edgestack.mobile.data.SettingsStore
import com.edgestack.mobile.data.SnapshotOrigin
import com.edgestack.mobile.data.SnapshotResult
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class MainUiState(
    val loading: Boolean = true,
    val snapshot: MobileSnapshot? = null,
    val origin: SnapshotOrigin = SnapshotOrigin.DEMO,
    val warning: String? = null,
    val fatalError: String? = null,
    val settings: AppSettings? = null,
    val token: String = "",
    val probing: Boolean = false,
    val probe: ConnectionProbe? = null,
    // Unsaved Setup edits; they survive tab switches so a toggle or URL edit
    // is never silently reverted before "Save and refresh".
    val draftApiUrl: String? = null,
    val draftDemo: Boolean? = null,
)

class MainViewModel(
    private val repository: EdgeStackRepository,
    private val settingsStore: SettingsStore,
) : ViewModel() {
    private val mutableState = MutableStateFlow(MainUiState())
    val state: StateFlow<MainUiState> = mutableState.asStateFlow()
    private var refreshJob: Job? = null
    private var loadedInitialSettings = false

    init {
        viewModelScope.launch {
            settingsStore.settings.collect { settings ->
                mutableState.update { it.copy(settings = settings) }
                if (!loadedInitialSettings) {
                    loadedInitialSettings = true
                    refresh()
                }
            }
        }
    }

    fun setToken(value: String) {
        mutableState.update { it.copy(token = value) }
    }

    fun setDraftApiUrl(value: String) {
        mutableState.update { it.copy(draftApiUrl = value) }
    }

    fun setDraftDemo(value: Boolean) {
        mutableState.update { it.copy(draftDemo = value) }
    }

    fun testConnection(apiUrl: String, token: String) {
        setToken(token)
        viewModelScope.launch {
            mutableState.update { it.copy(probing = true, probe = null) }
            val result = runCatching { repository.probe(apiUrl, token) }
                .getOrElse { error ->
                    ConnectionProbe(
                        ok = false,
                        serverReachable = false,
                        mode = null,
                        tokenAccepted = null,
                        message = error.message ?: "Connection test failed",
                    )
                }
            mutableState.update { it.copy(probing = false, probe = result) }
        }
    }

    fun saveSettings(apiUrl: String, demoMode: Boolean, token: String) {
        setToken(token)
        viewModelScope.launch {
            runCatching { settingsStore.save(apiUrl, demoMode) }
                .onSuccess {
                    mutableState.update {
                        it.copy(draftApiUrl = null, draftDemo = null)
                    }
                    refresh(AppSettings(apiUrl.trim().removeSuffix("/"), demoMode))
                }
                .onFailure { error ->
                    mutableState.update { it.copy(fatalError = error.message) }
                }
        }
    }

    fun refresh(settingsOverride: AppSettings? = null) {
        val settings = settingsOverride ?: mutableState.value.settings ?: return
        refreshJob?.cancel()
        refreshJob = viewModelScope.launch {
            mutableState.update { it.copy(loading = true, fatalError = null) }
            runCatching { repository.load(settings, mutableState.value.token) }
                .onSuccess(::show)
                .onFailure { error ->
                    mutableState.update {
                        it.copy(loading = false, fatalError = error.message ?: "Unknown error")
                    }
                }
        }
    }

    private fun show(result: SnapshotResult) {
        mutableState.update {
            it.copy(
                loading = false,
                snapshot = result.snapshot,
                origin = result.origin,
                warning = result.warning,
                fatalError = null,
            )
        }
    }

    class Factory(
        private val repository: EdgeStackRepository,
        private val settingsStore: SettingsStore,
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            require(modelClass.isAssignableFrom(MainViewModel::class.java))
            return MainViewModel(repository, settingsStore) as T
        }
    }
}
