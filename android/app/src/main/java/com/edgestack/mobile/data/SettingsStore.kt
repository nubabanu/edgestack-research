package com.edgestack.mobile.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.edgestack.mobile.BuildConfig
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "edgestack_settings")

class SettingsStore(private val context: Context) {
    private val apiUrlKey = stringPreferencesKey("api_url")
    private val demoModeKey = booleanPreferencesKey("demo_mode")
    private val rememberTokenKey = booleanPreferencesKey("remember_token")
    private val sealedTokenKey = stringPreferencesKey("sealed_token")

    val settings: Flow<AppSettings> = context.dataStore.data.map { values ->
        AppSettings(
            apiUrl = values[apiUrlKey] ?: BuildConfig.DEFAULT_API_URL,
            demoMode = values[demoModeKey] ?: true,
            rememberToken = values[rememberTokenKey] ?: false,
        )
    }

    suspend fun readSealedToken(): String? =
        context.dataStore.data.first()[sealedTokenKey]

    suspend fun save(
        apiUrl: String,
        demoMode: Boolean,
        rememberToken: Boolean = false,
        sealedToken: String? = null,
    ) {
        context.dataStore.edit { values ->
            values[apiUrlKey] = apiUrl.trim().removeSuffix("/")
            values[demoModeKey] = demoMode
            values[rememberTokenKey] = rememberToken
            if (rememberToken && sealedToken != null) {
                values[sealedTokenKey] = sealedToken
            } else {
                values.remove(sealedTokenKey)
            }
        }
    }
}
