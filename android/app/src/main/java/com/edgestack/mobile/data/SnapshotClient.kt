package com.edgestack.mobile.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL

class SnapshotClient {
    suspend fun fetch(baseUrl: String, bearerToken: String): String = withContext(Dispatchers.IO) {
        val normalized = validateBaseUrl(baseUrl)
        val connection = URL("$normalized/api/v1/mobile/snapshot").openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "GET"
            connection.connectTimeout = 8_000
            connection.readTimeout = 12_000
            connection.setRequestProperty("Accept", "application/json")
            connection.setRequestProperty("Authorization", "Bearer $bearerToken")
            connection.setRequestProperty("User-Agent", "EdgeStack-Android/1.0")
            connection.instanceFollowRedirects = false
            val status = connection.responseCode
            if (status != HttpURLConnection.HTTP_OK) {
                val detail = connection.errorStream?.bufferedReader()?.use { it.readText().take(240) }
                error("API returned HTTP $status${detail?.let { ": $it" } ?: ""}")
            }
            val bytes = connection.inputStream.use { it.readLimited(MAX_BYTES) }
            bytes.toString(Charsets.UTF_8)
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        private const val MAX_BYTES = 2_000_000

        fun validateBaseUrl(value: String): String {
            val normalized = value.trim().removeSuffix("/")
            val uri = URI(normalized)
            require(uri.userInfo == null && uri.query == null && uri.fragment == null) {
                "API URL must not contain credentials, query, or fragment"
            }
            val local = uri.host in setOf("10.0.2.2", "localhost", "127.0.0.1")
            require(uri.scheme == "https" || (uri.scheme == "http" && local)) {
                "Use HTTPS; cleartext is allowed only for local development"
            }
            require(uri.host != null) { "API URL must include a host" }
            return normalized
        }
    }
}

private fun InputStream.readLimited(maxBytes: Int): ByteArray {
    val output = ByteArrayOutputStream()
    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
    var total = 0
    while (true) {
        val count = read(buffer)
        if (count < 0) break
        total += count
        require(total <= maxBytes) { "Snapshot exceeds 2 MB safety limit" }
        output.write(buffer, 0, count)
    }
    return output.toByteArray()
}
