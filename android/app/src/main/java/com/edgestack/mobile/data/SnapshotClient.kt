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

    suspend fun probe(baseUrl: String, bearerToken: String): ConnectionProbe =
        withContext(Dispatchers.IO) {
            val normalized = try {
                validateBaseUrl(baseUrl)
            } catch (error: IllegalArgumentException) {
                return@withContext ConnectionProbe(
                    ok = false,
                    serverReachable = false,
                    mode = null,
                    tokenAccepted = null,
                    message = error.message ?: "Invalid API URL",
                )
            }
            val health = try {
                httpGet("$normalized/api/v1/health", token = null)
            } catch (error: Exception) {
                return@withContext ConnectionProbe(
                    ok = false,
                    serverReachable = false,
                    mode = null,
                    tokenAccepted = null,
                    message = "Server unreachable (${error.message?.take(80) ?: "no route"}). " +
                        "Check: same Wi-Fi (not mobile data), server window still running, " +
                        "firewall rule for port added.",
                )
            }
            val mode = if (health.second.contains("\"demo\"")) "demo" else "sealed"
            val snapshot = try {
                httpGet("$normalized/api/v1/mobile/snapshot", token = bearerToken)
            } catch (error: Exception) {
                return@withContext buildProbe(mode, snapshotStatus = -1)
            }
            buildProbe(mode, snapshotStatus = snapshot.first)
        }

    private fun httpGet(url: String, token: String?): Pair<Int, String> {
        val connection = URL(url).openConnection() as HttpURLConnection
        return try {
            connection.requestMethod = "GET"
            connection.connectTimeout = 6_000
            connection.readTimeout = 8_000
            connection.setRequestProperty("Accept", "application/json")
            token?.let { connection.setRequestProperty("Authorization", "Bearer $it") }
            connection.instanceFollowRedirects = false
            val status = connection.responseCode
            val body = (if (status in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader()?.use { it.readText().take(500) } ?: ""
            status to body
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
            require(uri.host != null) { "API URL must include a host" }
            require(uri.scheme == "https" || (uri.scheme == "http" && isPrivateHost(uri.host))) {
                "Use HTTPS; cleartext is allowed only for private LAN, Tailscale, or loopback addresses"
            }
            return normalized
        }

        /** Loopback, RFC 1918 private LAN, and the Tailscale CGNAT range. */
        fun isPrivateHost(host: String): Boolean {
            if (host in setOf("localhost", "127.0.0.1", "10.0.2.2")) return true
            val octets = host.split(".").mapNotNull { it.toIntOrNull() }
            if (octets.size != 4 || octets.any { it !in 0..255 }) return false
            return when {
                octets[0] == 10 -> true
                octets[0] == 192 && octets[1] == 168 -> true
                octets[0] == 172 && octets[1] in 16..31 -> true
                octets[0] == 100 && octets[1] in 64..127 -> true
                octets[0] == 127 -> true
                else -> false
            }
        }

        fun buildProbe(mode: String, snapshotStatus: Int): ConnectionProbe = when {
            snapshotStatus == 200 -> ConnectionProbe(
                ok = true,
                serverReachable = true,
                mode = mode,
                tokenAccepted = true,
                message = "Connected · server is ${mode.uppercase()} and the token is accepted. Save and refresh.",
            )
            snapshotStatus == 401 -> ConnectionProbe(
                ok = false,
                serverReachable = true,
                mode = mode,
                tokenAccepted = false,
                message = "Server reached, but the bearer token was rejected. Re-enter the exact token from the server.",
            )
            snapshotStatus == 503 -> ConnectionProbe(
                ok = false,
                serverReachable = true,
                mode = mode,
                tokenAccepted = true,
                message = "Server reached and token accepted, but no sealed campaign snapshot is available yet.",
            )
            else -> ConnectionProbe(
                ok = false,
                serverReachable = true,
                mode = mode,
                tokenAccepted = null,
                message = "Server reached but the snapshot request failed (HTTP $snapshotStatus).",
            )
        }
    }
}

data class ConnectionProbe(
    val ok: Boolean,
    val serverReachable: Boolean,
    val mode: String?,
    val tokenAccepted: Boolean?,
    val message: String,
)

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
