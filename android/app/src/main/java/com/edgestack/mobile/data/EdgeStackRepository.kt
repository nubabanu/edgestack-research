package com.edgestack.mobile.data

import android.content.Context
import com.edgestack.mobile.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class EdgeStackRepository(
    private val context: Context,
    private val client: SnapshotClient = SnapshotClient(),
) {
    private val cacheFile: File get() = File(context.filesDir, "last_sealed_snapshot.json")

    suspend fun load(settings: AppSettings, bearerToken: String): SnapshotResult {
        if (settings.demoMode) return demo()
        return runCatching {
            require(bearerToken.length >= 24) { "Enter the 24+ character API bearer token" }
            val raw = client.fetch(settings.apiUrl, bearerToken)
            val snapshot = SnapshotCodec.decode(raw)
            if (snapshot.meta.mode == SnapshotMode.SEALED) {
                withContext(Dispatchers.IO) { cacheFile.writeText(raw, Charsets.UTF_8) }
            }
            networkSnapshotResult(snapshot)
        }.getOrElse { failure ->
            val cached = runCatching {
                withContext(Dispatchers.IO) { cacheFile.readText(Charsets.UTF_8) }
            }.mapCatching(SnapshotCodec::decode).getOrNull()
            if (cached != null) {
                SnapshotResult(
                    cached,
                    SnapshotOrigin.CACHE,
                    "Network refresh failed; showing last sealed snapshot. ${failure.message}",
                )
            } else {
                demo().copy(warning = "No sealed cache is available. ${failure.message}")
            }
        }
    }

    private suspend fun demo(): SnapshotResult = withContext(Dispatchers.IO) {
        val raw = context.resources.openRawResource(R.raw.demo_snapshot)
            .bufferedReader()
            .use { it.readText() }
        SnapshotResult(SnapshotCodec.decode(raw), SnapshotOrigin.DEMO)
    }
}

internal fun networkSnapshotResult(snapshot: MobileSnapshot): SnapshotResult =
    when (snapshot.meta.mode) {
        SnapshotMode.SEALED -> SnapshotResult(snapshot, SnapshotOrigin.NETWORK)
        SnapshotMode.DEMO -> SnapshotResult(
            snapshot,
            SnapshotOrigin.DEMO,
            "Connected to the API demonstration endpoint; this is not sealed evidence.",
        )
    }
