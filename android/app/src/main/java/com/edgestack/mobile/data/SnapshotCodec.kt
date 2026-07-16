package com.edgestack.mobile.data

import kotlinx.serialization.json.Json

object SnapshotCodec {
    private val json = Json {
        ignoreUnknownKeys = false
        isLenient = false
        explicitNulls = true
    }

    fun decode(payload: String): MobileSnapshot =
        json.decodeFromString<MobileSnapshot>(payload).validate()
}
