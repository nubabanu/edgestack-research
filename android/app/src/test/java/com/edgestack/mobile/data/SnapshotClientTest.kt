package com.edgestack.mobile.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class SnapshotClientTest {
    @Test
    fun `https endpoint is accepted and normalized`() {
        assertEquals("https://api.example.com", SnapshotClient.validateBaseUrl("https://api.example.com/"))
    }

    @Test
    fun `emulator loopback may use cleartext`() {
        assertEquals("http://10.0.2.2:8765", SnapshotClient.validateBaseUrl("http://10.0.2.2:8765"))
    }

    @Test
    fun `remote cleartext and embedded credentials are rejected`() {
        assertThrows(IllegalArgumentException::class.java) {
            SnapshotClient.validateBaseUrl("http://api.example.com")
        }
        assertThrows(IllegalArgumentException::class.java) {
            SnapshotClient.validateBaseUrl("https://user:secret@api.example.com")
        }
    }
}
