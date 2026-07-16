package com.edgestack.mobile.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
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

    @Test
    fun `private lan and tailscale addresses may use cleartext`() {
        assertEquals(
            "http://192.168.2.100:8765",
            SnapshotClient.validateBaseUrl("http://192.168.2.100:8765"),
        )
        assertEquals(
            "http://10.1.2.3:8765",
            SnapshotClient.validateBaseUrl("http://10.1.2.3:8765"),
        )
        assertEquals(
            "http://172.20.0.5:8765",
            SnapshotClient.validateBaseUrl("http://172.20.0.5:8765"),
        )
        assertEquals(
            "http://100.101.102.103:8765",
            SnapshotClient.validateBaseUrl("http://100.101.102.103:8765"),
        )
    }

    @Test
    fun `public and near-miss addresses still require https`() {
        // 100.63.x is outside the Tailscale CGNAT range, 172.32.x outside RFC 1918.
        for (host in listOf("100.63.0.1", "172.32.0.1", "8.8.8.8", "192.169.0.1")) {
            assertThrows(IllegalArgumentException::class.java) {
                SnapshotClient.validateBaseUrl("http://$host:8765")
            }
        }
    }

    @Test
    fun `probe verdicts map snapshot status to actionable guidance`() {
        val connected = SnapshotClient.buildProbe("sealed", snapshotStatus = 200)
        assertTrue(connected.ok)
        assertEquals(true, connected.tokenAccepted)

        val badToken = SnapshotClient.buildProbe("sealed", snapshotStatus = 401)
        assertTrue(!badToken.ok && badToken.serverReachable)
        assertEquals(false, badToken.tokenAccepted)
        assertTrue(badToken.message.contains("token"))

        val noCampaign = SnapshotClient.buildProbe("sealed", snapshotStatus = 503)
        assertTrue(!noCampaign.ok && noCampaign.serverReachable)
        assertTrue(noCampaign.message.contains("campaign"))
    }
}
