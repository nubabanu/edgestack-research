package com.edgestack.mobile.data

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.edgestack.mobile.BuildConfig
import kotlinx.coroutines.runBlocking

/**
 * Debug-build-only self tests invocable from adb without touching the UI:
 *
 *   adb shell am broadcast -a com.edgestack.mobile.SELF_TEST \
 *       -n com.edgestack.mobile/.data.DebugSelfTestReceiver
 *
 * exercises the Keystore token vault round trip, and
 *
 *   adb shell am broadcast -a com.edgestack.mobile.SEED_AUTOCONNECT \
 *       -n com.edgestack.mobile/.data.DebugSelfTestReceiver \
 *       --es token <bearer> [--es url http://10.0.2.2:8765]
 *
 * persists exactly what the Setup screen's save would (sealed token,
 * remember on, demo off) so the auto-connect launch path can be verified
 * end-to-end. Results log under "EdgeStackSelfTest"; release builds no-op.
 */
class DebugSelfTestReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (!BuildConfig.DEBUG) return
        if (intent.action == "com.edgestack.mobile.SEED_AUTOCONNECT") {
            val token = intent.getStringExtra("token")
            if (token.isNullOrEmpty()) {
                Log.i("EdgeStackSelfTest", "seed=FAIL missing token extra")
                return
            }
            val url = intent.getStringExtra("url") ?: "http://10.0.2.2:8765"
            val sealed = TokenVault.seal(token)
            if (sealed == null) {
                Log.i("EdgeStackSelfTest", "seed=FAIL keystore seal failed")
                return
            }
            runBlocking {
                SettingsStore(context).save(url, false, rememberToken = true, sealedToken = sealed)
            }
            Log.i("EdgeStackSelfTest", "seed=DONE url=$url tokenLength=${token.length}")
            return
        }
        val sample = "selftest-" + System.nanoTime()
        val sealed = TokenVault.seal(sample)
        val reopened = sealed?.let(TokenVault::open)
        val roundTrip = reopened == sample
        val distinctCiphertext = sealed != null && sealed != TokenVault.seal(sample)
        val tamperRejected = sealed != null &&
            TokenVault.open(sealed.dropLast(4) + "AAAA") == null
        val verdict = if (roundTrip && distinctCiphertext && tamperRejected) {
            "PASS"
        } else {
            "FAIL"
        }
        Log.i(
            "EdgeStackSelfTest",
            "token_vault=$verdict roundTrip=$roundTrip " +
                "distinctCiphertext=$distinctCiphertext tamperRejected=$tamperRejected",
        )
    }
}
