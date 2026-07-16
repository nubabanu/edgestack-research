package com.edgestack.mobile.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ApiMeta(
    @SerialName("schema_version") val schemaVersion: String,
    @SerialName("generated_at") val generatedAt: String,
    @SerialName("market_as_of") val marketAsOf: String,
    val source: String,
    val mode: SnapshotMode,
    val stale: Boolean,
)

@Serializable
enum class SnapshotMode { SEALED, DEMO }

@Serializable
data class EntryInstruction(
    @SerialName("entry_session") val entrySession: String,
    @SerialName("entry_order") val entryOrder: String,
    @SerialName("submit_by_et") val submitByEt: String,
    @SerialName("exit_session") val exitSession: String,
    @SerialName("exit_order") val exitOrder: String,
    @SerialName("no_chase") val noChase: String,
    @SerialName("cancel_if") val cancelIf: List<String>,
)

@Serializable
data class Recommendation(
    @SerialName("recommendation_id") val recommendationId: String,
    val rank: Int,
    val symbol: String,
    val direction: String,
    @SerialName("confidence_ordinal") val confidenceOrdinal: Int,
    @SerialName("signal_close_usd") val signalCloseUsd: Double,
    @SerialName("trailing_return") val trailingReturn: Double,
    @SerialName("suggested_shares") val suggestedShares: Int,
    @SerialName("reference_stop_usd") val referenceStopUsd: Double? = null,
    @SerialName("event_risk") val eventRisk: String,
)

@Serializable
data class HoldoutEvidence(
    val status: String,
    val start: String,
    val end: String,
    val observations: Int,
    @SerialName("expected_sessions") val expectedSessions: Int,
    @SerialName("net_mean") val netMean: Double? = null,
    @SerialName("benchmark_excess_mean") val benchmarkExcessMean: Double? = null,
    @SerialName("terminal_wealth") val terminalWealth: Double? = null,
    @SerialName("benchmark_wealth") val benchmarkWealth: Double? = null,
    @SerialName("freeze_id") val freezeId: String,
    @SerialName("result_sha256") val resultSha256: String,
)

@Serializable
data class PortfolioSummary(
    @SerialName("paper_capital_usd") val paperCapitalUsd: Double,
    @SerialName("target_gross") val targetGross: Double,
    @SerialName("maximum_name_weight") val maximumNameWeight: Double,
    @SerialName("risk_budget_per_name_usd") val riskBudgetPerNameUsd: Double,
    @SerialName("shorts_enabled") val shortsEnabled: Boolean,
)

@Serializable
data class AuditItem(
    @SerialName("occurred_at") val occurredAt: String,
    @SerialName("event_type") val eventType: String,
    val message: String,
)

@Serializable
data class MobileSnapshot(
    val meta: ApiMeta,
    @SerialName("campaign_id") val campaignId: String,
    @SerialName("model_name") val modelName: String,
    @SerialName("model_status") val modelStatus: String,
    @SerialName("bias_tier") val biasTier: String,
    val watermark: String,
    @SerialName("basket_rule") val basketRule: String,
    val instruction: EntryInstruction,
    val portfolio: PortfolioSummary,
    val recommendations: List<Recommendation>,
    val skipped: List<Recommendation> = emptyList(),
    val holdout: HoldoutEvidence,
    val audit: List<AuditItem>,
    val disclaimer: String,
) {
    fun validate(): MobileSnapshot {
        require(meta.schemaVersion == "1.0") { "Unsupported mobile schema" }
        require(recommendations.map { it.rank } == (1..recommendations.size).toList()) {
            "Recommendation ranks are incomplete or reordered"
        }
        require(recommendations.map { it.recommendationId }.distinct().size == recommendations.size) {
            "Duplicate recommendation identity"
        }
        require(modelStatus != "PROMOTED" || holdout.status == "PASS") {
            "Promoted model lacks passed holdout evidence"
        }
        require(portfolio.shortsEnabled || recommendations.none { it.direction == "SHORT" }) {
            "Short recommendation emitted while shorts are disabled"
        }
        require(recommendations.all { it.confidenceOrdinal in 0..100 && it.suggestedShares >= 0 })
        return this
    }
}

data class AppSettings(val apiUrl: String, val demoMode: Boolean)

enum class SnapshotOrigin { NETWORK, CACHE, DEMO }

data class SnapshotResult(
    val snapshot: MobileSnapshot,
    val origin: SnapshotOrigin,
    val warning: String? = null,
)
