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
data class HorizonPlan(
    val horizon: String,
    val status: String,
    val title: String,
    @SerialName("holding_period") val holdingPeriod: String,
    @SerialName("entry_rule") val entryRule: String,
    @SerialName("review_rule") val reviewRule: String,
    @SerialName("exit_rule") val exitRule: String,
    @SerialName("recommendation_scope") val recommendationScope: String,
    val symbols: List<String>,
    val evidence: String,
    val invalidation: List<String>,
    @SerialName("unlock_requirement") val unlockRequirement: String,
)

@Serializable
data class AlignmentLayer(
    val horizon: String,
    val status: String,
    val evidence: String,
)

@Serializable
data class SniperPolicy(
    val status: String,
    val objective: String,
    @SerialName("candidate_symbols") val candidateSymbols: List<String>,
    @SerialName("max_name_weight") val maxNameWeight: Double,
    @SerialName("max_gross_exposure") val maxGrossExposure: Double,
    @SerialName("max_planned_loss_per_name_usd") val maxPlannedLossPerNameUsd: Double,
    @SerialName("max_planned_basket_loss_usd") val maxPlannedBasketLossUsd: Double,
    @SerialName("execution_window") val executionWindow: String,
    val alignments: List<AlignmentLayer>,
    @SerialName("hard_vetoes") val hardVetoes: List<String>,
    @SerialName("release_condition") val releaseCondition: String,
    @SerialName("stop_warning") val stopWarning: String,
    @SerialName("validation_status") val validationStatus: String,
)

@Serializable
data class MobileDataGate(
    val name: String,
    val status: String,
    val reason: String,
)

@Serializable
data class MobileLossMetrics(
    val status: String,
    @SerialName("loss_probability") val lossProbability: Double? = null,
    @SerialName("expected_shortfall_95") val expectedShortfall95: Double? = null,
    @SerialName("maximum_adverse_excursion") val maximumAdverseExcursion: Double? = null,
    @SerialName("tenth_percentile_return") val tenthPercentileReturn: Double? = null,
    @SerialName("losing_streak_p90") val losingStreakP90: Double? = null,
)

@Serializable
data class LossAwareV2Summary(
    val namespace: String,
    @SerialName("evidence_status") val evidenceStatus: String,
    @SerialName("selected_horizon") val selectedHorizon: String,
    @SerialName("selected_leverage") val selectedLeverage: Double,
    val ranking: String,
    @SerialName("loss_metrics") val lossMetrics: MobileLossMetrics,
    @SerialName("data_gates") val dataGates: List<MobileDataGate>,
    @SerialName("enabled_event_vetoes") val enabledEventVetoes: List<String>,
    val timing: String,
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
    val horizons: List<HorizonPlan>,
    val sniper: SniperPolicy,
    @SerialName("loss_aware_v2") val lossAwareV2: LossAwareV2Summary,
    val disclaimer: String,
) {
    fun validate(): MobileSnapshot {
        require(meta.schemaVersion == "1.3") { "Unsupported mobile schema" }
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
        require(horizons.map { it.horizon } == listOf("WEEK", "MONTH", "YEAR")) {
            "Horizon plans must contain WEEK, MONTH, YEAR in order"
        }
        require(
            horizons.first().recommendationScope == "BASKET" &&
                horizons.first().symbols == recommendations.map { it.symbol },
        ) { "Weekly horizon must preserve the complete tested basket" }
        require(
            horizons.filter { it.status == "DATA_UNAVAILABLE" }
                .all { it.recommendationScope == "NONE" && it.symbols.isEmpty() },
        ) { "Unavailable horizons cannot emit stock recommendations" }
        require(sniper.alignments.map { it.horizon } == listOf("YEAR", "MONTH", "WEEK", "DAY")) {
            "Sniper alignment must contain YEAR, MONTH, WEEK, DAY"
        }
        require(sniper.candidateSymbols == recommendations.map { it.symbol }) {
            "Sniper watchlist must preserve the complete weekly basket"
        }
        require(
            sniper.status != "CONDITIONAL_PAPER_CANDIDATE" ||
                sniper.alignments.all { it.status == "PASS" },
        ) { "Sniper candidate requires every alignment layer to pass" }
        require(sniper.status != "NO_TRADE" || sniper.hardVetoes.isNotEmpty()) {
            "Sniper no-trade status requires a visible hard veto"
        }
        require(lossAwareV2.dataGates.map { it.name } == listOf("PIT_MEMBERSHIP", "ESTIMATE_VINTAGES", "AUCTION_EXECUTION")) {
            "V2 data gates must be complete and ordered"
        }
        require(
            lossAwareV2.selectedHorizon == "NONE" ||
                lossAwareV2.dataGates.all { it.status == "PASS" },
        ) { "V2 selection requires every data gate to pass" }
        require(
            lossAwareV2.selectedHorizon == "NONE" ||
                lossAwareV2.lossMetrics.status == "AVAILABLE",
        ) { "V2 selection requires loss evidence" }
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
