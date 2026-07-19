package com.edgestack.mobile.data

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class SnapshotCodecTest {
    @Test
    fun `valid snapshot preserves ordered basket`() {
        val snapshot = SnapshotCodec.decode(validPayload())
        assertEquals(listOf(1, 2), snapshot.recommendations.map { it.rank })
        assertEquals(SnapshotMode.SEALED, snapshot.meta.mode)
    }

    @Test
    fun `reordered rank fails closed`() {
        val root = Json.parseToJsonElement(validPayload()).jsonObject.toMutableMap()
        val recommendations = root.getValue("recommendations").jsonArray.toMutableList()
        val first = recommendations.first().jsonObject.toMutableMap()
        first["rank"] = JsonPrimitive(2)
        recommendations[0] = JsonObject(first)
        root["recommendations"] = kotlinx.serialization.json.JsonArray(recommendations)
        assertThrows(IllegalArgumentException::class.java) {
            SnapshotCodec.decode(JsonObject(root).toString())
        }
    }

    @Test
    fun `available timing advisor without calendar rows fails closed`() {
        val root = Json.parseToJsonElement(validPayload()).jsonObject.toMutableMap()
        val timing = root.getValue("timing").jsonObject.toMutableMap()
        timing["calendar"] = kotlinx.serialization.json.JsonArray(emptyList())
        root["timing"] = JsonObject(timing)
        assertThrows(IllegalArgumentException::class.java) {
            SnapshotCodec.decode(JsonObject(root).toString())
        }
    }

    @Test
    fun `unknown fields are rejected`() {
        val root = Json.parseToJsonElement(validPayload()).jsonObject.toMutableMap()
        root["broker_order"] = JsonPrimitive("BUY")
        assertThrows(Exception::class.java) { SnapshotCodec.decode(JsonObject(root).toString()) }
    }

    @Test
    fun `oil snapshot preserves five isolated paper risk lanes`() {
        val root = Json.parseToJsonElement(validPayload()).jsonObject.toMutableMap()
        root["oil"] = oilPayload()

        val oil = SnapshotCodec.decode(JsonObject(root).toString()).oil!!

        assertEquals("NO_TRADE", oil.status)
        assertEquals(
            listOf("GOVERNED_0_5", "CHALLENGE_1", "CHALLENGE_2", "CHALLENGE_5", "CHALLENGE_10"),
            oil.intraday.lanes.map { it.name },
        )
        assertTrue(oil.intraday.lanes.last().label.contains("HIGH_RISK_NON_PROMOTABLE"))
    }

    @Test
    fun `remote demo stays visibly demo and is never classified as sealed network evidence`() {
        val sealed = SnapshotCodec.decode(validPayload())
        val demo = sealed.copy(
            meta = sealed.meta.copy(mode = SnapshotMode.DEMO),
            modelStatus = "DEMO",
        )

        val result = networkSnapshotResult(demo)

        assertEquals(SnapshotOrigin.DEMO, result.origin)
        assertTrue(result.warning.orEmpty().contains("not sealed evidence"))
    }

    private fun validPayload(): String = """
        {
          "meta":{"schema_version":"1.6","generated_at":"2026-07-16T12:00:00Z","market_as_of":"2026-07-15_CLOSE","source":"test","mode":"SEALED","stale":false},
          "campaign_id":"campaign","model_name":"model","model_status":"PROMOTED","bias_tier":"SURVIVORSHIP_BIASED","watermark":"SURVIVORSHIP_BIASED","basket_rule":"both names are required",
          "instruction":{"entry_session":"2026-07-16","entry_order":"MOC","submit_by_et":"15:45 ET","exit_session":"2026-07-23","exit_order":"MOC","no_chase":"wait","cancel_if":["stale"]},
          "portfolio":{"paper_capital_usd":100000.0,"target_gross":0.5,"maximum_name_weight":0.1,"risk_budget_per_name_usd":500.0,"shorts_enabled":false},
          "recommendations":[
            {"recommendation_id":"one","rank":1,"symbol":"AAA","direction":"LONG","confidence_ordinal":70,"signal_close_usd":10.0,"trailing_return":-0.1,"suggested_shares":10,"reference_stop_usd":8.0,"event_risk":"HIGH"},
            {"recommendation_id":"two","rank":2,"symbol":"BBB","direction":"LONG","confidence_ordinal":65,"signal_close_usd":20.0,"trailing_return":-0.08,"suggested_shares":5,"reference_stop_usd":17.0,"event_risk":"HIGH"}
          ],
          "skipped":[],
          "holdout":{"status":"PASS","start":"2023-01-01","end":"2026-01-01","observations":750,"expected_sessions":750,"net_mean":0.001,"benchmark_excess_mean":0.0002,"terminal_wealth":1.2,"benchmark_wealth":1.1,"freeze_id":"freeze","result_sha256":"hash"},
          "audit":[],
          "horizons":[
            {"horizon":"WEEK","status":"CONDITIONAL_PAPER_SIGNAL","title":"weekly basket","holding_period":"5 sessions","entry_rule":"MOC","review_rule":"daily","exit_rule":"MOC","recommendation_scope":"BASKET","symbols":["AAA","BBB"],"evidence":"passed","invalidation":["stale"],"unlock_requirement":"unlocked"},
            {"horizon":"MONTH","status":"DATA_UNAVAILABLE","title":"no monthly model","holding_period":"21 sessions","entry_rule":"none","review_rule":"new study","exit_rule":"none","recommendation_scope":"NONE","symbols":[],"evidence":"unavailable","invalidation":["inference invalid"],"unlock_requirement":"new holdout"},
            {"horizon":"YEAR","status":"DATA_UNAVAILABLE","title":"no annual model","holding_period":"252 sessions","entry_rule":"none","review_rule":"new study","exit_rule":"none","recommendation_scope":"NONE","symbols":[],"evidence":"unavailable","invalidation":["inference invalid"],"unlock_requirement":"new holdout"}
          ],
          "sniper":{"status":"NO_TRADE","objective":"LOSS_FIRST","candidate_symbols":["AAA","BBB"],"max_name_weight":0.05,"max_gross_exposure":0.25,"max_planned_loss_per_name_usd":100.0,"max_planned_basket_loss_usd":500.0,"execution_window":"pre-close","alignments":[{"horizon":"YEAR","status":"UNVALIDATED","evidence":"none"},{"horizon":"MONTH","status":"UNVALIDATED","evidence":"none"},{"horizon":"WEEK","status":"PASS","evidence":"passed"},{"horizon":"DAY","status":"PENDING","evidence":"pending"}],"hard_vetoes":["UNVALIDATED"],"release_condition":"all pass","stop_warning":"not guaranteed","validation_status":"RISK_OVERLAY_NOT_VALIDATED_ALPHA"},
          "loss_aware_v2":{"namespace":"loss-aware-v2","evidence_status":"FORWARD_REQUIRED","selected_horizon":"NONE","selected_leverage":1.0,"ranking":"LOSS_FIRST","loss_metrics":{"status":"DATA_UNAVAILABLE","loss_probability":null,"expected_shortfall_95":null,"maximum_adverse_excursion":null,"tenth_percentile_return":null,"losing_streak_p90":null},"data_gates":[{"name":"PIT_MEMBERSHIP","status":"DATA_UNAVAILABLE","reason":"missing"},{"name":"ESTIMATE_VINTAGES","status":"DATA_UNAVAILABLE","reason":"missing"},{"name":"AUCTION_EXECUTION","status":"DATA_UNAVAILABLE","reason":"missing"}],"enabled_event_vetoes":[],"timing":"NO TRADE"},
          "timing":{"status":"AVAILABLE","symbol":"SPY","as_of_session":"2026-07-15","policy":"reliability-weighted","anchors":{"status":"TWO_ANCHORS_ONLY","best_buy_anchor":"CLOSE_AUCTION","matching_sell_anchor":"next OPEN_AUCTION","overnight":{"n":5000,"mean_daily_bp":3.2,"hit_rate":0.55},"intraday":{"n":5000,"mean_daily_bp":1.8,"hit_rate":0.54},"finer_granularity":"DATA_UNAVAILABLE"},"calendar":[{"session":"2026-07-17","weekday":"FRI","win_score":55,"expected_daily_bp":2.4,"conditions":["weekday=FRI"]}],"diagnostic_watermark":"DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER"},
          "timing_symbols":[{"status":"AVAILABLE","symbol":"SPY","as_of_session":"2026-07-15","policy":"reliability-weighted","anchors":null,"calendar":[{"session":"2026-07-17","weekday":"FRI","win_score":55,"expected_daily_bp":2.4,"conditions":["weekday=FRI"]}],"diagnostic_watermark":"DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER"},{"status":"AVAILABLE","symbol":"QQQ","as_of_session":"2026-07-15","policy":"reliability-weighted","anchors":null,"calendar":[{"session":"2026-07-17","weekday":"FRI","win_score":56,"expected_daily_bp":2.9,"conditions":["weekday=FRI"]}],"diagnostic_watermark":"DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER"}],
          "disclaimer":"research only"
        }
    """.trimIndent()

    private fun oilPayload(): JsonObject {
        val lanes = """
            [
              {"name":"GOVERNED_0_5","label":"Governed research lane","risk_fraction":0.005,"status":"ACTIVE","equity_usd":100000.0,"peak_equity_usd":100000.0,"drawdown_fraction":0.0,"leverage":10.0,"notional_usd":10000.0,"margin_usd":1000.0,"stop_fraction":0.05,"stressed_move_fraction":0.06,"maximum_planned_loss_usd":500.0,"estimated_cost_usd":10.0,"reason":"risk-sized"},
              {"name":"CHALLENGE_1","label":"Challenge 1%","risk_fraction":0.01,"status":"ACTIVE","equity_usd":100000.0,"peak_equity_usd":100000.0,"drawdown_fraction":0.0,"leverage":5.0,"notional_usd":20000.0,"margin_usd":4000.0,"stop_fraction":0.05,"stressed_move_fraction":0.06,"maximum_planned_loss_usd":1000.0,"estimated_cost_usd":20.0,"reason":"risk-sized"},
              {"name":"CHALLENGE_2","label":"Challenge 2%","risk_fraction":0.02,"status":"ACTIVE","equity_usd":100000.0,"peak_equity_usd":100000.0,"drawdown_fraction":0.0,"leverage":5.0,"notional_usd":40000.0,"margin_usd":8000.0,"stop_fraction":0.05,"stressed_move_fraction":0.06,"maximum_planned_loss_usd":2000.0,"estimated_cost_usd":40.0,"reason":"risk-sized"},
              {"name":"CHALLENGE_5","label":"Challenge 5%","risk_fraction":0.05,"status":"ACTIVE","equity_usd":100000.0,"peak_equity_usd":100000.0,"drawdown_fraction":0.0,"leverage":5.0,"notional_usd":100000.0,"margin_usd":20000.0,"stop_fraction":0.05,"stressed_move_fraction":0.06,"maximum_planned_loss_usd":5000.0,"estimated_cost_usd":100.0,"reason":"risk-sized"},
              {"name":"CHALLENGE_10","label":"HIGH_RISK_NON_PROMOTABLE · 10% account risk","risk_fraction":0.10,"status":"ACTIVE","equity_usd":100000.0,"peak_equity_usd":100000.0,"drawdown_fraction":0.0,"leverage":5.0,"notional_usd":200000.0,"margin_usd":40000.0,"stop_fraction":0.05,"stressed_move_fraction":0.06,"maximum_planned_loss_usd":10000.0,"estimated_cost_usd":200.0,"reason":"risk-sized"}
            ]
        """.trimIndent()
        val horizon = """
            {"horizon":"INTRADAY","status":"NO_TRADE","evidence_status":"FORWARD_REQUIRED","direction":"LONG","proxy_symbol":"USO","signal_session":"2026-07-17","planned_entry":"2026-07-20 USO open","planned_exit":"2026-07-20 USO close","reference_price_usd":80.0,"atr14_usd":2.0,"p99_adverse_gap_fraction":0.01,"active_vetoes":["FROZEN_BASELINE"],"reasons":["forward required"],"lanes":$lanes}
        """.trimIndent()
        val swing = horizon
            .replace("\"INTRADAY\"", "\"SWING_3D\"")
            .replace("2026-07-20 USO close", "2026-07-23 USO close after three sessions")
        return Json.parseToJsonElement(
            """
            {"schema_version":"1.0","campaign_id":"oil-paper-v1","decision_id":"oil-decision","generated_at":"2026-07-20T12:30:00Z","market_as_of":"2026-07-17","status":"NO_TRADE","watermark":"PAPER_ONLY_NOT_AN_ORDER","outcome_proxy":"USO","basis_warning":"USO is not the eToro rolling WTI CFD","proxy_agreement":"BULLISH","data_gates":[{"name":"PROXY_BARS","status":"PASS","as_of":"2026-07-20T12:30:00Z","reason":"causal","raw_sha256":[]}],"intraday":$horizon,"swing":$swing,"provenance_warnings":[],"disclaimer":"research only"}
            """.trimIndent(),
        ).jsonObject
    }
}
