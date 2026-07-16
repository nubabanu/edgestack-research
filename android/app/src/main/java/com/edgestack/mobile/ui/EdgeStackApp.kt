package com.edgestack.mobile.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ShowChart
import androidx.compose.material.icons.outlined.Assessment
import androidx.compose.material.icons.outlined.CloudOff
import androidx.compose.material.icons.outlined.DateRange
import androidx.compose.material.icons.outlined.Refresh
import androidx.compose.material.icons.outlined.Science
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.Shield
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.edgestack.mobile.data.MobileSnapshot
import com.edgestack.mobile.data.HorizonPlan
import com.edgestack.mobile.data.Recommendation
import com.edgestack.mobile.data.SnapshotOrigin
import com.edgestack.mobile.data.SniperPolicy
import com.edgestack.mobile.ui.theme.Coral
import com.edgestack.mobile.ui.theme.Fog
import com.edgestack.mobile.ui.theme.Gold
import com.edgestack.mobile.ui.theme.Ink
import com.edgestack.mobile.ui.theme.Mint
import com.edgestack.mobile.ui.theme.PanelSoft
import java.text.NumberFormat
import java.util.Locale

private enum class AppTab(val label: String, val icon: ImageVector) {
    PLAN("Plan", Icons.AutoMirrored.Outlined.ShowChart),
    BASKET("Basket", Icons.Outlined.Assessment),
    HORIZONS("Sniper", Icons.Outlined.DateRange),
    EVIDENCE("Evidence", Icons.Outlined.Shield),
    SETUP("Setup", Icons.Outlined.Settings),
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EdgeStackApp(viewModel: MainViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var tab by remember { mutableStateOf(AppTab.PLAN) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("EDGESTACK", fontWeight = FontWeight.Black, letterSpacing = 2.sp)
                        Text(
                            "paper research companion",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                actions = {
                    if (state.origin != SnapshotOrigin.NETWORK) {
                        Icon(
                            Icons.Outlined.CloudOff,
                            contentDescription = "Offline or demo",
                            tint = Gold,
                        )
                    }
                    IconButton(onClick = viewModel::refresh, enabled = !state.loading) {
                        Icon(Icons.Outlined.Refresh, contentDescription = "Refresh")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = Ink),
            )
        },
        bottomBar = {
            NavigationBar(containerColor = MaterialTheme.colorScheme.surface) {
                AppTab.entries.forEach { item ->
                    NavigationBarItem(
                        selected = tab == item,
                        onClick = { tab = item },
                        icon = { Icon(item.icon, contentDescription = item.label) },
                        label = { Text(item.label) },
                    )
                }
            }
        },
    ) { padding ->
        Box(Modifier.fillMaxSize().padding(padding)) {
            when {
                state.snapshot != null -> SnapshotContent(
                    snapshot = state.snapshot!!,
                    tab = tab,
                    state = state,
                    viewModel = viewModel,
                )
                state.loading -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                else -> EmptyState(state.fatalError ?: "No snapshot available") {
                    tab = AppTab.SETUP
                }
            }
            if (state.loading && state.snapshot != null) {
                CircularProgressIndicator(
                    modifier = Modifier.align(Alignment.TopEnd).padding(12.dp).size(24.dp),
                    strokeWidth = 2.dp,
                )
            }
        }
    }
}

@Composable
private fun SnapshotContent(
    snapshot: MobileSnapshot,
    tab: AppTab,
    state: MainUiState,
    viewModel: MainViewModel,
) {
    when (tab) {
        AppTab.PLAN -> PlanScreen(snapshot, state)
        AppTab.BASKET -> BasketScreen(snapshot)
        AppTab.HORIZONS -> HorizonsScreen(snapshot)
        AppTab.EVIDENCE -> EvidenceScreen(snapshot)
        AppTab.SETUP -> SetupScreen(state, viewModel)
    }
}

@Composable
private fun HorizonsScreen(snapshot: MobileSnapshot) {
    BaseList(snapshot) {
        item {
            Text("Sniper decision", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Text("Loss-first: no trade unless every causal layer passes.", color = Fog)
        }
        item { SniperCard(snapshot.sniper, snapshot.lossAwareV2) }
        item { Text("Horizon evidence", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold) }
        items(snapshot.horizons, key = { it.horizon }) { plan ->
            HorizonCard(plan)
        }
    }
}

@Composable
private fun SniperCard(policy: SniperPolicy, v2: com.edgestack.mobile.data.LossAwareV2Summary) {
    val active = policy.status == "CONDITIONAL_PAPER_CANDIDATE"
    Card(
        colors = CardDefaults.cardColors(
            containerColor = (if (active) Mint else Coral).copy(alpha = 0.10f),
        ),
    ) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                policy.status.replace('_', ' '),
                color = if (active) Mint else Coral,
                fontSize = 34.sp,
                fontWeight = FontWeight.Black,
            )
            Text("WATCHLIST ONLY · ${policy.candidateSymbols.joinToString(" · ")}", color = Gold, fontWeight = FontWeight.Bold)
            Text(policy.releaseCondition, color = Fog)
            SectionCard("V2 selection · loss before return") {
                KeyValue("Horizon", v2.selectedHorizon.replace('_', ' '))
                KeyValue("Gross leverage", "${v2.selectedLeverage}× paper only")
                KeyValue("Evidence", v2.evidenceStatus.replace('_', ' '))
                if (v2.lossMetrics.status == "AVAILABLE") {
                    KeyValue("Loss probability", percent(v2.lossMetrics.lossProbability ?: 0.0))
                    KeyValue("Expected shortfall 95%", percent(v2.lossMetrics.expectedShortfall95 ?: 0.0))
                    KeyValue("Maximum adverse excursion", percent(v2.lossMetrics.maximumAdverseExcursion ?: 0.0))
                    KeyValue("Worst-decile threshold", percent(v2.lossMetrics.tenthPercentileReturn ?: 0.0))
                    KeyValue("90% loss streak", "${v2.lossMetrics.losingStreakP90 ?: 0.0} cohorts")
                } else {
                    Text("Loss metrics unavailable · NO TRADE", color = Coral, fontWeight = FontWeight.Bold)
                }
            }
            SectionCard("Event and data gates") {
                v2.dataGates.forEach { gate ->
                    KeyValue(gate.name.replace('_', ' '), gate.status.replace('_', ' '))
                    Text(gate.reason, color = Fog, style = MaterialTheme.typography.bodySmall)
                }
                Text(
                    if (v2.enabledEventVetoes.isEmpty()) "No veto enabled without evidence"
                    else "Enabled: ${v2.enabledEventVetoes.joinToString()}",
                    color = Gold,
                )
            }
            SectionCard("Tailwind alignment") {
                policy.alignments.forEach { layer ->
                    val passed = layer.status == "PASS"
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(layer.horizon, fontWeight = FontWeight.Black)
                        Text(layer.status, color = if (passed) Mint else Coral, fontWeight = FontWeight.Bold)
                    }
                    Text(layer.evidence, color = Fog, style = MaterialTheme.typography.bodySmall)
                }
            }
            SectionCard("Loss budget · $100,000 paper account") {
                KeyValue("Maximum / name", percent(policy.maxNameWeight))
                KeyValue("Maximum gross", percent(policy.maxGrossExposure))
                KeyValue("Planned loss / name", money(policy.maxPlannedLossPerNameUsd))
                KeyValue("Planned basket loss", money(policy.maxPlannedBasketLossUsd))
                Text(policy.validationStatus.replace('_', ' '), color = Gold, style = MaterialTheme.typography.labelSmall)
            }
            SectionCard("Hard vetoes") {
                policy.hardVetoes.forEach { Text("• ${it.replace('_', ' ')}", color = Coral) }
            }
            Text(policy.executionWindow, color = Gold, fontWeight = FontWeight.Bold)
            Text(v2.timing, color = Gold, fontWeight = FontWeight.Bold)
            Notice(policy.stopWarning, Coral)
        }
    }
}

@Composable
private fun HorizonCard(plan: HorizonPlan) {
    val available = plan.status == "CONDITIONAL_PAPER_SIGNAL"
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column(Modifier.weight(1f)) {
                    Text(plan.horizon, color = Gold, fontWeight = FontWeight.Black, letterSpacing = 1.sp)
                    Text(plan.title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                }
                Text(
                    if (available) "CONDITIONAL" else "NO MODEL",
                    modifier = Modifier
                        .background(
                            (if (available) Mint else Coral).copy(alpha = 0.14f),
                            RoundedCornerShape(100.dp),
                        )
                        .padding(horizontal = 9.dp, vertical = 5.dp),
                    color = if (available) Mint else Coral,
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.labelSmall,
                )
            }
            if (plan.symbols.isNotEmpty()) {
                Notice("COMPLETE BASKET · ${plan.symbols.joinToString(" · ")}", Mint)
            } else {
                Notice("NO STOCK RECOMMENDATION", Coral)
            }
            KeyValue("Holding period", plan.holdingPeriod)
            HorizontalDivider(color = PanelSoft)
            Text("WHEN TO ENTER", color = Gold, fontWeight = FontWeight.Bold)
            Text(plan.entryRule, color = Fog)
            Text("WHEN TO REVIEW", color = Gold, fontWeight = FontWeight.Bold)
            Text(plan.reviewRule, color = Fog)
            Text("WHEN TO EXIT", color = Gold, fontWeight = FontWeight.Bold)
            Text(plan.exitRule, color = Fog)
            SectionCard("Evidence boundary") { Text(plan.evidence, color = Fog) }
            SectionCard("Reverse or cancel if") {
                plan.invalidation.forEach { Text("• $it", color = Fog) }
            }
            Text("Unlock: ${plan.unlockRequirement}", color = if (available) Mint else Coral)
        }
    }
}

@Composable
private fun BaseList(
    snapshot: MobileSnapshot,
    warning: String? = null,
    content: androidx.compose.foundation.lazy.LazyListScope.() -> Unit,
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item { Watermark(snapshot.watermark) }
        if (snapshot.meta.stale) item { Notice("STALE SNAPSHOT · verify before acting", Coral) }
        warning?.let { item { Notice(it, Gold) } }
        content()
        item { Disclaimer(snapshot.disclaimer) }
    }
}

@Composable
private fun PlanScreen(snapshot: MobileSnapshot, state: MainUiState) {
    BaseList(snapshot, state.warning) {
        item {
            Text("Next paper action", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        }
        item {
            HeroCard(
                entry = snapshot.instruction.entrySession,
                submit = snapshot.instruction.submitByEt,
                exit = snapshot.instruction.exitSession,
                count = snapshot.recommendations.size,
            )
        }
        item {
            SectionCard("Execution contract") {
                KeyValue("Entry", "${snapshot.instruction.entryOrder} · ${snapshot.instruction.entrySession}")
                KeyValue("Submit by", snapshot.instruction.submitByEt)
                KeyValue("Time exit", "${snapshot.instruction.exitOrder} · ${snapshot.instruction.exitSession}")
                HorizontalDivider(color = PanelSoft)
                Text(snapshot.instruction.noChase, color = Gold)
            }
        }
        item {
            SectionCard("Cancel the basket if") {
                snapshot.instruction.cancelIf.forEach { Text("• $it", color = Fog) }
            }
        }
        item {
            SectionCard("Portfolio guardrails") {
                KeyValue("Paper capital", money(snapshot.portfolio.paperCapitalUsd))
                KeyValue("Target gross", percent(snapshot.portfolio.targetGross))
                KeyValue("Maximum per name", percent(snapshot.portfolio.maximumNameWeight))
                KeyValue("Risk budget / name", money(snapshot.portfolio.riskBudgetPerNameUsd))
                KeyValue("Shorts", if (snapshot.portfolio.shortsEnabled) "Enabled" else "Disabled")
            }
        }
        item {
            SectionCard("Basket rule") { Text(snapshot.basketRule, color = Fog) }
        }
    }
}

@Composable
private fun BasketScreen(snapshot: MobileSnapshot) {
    BaseList(snapshot) {
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column {
                    Text("Five-name basket", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
                    Text("One tested portfolio · no substitutions", color = Fog)
                }
                StatusPill(snapshot.modelStatus)
            }
        }
        items(snapshot.recommendations, key = { it.recommendationId }) { item ->
            RecommendationCard(item)
        }
        if (!snapshot.portfolio.shortsEnabled) {
            item { Notice("SHORT LIST DISABLED · declared short rules failed validation", Gold) }
        }
    }
}

@Composable
private fun RecommendationCard(item: Recommendation) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier.size(34.dp).background(Mint, RoundedCornerShape(10.dp)),
                    contentAlignment = Alignment.Center,
                ) {
                    Text("${item.rank}", color = Ink, fontWeight = FontWeight.Black)
                }
                Column(Modifier.padding(start = 12.dp).weight(1f)) {
                    Text(item.symbol, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Black)
                    Text("${item.direction} · ${item.suggestedShares} shares", color = Fog)
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text("${item.confidenceOrdinal}", color = Mint, fontWeight = FontWeight.Black, fontSize = 25.sp)
                    Text("ordinal", style = MaterialTheme.typography.labelSmall, color = Fog)
                }
            }
            HorizontalDivider(color = PanelSoft)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                MiniMetric("Signal close", money(item.signalCloseUsd))
                MiniMetric("5-session move", percent(item.trailingReturn), Coral)
                MiniMetric("2×ATR ref.", item.referenceStopUsd?.let(::money) ?: "—")
            }
            Notice(item.eventRisk, Coral)
        }
    }
}

@Composable
private fun EvidenceScreen(snapshot: MobileSnapshot) {
    BaseList(snapshot) {
        item {
            Text("Sealed evidence", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        }
        item {
            SectionCard("Final holdout · ${snapshot.holdout.status}") {
                KeyValue("Window", "${snapshot.holdout.start} → ${snapshot.holdout.end}")
                KeyValue("Coverage", "${snapshot.holdout.observations}/${snapshot.holdout.expectedSessions} sessions")
                KeyValue("Net mean / day", snapshot.holdout.netMean?.let(::basisPoints) ?: "—")
                KeyValue("Excess mean / day", snapshot.holdout.benchmarkExcessMean?.let(::basisPoints) ?: "—")
                KeyValue("Terminal wealth", snapshot.holdout.terminalWealth?.let { "${it.format(3)}×" } ?: "—")
                KeyValue("Benchmark wealth", snapshot.holdout.benchmarkWealth?.let { "${it.format(3)}×" } ?: "—")
            }
        }
        item {
            SectionCard("Immutable identity") {
                HashRow("Freeze", snapshot.holdout.freezeId)
                HashRow("Result", snapshot.holdout.resultSha256)
                KeyValue("Campaign", snapshot.campaignId)
                KeyValue("As of", snapshot.meta.marketAsOf)
                KeyValue("Source", snapshot.meta.source)
            }
        }
        item { Text("Audit trail", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold) }
        items(snapshot.audit) { event ->
            SectionCard(event.eventType.replace('_', ' ')) {
                Text(event.occurredAt, style = MaterialTheme.typography.labelMedium, color = Mint)
                Text(event.message, color = Fog)
            }
        }
    }
}

@Composable
private fun SetupScreen(state: MainUiState, viewModel: MainViewModel) {
    val settings = state.settings ?: return
    var endpoint by remember(settings.apiUrl) { mutableStateOf(settings.apiUrl) }
    var demo by remember(settings.demoMode) { mutableStateOf(settings.demoMode) }
    var token by remember(state.token) { mutableStateOf(state.token) }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        item {
            Text("Connection", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
            Text("Read-only evidence API. This app has no broker or order endpoint.", color = Fog)
        }
        item {
            SectionCard("Data mode") {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("Offline demonstration", fontWeight = FontWeight.Bold)
                        Text("Clearly labeled static sample data", color = Fog)
                    }
                    Switch(checked = demo, onCheckedChange = { demo = it })
                }
            }
        }
        item {
            OutlinedTextField(
                value = endpoint,
                onValueChange = { endpoint = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("API base URL") },
                supportingText = { Text("HTTPS, or local emulator host 10.0.2.2") },
                enabled = !demo,
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
            )
        }
        item {
            OutlinedTextField(
                value = token,
                onValueChange = { token = it; viewModel.setToken(it) },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Bearer token") },
                supportingText = { Text("Held in memory only; never persisted") },
                enabled = !demo,
                singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
            )
        }
        item {
            Button(
                onClick = { viewModel.saveSettings(endpoint, demo, token) },
                modifier = Modifier.fillMaxWidth(),
                enabled = demo || token.length >= 24,
            ) { Text("Save and refresh") }
        }
        state.fatalError?.let { item { Notice(it, Coral) } }
        item {
            SectionCard("Run the server") {
                Text("edgestack mobile-api --host 0.0.0.0 --campaign <id>", fontWeight = FontWeight.Bold)
                Text("Set EDGESTACK_MOBILE_TOKEN in the server environment. Use TLS outside local development.", color = Fog)
            }
        }
        item {
            Notice("PAPER ONLY · no broker integration · no real-order execution", Gold)
        }
    }
}

@Composable
private fun HeroCard(entry: String, submit: String, exit: String, count: Int) {
    Card(colors = CardDefaults.cardColors(containerColor = Mint)) {
        Column(Modifier.fillMaxWidth().padding(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("$count LONG", color = Ink, fontSize = 42.sp, fontWeight = FontWeight.Black)
            Text("closing-auction basket", color = Ink, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                MiniMetric("ENTRY", entry, Ink)
                MiniMetric("SUBMIT", submit.substringBefore(' '), Ink)
                MiniMetric("EXIT", exit, Ink)
            }
        }
    }
}

@Composable
private fun Watermark(text: String) {
    Box(
        Modifier.fillMaxWidth().background(Gold.copy(alpha = 0.14f), RoundedCornerShape(10.dp)).padding(10.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, color = Gold, fontWeight = FontWeight.Black, letterSpacing = 1.sp)
    }
}

@Composable
private fun Notice(text: String, color: Color) {
    Text(
        text,
        modifier = Modifier.fillMaxWidth().background(color.copy(alpha = 0.12f), RoundedCornerShape(9.dp)).padding(10.dp),
        color = color,
        style = MaterialTheme.typography.labelLarge,
    )
}

@Composable
private fun SectionCard(title: String, content: @Composable () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            content()
        }
    }
}

@Composable
private fun KeyValue(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = Fog, modifier = Modifier.weight(0.45f))
        Text(value, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(0.55f), maxLines = 2, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
private fun HashRow(label: String, value: String) {
    Column {
        Text(label, color = Fog, style = MaterialTheme.typography.labelMedium)
        Text(value, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
private fun MiniMetric(label: String, value: String, color: Color = Color.Unspecified) {
    Column {
        Text(label, style = MaterialTheme.typography.labelSmall, color = if (color == Color.Unspecified) Fog else color.copy(alpha = 0.75f))
        Text(value, fontWeight = FontWeight.Bold, color = color)
    }
}

@Composable
private fun StatusPill(status: String) {
    Text(
        status,
        modifier = Modifier.background(Mint.copy(alpha = 0.15f), RoundedCornerShape(100.dp)).padding(horizontal = 10.dp, vertical = 6.dp),
        color = Mint,
        fontWeight = FontWeight.Bold,
    )
}

@Composable
private fun Disclaimer(text: String) {
    Column(Modifier.padding(vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.Science, contentDescription = null, tint = Gold)
            Text(" RESEARCH & EDUCATION ONLY", color = Gold, fontWeight = FontWeight.Bold)
        }
        Text(text, color = Fog, style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun EmptyState(message: String, openSetup: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Icon(Icons.Outlined.CloudOff, contentDescription = null, modifier = Modifier.size(48.dp), tint = Coral)
        Spacer(Modifier.height(16.dp))
        Text(message, color = Fog)
        Spacer(Modifier.height(16.dp))
        Button(onClick = openSetup) { Text("Open setup") }
    }
}

private fun money(value: Double): String =
    NumberFormat.getCurrencyInstance(Locale.US).format(value)

private fun percent(value: Double): String = String.format(Locale.US, "%.2f%%", value * 100)

private fun basisPoints(value: Double): String = String.format(Locale.US, "%+.2f bp", value * 10_000)

private fun Double.format(decimals: Int): String = String.format(Locale.US, "%.${decimals}f", this)
