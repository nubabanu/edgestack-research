package com.edgestack.mobile.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val Ink = Color(0xFF071512)
val Panel = Color(0xFF10231E)
val PanelSoft = Color(0xFF173029)
val Mint = Color(0xFF51E2B3)
val Gold = Color(0xFFF5C96A)
val Coral = Color(0xFFFF8D7A)
val Fog = Color(0xFFC5D8D1)

private val EdgeColors = darkColorScheme(
    primary = Mint,
    onPrimary = Ink,
    secondary = Gold,
    onSecondary = Ink,
    tertiary = Coral,
    background = Ink,
    onBackground = Color(0xFFF3F8F6),
    surface = Panel,
    onSurface = Color(0xFFF3F8F6),
    surfaceVariant = PanelSoft,
    onSurfaceVariant = Fog,
    error = Coral,
)

@Composable
fun EdgeStackTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = EdgeColors, content = content)
}
