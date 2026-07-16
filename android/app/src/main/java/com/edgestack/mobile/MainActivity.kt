package com.edgestack.mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.viewmodel.compose.viewModel
import com.edgestack.mobile.data.EdgeStackRepository
import com.edgestack.mobile.data.SettingsStore
import com.edgestack.mobile.ui.EdgeStackApp
import com.edgestack.mobile.ui.MainViewModel
import com.edgestack.mobile.ui.theme.EdgeStackTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val repository = EdgeStackRepository(applicationContext)
        val settings = SettingsStore(applicationContext)
        setContent {
            EdgeStackTheme {
                val model: MainViewModel = viewModel(
                    factory = MainViewModel.Factory(repository, settings),
                )
                EdgeStackApp(model)
            }
        }
    }
}
