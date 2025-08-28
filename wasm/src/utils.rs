//! Utility functions for WASM module

use wasm_bindgen::prelude::*;

/// Set up better panic messages in debug mode
pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Log a message to the browser console
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Macro for logging to console
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => (crate::utils::log(&format_args!($($t)*).to_string()))
}

/// Convert Rust Result to JS Result
pub fn result_to_js<T, E>(result: Result<T, E>) -> Result<JsValue, JsValue>
where
    T: serde::Serialize,
    E: std::fmt::Display,
{
    match result {
        Ok(value) => serde_wasm_bindgen::to_value(&value)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e))),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}

/// Performance timing utilities
pub struct Timer {
    start: f64,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: js_sys::Date::now(),
        }
    }
    
    pub fn elapsed(&self) -> f64 {
        js_sys::Date::now() - self.start
    }
}

/// Memory usage utilities
#[wasm_bindgen]
pub fn get_memory_usage() -> JsValue {
    // Get memory size using wasm-bindgen's memory API
    #[cfg(target_arch = "wasm32")]
    {
        let memory = wasm_bindgen::memory();
         // Cast JsValue to WebAssembly.Memory and get buffer
         let memory_obj: js_sys::WebAssembly::Memory = memory.unchecked_into();
         let buffer = memory_obj.buffer();
         let array_buffer: js_sys::ArrayBuffer = buffer.unchecked_into();
         let bytes_used = array_buffer.byte_length() as f64;
        
        js_sys::Object::from(js_sys::Array::of2(
            &JsValue::from_str("bytes_used"),
            &JsValue::from_f64(bytes_used),
        )).into()
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Fallback for non-WASM targets
        js_sys::Object::from(js_sys::Array::of2(
            &JsValue::from_str("bytes_used"),
            &JsValue::from_f64(0.0),
        )).into()
    }
}