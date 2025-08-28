//! Internationalization framework for Augustium
//! Provides localization support for compiler messages, documentation, and runtime errors

use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Supported locales
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Locale {
    English,
    Spanish,
    French,
    German,
    Chinese,
    Japanese,
    Korean,
    Russian,
    Portuguese,
    Italian,
    Dutch,
    Arabic,
    Hindi,
}

impl fmt::Display for Locale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code = match self {
            Locale::English => "en",
            Locale::Spanish => "es",
            Locale::French => "fr",
            Locale::German => "de",
            Locale::Chinese => "zh",
            Locale::Japanese => "ja",
            Locale::Korean => "ko",
            Locale::Russian => "ru",
            Locale::Portuguese => "pt",
            Locale::Italian => "it",
            Locale::Dutch => "nl",
            Locale::Arabic => "ar",
            Locale::Hindi => "hi",
        };
        write!(f, "{}", code)
    }
}

impl Locale {
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en" | "en-us" | "en-gb" => Some(Locale::English),
            "es" | "es-es" | "es-mx" => Some(Locale::Spanish),
            "fr" | "fr-fr" | "fr-ca" => Some(Locale::French),
            "de" | "de-de" | "de-at" => Some(Locale::German),
            "zh" | "zh-cn" | "zh-tw" => Some(Locale::Chinese),
            "ja" | "ja-jp" => Some(Locale::Japanese),
            "ko" | "ko-kr" => Some(Locale::Korean),
            "ru" | "ru-ru" => Some(Locale::Russian),
            "pt" | "pt-br" | "pt-pt" => Some(Locale::Portuguese),
            "it" | "it-it" => Some(Locale::Italian),
            "nl" | "nl-nl" => Some(Locale::Dutch),
            "ar" | "ar-sa" => Some(Locale::Arabic),
            "hi" | "hi-in" => Some(Locale::Hindi),
            _ => None,
        }
    }
}

/// Message key for localized strings
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MessageKey(pub &'static str);

/// Localized message with parameters
#[derive(Debug, Clone)]
pub struct LocalizedMessage {
    pub key: MessageKey,
    pub params: HashMap<String, String>,
}

impl LocalizedMessage {
    pub fn new(key: &'static str) -> Self {
        Self {
            key: MessageKey(key),
            params: HashMap::new(),
        }
    }

    pub fn with_param(mut self, name: &str, value: &str) -> Self {
        self.params.insert(name.to_string(), value.to_string());
        self
    }
}

/// Translation bundle for a specific locale
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationBundle {
    pub locale: Locale,
    pub messages: HashMap<String, String>,
}

impl TranslationBundle {
    pub fn new(locale: Locale) -> Self {
        Self {
            locale,
            messages: HashMap::new(),
        }
    }

    pub fn add_message(&mut self, key: &str, message: &str) {
        self.messages.insert(key.to_string(), message.to_string());
    }

    pub fn get_message(&self, key: &str) -> Option<&String> {
        self.messages.get(key)
    }
}

/// Main internationalization manager
pub struct I18nManager {
    current_locale: Locale,
    bundles: HashMap<Locale, TranslationBundle>,
    fallback_locale: Locale,
}

impl I18nManager {
    pub fn new() -> Self {
        let mut manager = Self {
            current_locale: Locale::English,
            bundles: HashMap::new(),
            fallback_locale: Locale::English,
        };
        
        manager.load_default_bundles();
        manager
    }

    pub fn set_locale(&mut self, locale: Locale) {
        self.current_locale = locale;
    }

    pub fn get_locale(&self) -> &Locale {
        &self.current_locale
    }

    pub fn add_bundle(&mut self, bundle: TranslationBundle) {
        self.bundles.insert(bundle.locale.clone(), bundle);
    }

    /// Get localized message with parameter substitution
    pub fn get_message(&self, msg: &LocalizedMessage) -> String {
        let key = msg.key.0;
        
        // Try current locale first
        let template = if let Some(bundle) = self.bundles.get(&self.current_locale) {
            bundle.get_message(key)
        } else {
            None
        };

        // Fall back to fallback locale
        let template = template.or_else(|| {
            if let Some(bundle) = self.bundles.get(&self.fallback_locale) {
                bundle.get_message(key)
            } else {
                None
            }
        });

        // If no translation found, return the key
        let template = template.map(|s| s.as_str()).unwrap_or(key);

        // Substitute parameters
        self.substitute_params(template, &msg.params)
    }

    fn substitute_params(&self, template: &str, params: &HashMap<String, String>) -> String {
        let mut result = template.to_string();
        
        for (key, value) in params {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }
        
        result
    }

    fn load_default_bundles(&mut self) {
        // English bundle (default)
        let mut en_bundle = TranslationBundle::new(Locale::English);
        en_bundle.add_message("compiler.error.syntax", "Syntax error: {message}");
        en_bundle.add_message("compiler.error.type_mismatch", "Type mismatch: expected {expected}, found {found}");
        en_bundle.add_message("compiler.error.undefined_variable", "Undefined variable: {name}");
        en_bundle.add_message("compiler.error.undefined_function", "Undefined function: {name}");
        en_bundle.add_message("compiler.error.gas_limit_exceeded", "Gas limit exceeded");
        en_bundle.add_message("compiler.warning.unused_variable", "Unused variable: {name}");
        en_bundle.add_message("compiler.warning.dead_code", "Dead code detected");
        en_bundle.add_message("vm.error.stack_overflow", "Stack overflow");
        en_bundle.add_message("vm.error.out_of_gas", "Out of gas");
        en_bundle.add_message("vm.error.revert", "Transaction reverted: {reason}");
        en_bundle.add_message("ml.error.invalid_model", "Invalid ML model: {details}");
        en_bundle.add_message("ml.error.training_failed", "Model training failed: {reason}");
        self.add_bundle(en_bundle);

        // Spanish bundle
        let mut es_bundle = TranslationBundle::new(Locale::Spanish);
        es_bundle.add_message("compiler.error.syntax", "Error de sintaxis: {message}");
        es_bundle.add_message("compiler.error.type_mismatch", "Tipos incompatibles: esperado {expected}, encontrado {found}");
        es_bundle.add_message("compiler.error.undefined_variable", "Variable indefinida: {name}");
        es_bundle.add_message("compiler.error.undefined_function", "Función indefinida: {name}");
        es_bundle.add_message("compiler.error.gas_limit_exceeded", "Límite de gas excedido");
        es_bundle.add_message("compiler.warning.unused_variable", "Variable no utilizada: {name}");
        es_bundle.add_message("compiler.warning.dead_code", "Código muerto detectado");
        es_bundle.add_message("vm.error.stack_overflow", "Desbordamiento de pila");
        es_bundle.add_message("vm.error.out_of_gas", "Sin gas");
        es_bundle.add_message("vm.error.revert", "Transacción revertida: {reason}");
        es_bundle.add_message("ml.error.invalid_model", "Modelo ML inválido: {details}");
        es_bundle.add_message("ml.error.training_failed", "Entrenamiento del modelo falló: {reason}");
        self.add_bundle(es_bundle);

        // French bundle
        let mut fr_bundle = TranslationBundle::new(Locale::French);
        fr_bundle.add_message("compiler.error.syntax", "Erreur de syntaxe: {message}");
        fr_bundle.add_message("compiler.error.type_mismatch", "Types incompatibles: attendu {expected}, trouvé {found}");
        fr_bundle.add_message("compiler.error.undefined_variable", "Variable indéfinie: {name}");
        fr_bundle.add_message("compiler.error.undefined_function", "Fonction indéfinie: {name}");
        fr_bundle.add_message("compiler.error.gas_limit_exceeded", "Limite de gaz dépassée");
        fr_bundle.add_message("compiler.warning.unused_variable", "Variable inutilisée: {name}");
        fr_bundle.add_message("compiler.warning.dead_code", "Code mort détecté");
        fr_bundle.add_message("vm.error.stack_overflow", "Débordement de pile");
        fr_bundle.add_message("vm.error.out_of_gas", "Plus de gaz");
        fr_bundle.add_message("vm.error.revert", "Transaction annulée: {reason}");
        fr_bundle.add_message("ml.error.invalid_model", "Modèle ML invalide: {details}");
        fr_bundle.add_message("ml.error.training_failed", "Échec de l'entraînement du modèle: {reason}");
        self.add_bundle(fr_bundle);

        // German bundle
        let mut de_bundle = TranslationBundle::new(Locale::German);
        de_bundle.add_message("compiler.error.syntax", "Syntaxfehler: {message}");
        de_bundle.add_message("compiler.error.type_mismatch", "Typen stimmen nicht überein: erwartet {expected}, gefunden {found}");
        de_bundle.add_message("compiler.error.undefined_variable", "Undefinierte Variable: {name}");
        de_bundle.add_message("compiler.error.undefined_function", "Undefinierte Funktion: {name}");
        de_bundle.add_message("compiler.error.gas_limit_exceeded", "Gas-Limit überschritten");
        de_bundle.add_message("compiler.warning.unused_variable", "Unbenutzte Variable: {name}");
        de_bundle.add_message("compiler.warning.dead_code", "Toter Code erkannt");
        de_bundle.add_message("vm.error.stack_overflow", "Stack-Überlauf");
        de_bundle.add_message("vm.error.out_of_gas", "Kein Gas mehr");
        de_bundle.add_message("vm.error.revert", "Transaktion rückgängig gemacht: {reason}");
        de_bundle.add_message("ml.error.invalid_model", "Ungültiges ML-Modell: {details}");
        de_bundle.add_message("ml.error.training_failed", "Modelltraining fehlgeschlagen: {reason}");
        self.add_bundle(de_bundle);

        // Chinese bundle
        let mut zh_bundle = TranslationBundle::new(Locale::Chinese);
        zh_bundle.add_message("compiler.error.syntax", "语法错误: {message}");
        zh_bundle.add_message("compiler.error.type_mismatch", "类型不匹配: 期望 {expected}, 发现 {found}");
        zh_bundle.add_message("compiler.error.undefined_variable", "未定义变量: {name}");
        zh_bundle.add_message("compiler.error.undefined_function", "未定义函数: {name}");
        zh_bundle.add_message("compiler.error.gas_limit_exceeded", "Gas限制超出");
        zh_bundle.add_message("compiler.warning.unused_variable", "未使用变量: {name}");
        zh_bundle.add_message("compiler.warning.dead_code", "检测到死代码");
        zh_bundle.add_message("vm.error.stack_overflow", "堆栈溢出");
        zh_bundle.add_message("vm.error.out_of_gas", "Gas不足");
        zh_bundle.add_message("vm.error.revert", "交易回滚: {reason}");
        zh_bundle.add_message("ml.error.invalid_model", "无效的ML模型: {details}");
        zh_bundle.add_message("ml.error.training_failed", "模型训练失败: {reason}");
        self.add_bundle(zh_bundle);
    }
}

/// Global i18n manager instance
static mut GLOBAL_I18N: Option<I18nManager> = None;
static mut I18N_INITIALIZED: bool = false;

/// Initialize the global i18n manager
pub fn init_i18n() {
    unsafe {
        if !I18N_INITIALIZED {
            GLOBAL_I18N = Some(I18nManager::new());
            I18N_INITIALIZED = true;
        }
    }
}

/// Get the global i18n manager
pub fn get_i18n() -> &'static mut I18nManager {
    unsafe {
        if !I18N_INITIALIZED {
            init_i18n();
        }
        GLOBAL_I18N.as_mut().unwrap()
    }
}

/// Convenient macro for creating localized messages
#[macro_export]
macro_rules! t {
    ($key:expr) => {
        $crate::i18n::LocalizedMessage::new($key)
    };
    ($key:expr, $($param:expr => $value:expr),+) => {
        {
            let mut msg = $crate::i18n::LocalizedMessage::new($key);
            $(
                msg = msg.with_param($param, $value);
            )+
            msg
        }
    };
}

/// Convenient function to get localized string
pub fn localize(msg: &LocalizedMessage) -> String {
    get_i18n().get_message(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locale_from_code() {
        assert_eq!(Locale::from_code("en"), Some(Locale::English));
        assert_eq!(Locale::from_code("es-mx"), Some(Locale::Spanish));
        assert_eq!(Locale::from_code("invalid"), None);
    }

    #[test]
    fn test_translation_bundle() {
        let mut bundle = TranslationBundle::new(Locale::English);
        bundle.add_message("test.key", "Test message");
        
        assert_eq!(bundle.get_message("test.key"), Some(&"Test message".to_string()));
        assert_eq!(bundle.get_message("nonexistent"), None);
    }

    #[test]
    fn test_i18n_manager() {
        let mut manager = I18nManager::new();
        manager.set_locale(Locale::Spanish);
        
        let msg = LocalizedMessage::new("compiler.error.syntax")
            .with_param("message", "unexpected token");
        
        let result = manager.get_message(&msg);
        assert!(result.contains("Error de sintaxis"));
        assert!(result.contains("unexpected token"));
    }

    #[test]
    fn test_parameter_substitution() {
        let manager = I18nManager::new();
        let params = vec![
            ("name".to_string(), "myVar".to_string()),
        ].into_iter().collect();
        
        let result = manager.substitute_params("Unused variable: {name}", &params);
        assert_eq!(result, "Unused variable: myVar");
    }
}
