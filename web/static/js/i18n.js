/**
 * Smart Eye Diagnosis System - Internationalization (i18n) Engine
 * 
 * Features:
 * - Load user language preference from localStorage (default: 'ko')
 * - Change language dynamically with DOM update
 * - Apply translations to DOM elements with data-i18n attributes
 * - Expose current language to backend API for LLM requests
 */

class I18nEngine {
  constructor() {
    this.currentLanguage = this.loadLanguage();
    this.supportedLanguages = ['ko', 'en', 'zh', 'vi', 'ru', 'ja'];
    this.listeners = [];
    this.init();
  }

  /**
   * Initialize i18n engine on DOMContentLoaded
   */
  init() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        this.applyTranslations();
      });
    } else {
      this.applyTranslations();
    }
  }

  /**
   * Load language preference from localStorage
   * @returns {string} Language code (ko, en, zh, vi, ru, ja)
   */
  loadLanguage() {
    try {
      const saved = localStorage.getItem('i18n_language');
      if (saved && this.supportedLanguages.includes(saved)) {
        return saved;
      }
    } catch (e) {
      // localStorage not available
    }

    // Fallback to browser language
    const browserLang = navigator.language?.split('-')[0]?.toLowerCase();
    if (browserLang && this.supportedLanguages.includes(browserLang)) {
      return browserLang;
    }

    return 'ko'; // Default to Korean
  }

  /**
   * Get the current active language
   * @returns {string} Current language code
   */
  getCurrentLanguage() {
    return this.currentLanguage;
  }

  /**
   * Get a translation string by key
   * @param {string} key - Translation key
   * @param {object} params - Optional parameters for template replacement
   * @returns {string} Translated string
   */
  getTranslation(key, params = {}) {
    const langDict = translations[this.currentLanguage];
    let text = langDict && langDict[key] ? langDict[key] : key;

    // Replace template parameters (e.g., {count}, {phone})
    Object.keys(params).forEach(paramKey => {
      text = text.replace(new RegExp(`\\{${paramKey}\\}`, 'g'), params[paramKey]);
    });

    return text;
  }

  /**
   * Change the current language and update DOM
   * @param {string} lang - Language code (ko, en, zh, vi, ru, ja)
   */
  changeLanguage(lang) {
    if (!this.supportedLanguages.includes(lang)) {
      console.warn(`Unsupported language: ${lang}`);
      return;
    }

    this.currentLanguage = lang;

    // Save to localStorage
    try {
      localStorage.setItem('i18n_language', lang);
    } catch (e) {
      // localStorage not available
    }

    // Update DOM
    this.applyTranslations();

    // Notify listeners
    this.listeners.forEach(callback => callback(lang));
  }

  /**
   * Register a callback to be called when language changes
   * @param {function} callback - Function to call on language change
   */
  onLanguageChange(callback) {
    if (typeof callback === 'function') {
      this.listeners.push(callback);
    }
  }

  /**
   * Apply translations to all elements with data-i18n attribute
   */
  applyTranslations() {
    // Update document language attribute
    document.documentElement.lang = this.currentLanguage;

    // Find all elements with data-i18n attribute
    const elements = document.querySelectorAll('[data-i18n]');

    elements.forEach(element => {
      const key = element.getAttribute('data-i18n');
      const params = this.extractParamsFromElement(element);
      const text = this.getTranslation(key, params);

      // Handle different element types
      if (element.tagName === 'INPUT' && element.type === 'text') {
        // For input elements, update placeholder
        element.placeholder = text;
      } else if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
        // For textarea or other input types
        if (element.hasAttribute('data-i18n-placeholder')) {
          element.placeholder = text;
        } else {
          element.value = text;
        }
      } else {
        // For other elements, update textContent or innerHTML based on data-i18n-html
        if (element.hasAttribute('data-i18n-html')) {
          element.innerHTML = text;
        } else {
          element.textContent = text;
        }
      }
    });
  }

  /**
   * Extract parameters from element attributes for translation replacement
   * @param {HTMLElement} element - Element to extract params from
   * @returns {object} Parameters object
   */
  extractParamsFromElement(element) {
    const params = {};
    const attributes = element.attributes;

    for (let i = 0; i < attributes.length; i++) {
      const attr = attributes[i];
      if (attr.name.startsWith('data-param-')) {
        const paramKey = attr.name.substring(11); // Remove 'data-param-' prefix
        params[paramKey] = attr.value;
      }
    }

    return params;
  }

  /**
   * Utility: Get all supported languages
   * @returns {array} Array of language codes
   */
  getSupportedLanguages() {
    return [...this.supportedLanguages];
  }

  /**
   * Utility: Get language name in current language
   * @param {string} lang - Language code
   * @returns {string} Language name
   */
  getLanguageName(lang) {
    const key = `lang_${lang}`;
    return this.getTranslation(key);
  }
}

/**
 * Initialize and expose i18n globally
 */
const i18n = new I18nEngine();

/**
 * ============================================================================
 * LLM INTEGRATION - ACTIVE FUNCTIONS
 * ============================================================================
 */

/**
 * ============================================================================
 * Step 1: API Helper Functions for LLM Integration
 * ============================================================================
 */

/**
 * Enhanced fetch that automatically adds language to request body
 * @param {string} url - API endpoint
 * @param {object} options - Fetch options (method, body, etc.)
 * @returns {Promise<Response>} Fetch response
 */
i18n.apiFetch = function(url, options = {}) {
  const defaultOptions = {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      'Accept-Language': this.currentLanguage
    }
  };

  const mergedOptions = { ...defaultOptions, ...options };

  // If body exists, inject language field
  if (mergedOptions.body && typeof mergedOptions.body === 'string') {
    try {
      const bodyObj = JSON.parse(mergedOptions.body);
      bodyObj.language = this.currentLanguage;  // Step 2: Add language to request
      mergedOptions.body = JSON.stringify(bodyObj);
    } catch (e) {
      // If body is not JSON, leave as is
    }
  }

  return fetch(url, mergedOptions);
};

/**
 * Diagnose eye image with language context
 * @param {Blob|FormData} imageData - Image file or FormData
 * @param {object} metadata - Additional metadata (optional)
 * @returns {Promise<object>} Diagnosis result with language
 */
i18n.diagnosisWithLanguage = async function(imageData, metadata = {}) {
  try {
    const formData = imageData instanceof FormData ? imageData : new FormData();
    if (!(imageData instanceof FormData)) {
      formData.append('image', imageData);
    }
    
    // Step 2: Add language to request
    formData.append('language', this.currentLanguage);
    formData.append('metadata', JSON.stringify(metadata));

    const response = await fetch('/api/diagnose', {
      method: 'POST',
      body: formData,
      headers: {
        'Accept-Language': this.currentLanguage
      }
    });

    const result = await response.json();
    // Step 3: Backend should return language-aware response
    return {
      ...result,
      language: this.currentLanguage,
      timestamp: new Date().toISOString()
    };
  } catch (error) {
    console.error('Diagnosis API error:', error);
    throw error;
  }
};

/**
 * Generate medical report with LLM (language-aware)
 * @param {object} diagnosisResult - Diagnosis data
 * @param {object} patientInfo - Patient metadata
 * @returns {Promise<object>} Generated report
 */
i18n.generateMedicalReport = async function(diagnosisResult, patientInfo = {}) {
  try {
    const payload = {
      diagnosis_result: diagnosisResult,
      patient_info: patientInfo,
      language: this.currentLanguage  // Step 2: Include language
    };

    // Step 3: Backend receives language field
    const response = await this.apiFetch('/api/generate_report', {
      method: 'POST',
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Report generation failed: ${response.statusText}`);
    }

    const report = await response.json();
    
    // Step 4: LLM generates report in selected language
    return {
      ...report,
      language: this.currentLanguage,
      generated_at: new Date().toISOString()
    };
  } catch (error) {
    console.error('Report generation error:', error);
    throw error;
  }
};

/**
 * Generic API call with automatic language injection
 * @param {string} endpoint - API endpoint path
 * @param {object} data - Request data
 * @param {string} method - HTTP method (default: POST)
 * @returns {Promise<object>} API response
 */
i18n.callAPI = async function(endpoint, data = {}, method = 'POST') {
  try {
    const payload = {
      ...data,
      language: this.currentLanguage  // Step 2: Always include language
    };

    const response = await this.apiFetch(endpoint, {
      method: method,
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`API call failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API call error (${endpoint}):`, error);
    throw error;
  }
};

/**
 * ============================================================================
 * BACKEND INTEGRATION CHECKLIST
 * ============================================================================
 * 
 * Step 1: ✓ i18n.getCurrentLanguage()
 *   - Returns current language code: 'ko' | 'en' | 'zh' | 'vi' | 'ru' | 'ja'
 * 
 * Step 2: ✓ API Request with Language
 *   - Use i18n.diagnosisWithLanguage(image) or i18n.callAPI(endpoint, data)
 *   - Automatic language injection into request body
 * 
 * Step 3: Backend Receives Language
 *   - In Flask route: language = request.json.get('language', 'ko')
 *   - Store in database or pass to LLM
 * 
 * Step 4: LLM Generates Report
 *   - Python backend example:
 *     @app.route('/api/generate_report', methods=['POST'])
 *     def generate_report():
 *         language = request.json.get('language', 'ko')
 *         diagnosis = request.json.get('diagnosis_result')
 *         
 *         # LLM prompt template
 *         prompt = f"Generate a medical diagnosis report in {language} language..."
 *         report = llm_client.generate(prompt, diagnosis)
 *         
 *         return {
 *           'report': report,
 *           'language': language,
 *           'model_version': '1.0'
 *         }
 * 
 * ============================================================================
 */

/**
 * Export i18n for use in other scripts (ES6 modules)
 */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = i18n;
}
