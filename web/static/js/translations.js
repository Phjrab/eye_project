/**
 * Smart Eye Diagnosis System - Translation Dictionary
 * Supports: Korean (ko), English (en), Chinese (zh), Vietnamese (vi), Russian (ru), Japanese (ja)
 */

const translations = {
  ko: {
    // Main title and subtitle
    main_title: '안구 건강 진단 서비스',
    subtitle_line1: '<strong>NVIDIA Jetson AI</strong> 기술을 활용하여',
    subtitle_line2: '빠르고 정확하게 눈 상태를 점검하세요.',
    
    // Feature tags
    tag_conjunctivitis: '#결막염',
    tag_cataract: '#백내장',
    tag_pterygium: '#익상편',
    
    // Main buttons
    btn_start_diagnosis: '진단 시작하기',
    btn_kakao_link: '💬 카카오 연동',
    btn_mobile_access: '📱 모바일 기기로 접속',
    
    // Helper text
    helper_disclaimer: '※ 본 서비스는 의료 보조 및 스크리닝 목적으로 제공됩니다.',
    
    // Theme toggle
    theme_dark_mode: '다크 모드',
    theme_light_mode: '라이트 모드',
    theme_icon_dark: '🌙',
    theme_icon_light: '☀️',
    
    // Language selector
    lang_selector_title: '언어 선택',
    lang_ko: '한국어',
    lang_en: 'English',
    lang_zh: '中文',
    lang_vi: 'Tiếng Việt',
    
    // Mobile O2O modal
    mobile_o2o_title: '모바일 연동 (O2O)',
    mobile_o2o_close: '닫기',
    mobile_o2o_step1: '1단계: 스마트폰으로 아래 QR 코드를 스캔해 접속하세요.',
    mobile_o2o_step2: '2단계: 스마트폰에 아래 PIN을 입력하세요',
    mobile_o2o_waiting: '모바일 접속 대기 중...',
    mobile_o2o_connected: '모바일 접속 확인됨. PIN을 입력해 주세요.',
    mobile_o2o_verified: '모바일 인증이 완료되었습니다.',
    mobile_o2o_redirect: '기본 화면으로 {count}초 후 돌아갑니다.',
    mobile_qr_preparing: 'QR 준비 중...',
    mobile_qr_placeholder: '초기화 중...',
    
    // Kakao link modal
    kakao_link_title: '카카오 계정 연동',
    kakao_link_close: '닫기',
    kakao_link_description: '전화번호를 입력한 뒤 카카오 로그인 창에서 동의하면 이 기기 사용자와 계정이 연결됩니다.',
    kakao_link_placeholder: '010-1234-5678',
    kakao_link_cancel: '취소',
    kakao_link_submit: '로그인 팝업 열기',
    kakao_link_error_format: '전화번호 11자리를 입력해 주세요. 예: 010-1234-5678',
    kakao_link_loading: '카카오 로그인 창을 여는 중입니다...',
    kakao_link_success: '연동 완료: {phone}',
    kakao_link_popup_blocked: '팝업이 차단되었습니다. 브라우저 팝업 허용 후 다시 시도해 주세요.',
    kakao_link_status_wait: '카카오 로그인/동의 후 창이 자동으로 닫힙니다.',
    kakao_link_error: '연동에 실패했습니다.',
  },

  en: {
    // Main title and subtitle
    main_title: 'Eye Health Diagnosis Service',
    subtitle_line1: 'Powered by <strong>NVIDIA Jetson AI</strong>',
    subtitle_line2: 'Check your eye condition quickly and accurately.',
    
    // Feature tags
    tag_conjunctivitis: '#Conjunctivitis',
    tag_cataract: '#Cataract',
    tag_pterygium: '#Pterygium',
    
    // Main buttons
    btn_start_diagnosis: 'Start Diagnosis',
    btn_kakao_link: '💬 Link Kakao Account',
    btn_mobile_access: '📱 Access via Mobile Device',
    
    // Helper text
    helper_disclaimer: '※ This service is provided for medical support and screening purposes.',
    
    // Theme toggle
    theme_dark_mode: 'Dark Mode',
    theme_light_mode: 'Light Mode',
    theme_icon_dark: '🌙',
    theme_icon_light: '☀️',
    
    // Language selector
    lang_selector_title: 'Select Language',
    lang_ko: '한국어',
    lang_en: 'English',
    lang_zh: '中文',
    lang_vi: 'Tiếng Việt',
    
    // Mobile O2O modal
    mobile_o2o_title: 'Mobile Access (O2O)',
    mobile_o2o_close: 'Close',
    mobile_o2o_step1: 'Step 1: Scan the QR code below on your smartphone.',
    mobile_o2o_step2: 'Step 2: Enter the PIN displayed on your smartphone',
    mobile_o2o_waiting: 'Waiting for mobile access...',
    mobile_o2o_connected: 'Mobile device detected. Please enter the PIN.',
    mobile_o2o_verified: 'Mobile authentication completed.',
    mobile_o2o_redirect: 'Returning to home screen in {count} seconds.',
    mobile_qr_preparing: 'QR Preparing...',
    mobile_qr_placeholder: 'Initializing...',
    
    // Kakao link modal
    kakao_link_title: 'Link Kakao Account',
    kakao_link_close: 'Close',
    kakao_link_description: 'Enter your phone number and consent to Kakao login to link your account with this device.',
    kakao_link_placeholder: '010-1234-5678',
    kakao_link_cancel: 'Cancel',
    kakao_link_submit: 'Open Login Popup',
    kakao_link_error_format: 'Please enter 11 digits. Example: 010-1234-5678',
    kakao_link_loading: 'Opening Kakao login...',
    kakao_link_success: 'Linked: {phone}',
    kakao_link_popup_blocked: 'Popup blocked. Please allow popups in browser settings.',
    kakao_link_status_wait: 'Window will close automatically after consent.',
    kakao_link_error: 'Linking failed.',
  },

  zh: {
    // Main title and subtitle
    main_title: '眼健康诊断服务',
    subtitle_line1: '由 <strong>NVIDIA Jetson AI</strong> 提供支持',
    subtitle_line2: '快速准确地检查您的眼睛状况。',
    
    // Feature tags
    tag_conjunctivitis: '#结膜炎',
    tag_cataract: '#白内障',
    tag_pterygium: '#翼状胬肉',
    
    // Main buttons
    btn_start_diagnosis: '开始诊断',
    btn_kakao_link: '💬 关联Kakao账户',
    btn_mobile_access: '📱 通过移动设备访问',
    
    // Helper text
    helper_disclaimer: '※ 本服务仅供医疗辅助和筛查之用。',
    
    // Theme toggle
    theme_dark_mode: '暗黑模式',
    theme_light_mode: '浅色模式',
    theme_icon_dark: '🌙',
    theme_icon_light: '☀️',
    
    // Language selector
    lang_selector_title: '选择语言',
    lang_ko: '한국어',
    lang_en: 'English',
    lang_zh: '中文',
    lang_vi: 'Tiếng Việt',
    
    // Mobile O2O modal
    mobile_o2o_title: '移动访问 (O2O)',
    mobile_o2o_close: '关闭',
    mobile_o2o_step1: '第1步：在智能手机上扫描下面的二维码。',
    mobile_o2o_step2: '第2步：输入智能手机上显示的PIN码',
    mobile_o2o_waiting: '等待移动设备接入...',
    mobile_o2o_connected: '已检测到移动设备。请输入PIN码。',
    mobile_o2o_verified: '移动设备认证完成。',
    mobile_o2o_redirect: '{count}秒后返回首页。',
    mobile_qr_preparing: '正在准备二维码...',
    mobile_qr_placeholder: '初始化中...',
    
    // Kakao link modal
    kakao_link_title: '关联Kakao账户',
    kakao_link_close: '关闭',
    kakao_link_description: '输入您的电话号码并同意Kakao登录，将您的账户与此设备关联。',
    kakao_link_placeholder: '010-1234-5678',
    kakao_link_cancel: '取消',
    kakao_link_submit: '打开登录弹窗',
    kakao_link_error_format: '请输入11位数字。示例：010-1234-5678',
    kakao_link_loading: '正在打开Kakao登录...',
    kakao_link_success: '已关联：{phone}',
    kakao_link_popup_blocked: '弹窗被阻止。请在浏览器设置中允许弹窗。',
    kakao_link_status_wait: '同意后窗口将自动关闭。',
    kakao_link_error: '关联失败。',
  },

  vi: {
    // Main title and subtitle
    main_title: 'Dịch vụ Chẩn đoán Sức khỏe Mắt',
    subtitle_line1: 'Được hỗ trợ bởi <strong>NVIDIA Jetson AI</strong>',
    subtitle_line2: 'Kiểm tra tình trạng mắt của bạn nhanh chóng và chính xác.',
    
    // Feature tags
    tag_conjunctivitis: '#Viêm kết mạc',
    tag_cataract: '#Đục thủy tinh thể',
    tag_pterygium: '#Cánh mắt',
    
    // Main buttons
    btn_start_diagnosis: 'Bắt đầu Chẩn đoán',
    btn_kakao_link: '💬 Liên kết Tài khoản Kakao',
    btn_mobile_access: '📱 Truy cập qua Thiết bị Di động',
    
    // Helper text
    helper_disclaimer: '※ Dịch vụ này được cung cấp cho mục đích hỗ trợ y tế và sàng lọc.',
    
    // Theme toggle
    theme_dark_mode: 'Chế độ Tối',
    theme_light_mode: 'Chế độ Sáng',
    theme_icon_dark: '🌙',
    theme_icon_light: '☀️',
    
    // Language selector
    lang_selector_title: 'Chọn Ngôn ngữ',
    lang_ko: '한국어',
    lang_en: 'English',
    lang_zh: '中文',
    lang_vi: 'Tiếng Việt',
    
    // Mobile O2O modal
    mobile_o2o_title: 'Truy cập Di động (O2O)',
    mobile_o2o_close: 'Đóng',
    mobile_o2o_step1: 'Bước 1: Quét mã QR bên dưới trên điện thoại thông minh của bạn.',
    mobile_o2o_step2: 'Bước 2: Nhập mã PIN được hiển thị trên điện thoại thông minh của bạn',
    mobile_o2o_waiting: 'Đang chờ truy cập từ thiết bị di động...',
    mobile_o2o_connected: 'Đã phát hiện thiết bị di động. Vui lòng nhập mã PIN.',
    mobile_o2o_verified: 'Xác thực thiết bị di động hoàn thành.',
    mobile_o2o_redirect: 'Quay lại trang chủ trong {count} giây.',
    mobile_qr_preparing: 'Đang chuẩn bị mã QR...',
    mobile_qr_placeholder: 'Đang khởi tạo...',
    
    // Kakao link modal
    kakao_link_title: 'Liên kết Tài khoản Kakao',
    kakao_link_close: 'Đóng',
    kakao_link_description: 'Nhập số điện thoại của bạn và đồng ý đăng nhập Kakao để liên kết tài khoản của bạn với thiết bị này.',
    kakao_link_placeholder: '010-1234-5678',
    kakao_link_cancel: 'Hủy',
    kakao_link_submit: 'Mở Popup Đăng nhập',
    kakao_link_error_format: 'Vui lòng nhập 11 chữ số. Ví dụ: 010-1234-5678',
    kakao_link_loading: 'Đang mở đăng nhập Kakao...',
    kakao_link_success: 'Đã liên kết: {phone}',
    kakao_link_popup_blocked: 'Popup bị chặn. Vui lòng cho phép popup trong cài đặt trình duyệt.',
    kakao_link_status_wait: 'Cửa sổ sẽ tự động đóng sau khi đồng ý.',
    kakao_link_error: 'Liên kết không thành công.',
  },

  ru: {
    // Main title and subtitle
    main_title: 'Сервис диагностики здоровья глаз',
    subtitle_line1: 'Работает на базе <strong>NVIDIA Jetson AI</strong>',
    subtitle_line2: 'Быстро и точно проверьте состояние ваших глаз.',
    
    // Feature tags
    tag_conjunctivitis: '#Конъюнктивит',
    tag_cataract: '#Катаракта',
    tag_pterygium: '#Птеригиум',
    
    // Main buttons
    btn_start_diagnosis: 'Начать диагностику',
    btn_kakao_link: '💬 Связать аккаунт Kakao',
    btn_mobile_access: '📱 Доступ с мобильного устройства',
    
    // Helper text
    helper_disclaimer: '※ Этот сервис предоставляется в целях медицинской поддержки и скрининга.',
    
    // Theme toggle
    theme_dark_mode: 'Темный режим',
    theme_light_mode: 'Светлый режим',
    theme_icon_dark: '🌙',
    theme_icon_light: '☀️',
    
    // Language selector
    lang_selector_title: 'Выбрать язык',
    lang_ko: '한국어',
    lang_en: 'English',
    lang_zh: '中文',
    lang_vi: 'Tiếng Việt',
    lang_ru: 'Русский',
    lang_ja: '日本語',
    
    // Mobile O2O modal
    mobile_o2o_title: 'Мобильный доступ (O2O)',
    mobile_o2o_close: 'Закрыть',
    mobile_o2o_step1: 'Шаг 1: Отсканируйте QR-код ниже на своем смартфоне.',
    mobile_o2o_step2: 'Шаг 2: Введите PIN-код, отображаемый на вашем смартфоне',
    mobile_o2o_waiting: 'Ожидание доступа с мобильного устройства...',
    mobile_o2o_connected: 'Мобильное устройство обнаружено. Введите PIN-код.',
    mobile_o2o_verified: 'Аутентификация мобильного устройства завершена.',
    mobile_o2o_redirect: 'Возврат на главный экран через {count} секунд.',
    mobile_qr_preparing: 'Подготовка QR-кода...',
    mobile_qr_placeholder: 'Инициализация...',
    
    // Kakao link modal
    kakao_link_title: 'Связать аккаунт Kakao',
    kakao_link_close: 'Закрыть',
    kakao_link_description: 'Введите номер телефона и согласитесь с входом Kakao, чтобы связать ваш аккаунт с этим устройством.',
    kakao_link_placeholder: '010-1234-5678',
    kakao_link_cancel: 'Отмена',
    kakao_link_submit: 'Открыть всплывающее окно входа',
    kakao_link_error_format: 'Пожалуйста, введите 11 цифр. Пример: 010-1234-5678',
    kakao_link_loading: 'Открытие входа Kakao...',
    kakao_link_success: 'Связано: {phone}',
    kakao_link_popup_blocked: 'Всплывающее окно заблокировано. Разрешите всплывающие окна в настройках браузера.',
    kakao_link_status_wait: 'Окно закроется автоматически после согласия.',
    kakao_link_error: 'Ошибка связи.',
  },

  ja: {
    // Main title and subtitle
    main_title: '眼の健康診断サービス',
    subtitle_line1: '<strong>NVIDIA Jetson AI</strong>を利用',
    subtitle_line2: 'あなたの眼の状態を迅速かつ正確に検査してください。',
    
    // Feature tags
    tag_conjunctivitis: '#結膜炎',
    tag_cataract: '#白内障',
    tag_pterygium: '#翼状片',
    
    // Main buttons
    btn_start_diagnosis: '診断を開始',
    btn_kakao_link: '💬 Kakaoアカウントをリンク',
    btn_mobile_access: '📱 モバイルデバイスからアクセス',
    
    // Helper text
    helper_disclaimer: '※ このサービスは医療補助とスクリーニング目的で提供されます。',
    
    // Theme toggle
    theme_dark_mode: 'ダークモード',
    theme_light_mode: 'ライトモード',
    theme_icon_dark: '🌙',
    theme_icon_light: '☀️',
    
    // Language selector
    lang_selector_title: '言語を選択',
    lang_ko: '한국어',
    lang_en: 'English',
    lang_zh: '中文',
    lang_vi: 'Tiếng Việt',
    lang_ru: 'Русский',
    lang_ja: '日本語',
    
    // Mobile O2O modal
    mobile_o2o_title: 'モバイルアクセス (O2O)',
    mobile_o2o_close: '閉じる',
    mobile_o2o_step1: 'ステップ1: 下のQRコードをスマートフォンでスキャンしてください。',
    mobile_o2o_step2: 'ステップ2: スマートフォンに表示されているPINコードを入力してください',
    mobile_o2o_waiting: 'モバイルデバイスからのアクセスを待機中...',
    mobile_o2o_connected: 'モバイルデバイスが検出されました。PINコードを入力してください。',
    mobile_o2o_verified: 'モバイルデバイスの認証が完了しました。',
    mobile_o2o_redirect: '{count}秒後にホーム画面に戻ります。',
    mobile_qr_preparing: 'QRコードを準備中...',
    mobile_qr_placeholder: '初期化中...',
    
    // Kakao link modal
    kakao_link_title: 'Kakaoアカウントをリンク',
    kakao_link_close: '閉じる',
    kakao_link_description: '電話番号を入力し、Kakaoログインに同意してアカウントをこのデバイスにリンクしてください。',
    kakao_link_placeholder: '010-1234-5678',
    kakao_link_cancel: 'キャンセル',
    kakao_link_submit: 'ログインポップアップを開く',
    kakao_link_error_format: '11桁を入力してください。例: 010-1234-5678',
    kakao_link_loading: 'Kakaoログインを開いています...',
    kakao_link_success: 'リンク完了: {phone}',
    kakao_link_popup_blocked: 'ポップアップがブロックされました。ブラウザ設定でポップアップを許可してください。',
    kakao_link_status_wait: '同意後、ウィンドウは自動的に閉じます。',
    kakao_link_error: 'リンクに失敗しました。',
  }
};

