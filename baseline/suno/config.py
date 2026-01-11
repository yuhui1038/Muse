"""
配置文件示例
复制此文件为 config.py 并填入你的实际配置
"""

# ============== API 配置 ==============

# Suno API 密钥 (从 https://sunoapi.org 获取)
SUNO_API_KEY = ""

# API 基础 URL (通常不需要修改)
SUNO_API_BASE_URL = "https://api.sunoapi.org"

# ============== 生成配置 ==============

# 默认模型版本
DEFAULT_MODEL_VERSION = "V5"  # 可选: V3_5, V4, V4_5, V4_5PLUS, V5

# 是否启用自定义模式
DEFAULT_CUSTOM_MODE = True

# 是否默认生成纯音乐
DEFAULT_INSTRUMENTAL = False

# ============== 任务配置 ==============

# 最大等待时间 (秒)
MAX_WAIT_TIME = 300

# 检查间隔 (秒)
CHECK_INTERVAL = 10

# 重试次数
MAX_RETRIES = 3

# ============== 文件配置 ==============

# 音乐文件保存目录
OUTPUT_DIRECTORY = "./generated_music"

# 音频格式
AUDIO_FORMAT = "mp3"  # 可选: mp3, wav

# ============== 批量生成配置 ==============

# 批量生成时的并发数
BATCH_CONCURRENCY = 5

# 批量生成的延迟 (秒，避免限流)
BATCH_DELAY = 2

# ============== 日志配置 ==============

# 日志级别
LOG_LEVEL = "INFO"  # 可选: DEBUG, INFO, WARNING, ERROR

# 日志文件路径
LOG_FILE = "./suno_api.log"

# 是否输出到控制台
LOG_TO_CONSOLE = True

# ============== Webhook 配置 ==============

# Webhook 回调 URL (可选)
WEBHOOK_URL = None  # 例如: "https://your-domain.com/webhook"

# Webhook 密钥 (用于验证回调请求)
WEBHOOK_SECRET = None

