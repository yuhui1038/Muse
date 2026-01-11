"""
Configuration file example
Copy this file as config.py and fill in your actual configuration
"""

# ============== API Configuration ==============

# Suno API key (obtain from https://sunoapi.org)
SUNO_API_KEY = ""

# API base URL (usually no need to modify)
SUNO_API_BASE_URL = "https://api.sunoapi.org"

# ============== Generation Configuration ==============

# Default model version
DEFAULT_MODEL_VERSION = "V5"  # Options: V3_5, V4, V4_5, V4_5PLUS, V5

# Whether to enable custom mode
DEFAULT_CUSTOM_MODE = True

# Whether to generate instrumental by default
DEFAULT_INSTRUMENTAL = False

# ============== Task Configuration ==============

# Maximum wait time (seconds)
MAX_WAIT_TIME = 300

# Check interval (seconds)
CHECK_INTERVAL = 10

# Retry count
MAX_RETRIES = 3

# ============== File Configuration ==============

# Music file save directory
OUTPUT_DIRECTORY = "./generated_music"

# Audio format
AUDIO_FORMAT = "mp3"  # Options: mp3, wav

# ============== Batch Generation Configuration ==============

# Concurrency for batch generation
BATCH_CONCURRENCY = 5

# Batch generation delay (seconds, to avoid rate limiting)
BATCH_DELAY = 2

# ============== Logging Configuration ==============

# Log level
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# Log file path
LOG_FILE = "./suno_api.log"

# Whether to output to console
LOG_TO_CONSOLE = True

# ============== Webhook Configuration ==============

# Webhook callback URL (optional)
WEBHOOK_URL = None  # Example: "https://your-domain.com/webhook"

# Webhook secret (for verifying callback requests)
WEBHOOK_SECRET = None

