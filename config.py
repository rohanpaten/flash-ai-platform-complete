"""
Configuration module for FLASH API
Handles environment variables and security settings
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings from environment variables"""
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # API Settings
    API_TITLE: str = "FLASH 2.0 API"
    API_VERSION: str = "2.0.0"
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")  # Secure default: localhost only
    
    # Security Settings
    # Generate secure secret in production
    if ENVIRONMENT == "production" and not os.getenv("SECRET_KEY"):
        raise ValueError("SECRET_KEY must be set in production environment")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    ALLOWED_HOSTS: List[str] = os.getenv("ALLOWED_HOSTS", "localhost,*.flash-platform.com").split(",")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # seconds
    
    # Model Settings
    MODEL_BASE_PATH: str = os.getenv("MODEL_BASE_PATH", "models/v2_enhanced")
    PILLAR_MODEL_PATH: str = os.getenv("PILLAR_MODEL_PATH", "models/v2")
    MODEL_CACHE_SIZE: int = int(os.getenv("MODEL_CACHE_SIZE", "5"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Database (if needed in future)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    
    # Feature Flags
    ENABLE_EXPLANATION_API: bool = os.getenv("ENABLE_EXPLANATION_API", "true").lower() == "true"
    ENABLE_BATCH_PREDICTIONS: bool = os.getenv("ENABLE_BATCH_PREDICTIONS", "false").lower() == "true"
    DISABLE_AUTH: bool = os.getenv("DISABLE_AUTH", "false").lower() == "true"  # Development/testing only
    
    # Validation Limits
    MAX_REQUEST_SIZE: int = int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "100"))
    
    # API Keys (for production)
    API_KEYS: List[str] = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
    
    # Require API keys in production
    if ENVIRONMENT == "production" and not API_KEYS:
        raise ValueError("API_KEYS must be set in production environment")
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production"""
        return cls.ENVIRONMENT == "production"
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development"""
        return cls.ENVIRONMENT == "development"
    
    @classmethod
    def get_log_config(cls) -> dict:
        """Get logging configuration"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": cls.LOG_FORMAT,
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": cls.LOG_LEVEL,
                "handlers": ["default"],
            },
        }


# Create settings instance
settings = Settings()