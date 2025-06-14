#!/usr/bin/env python3
"""
FLASH API Server - Consolidated Final Version
Integrates all fixes and improvements
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Security, status, Request, Response, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Annotated

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import models and utilities
from models.unified_orchestrator_v3_integrated import UnifiedOrchestratorV3
from type_converter_simple import TypeConverter
from feature_config import ALL_FEATURES
from frontend_models import FrontendStartupData
from config import settings
from utils.sanitization import sanitize_startup_data, sanitize_string
from utils.error_handling import (
    error_handler, handle_prediction_errors, DataValidationError,
    ModelError, validate_numeric_value, CircuitBreaker
)
from utils.data_validation import data_validator
from utils.redis_cache import redis_cache
from utils.background_tasks import task_manager, async_batch_predict, async_generate_report
from auth import auth_router, get_current_active_user, CurrentUser
from auth.api_key_or_jwt import get_current_user_flexible
from monitoring.metrics_collector import (
    metrics_collector, record_prometheus_request, record_prometheus_prediction,
    record_prometheus_error, update_prometheus_system_metrics, get_prometheus_metrics,
    PROMETHEUS_ENABLED
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import enhanced analysis endpoint
try:
    from api_analysis_enhanced import analyze_enhanced
    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced analysis module not available")
    ENHANCED_ANALYSIS_AVAILABLE = False

# Import configuration service
try:
    from config_service import config_service
    CONFIG_SERVICE_AVAILABLE = True
except ImportError:
    logger.warning("Configuration service not available")
    CONFIG_SERVICE_AVAILABLE = False

# Import LLM endpoints
try:
    from api_llm_endpoints import llm_router, shutdown_llm_engine
    LLM_ENDPOINTS_AVAILABLE = True
except ImportError:
    logger.warning("LLM endpoints not available")
    LLM_ENDPOINTS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="FLASH API", 
    version="1.0.0",
    description="FLASH Platform API with JWT Authentication"
)

# Add global exception handlers
app.add_exception_handler(Exception, error_handler)
app.add_exception_handler(ValidationError, error_handler)
app.add_exception_handler(DataValidationError, error_handler)
app.add_exception_handler(ModelError, error_handler)

# Security middleware for API key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Validate API key from header"""
    # Bypass authentication if DISABLE_AUTH is set in development
    if settings.DISABLE_AUTH and settings.is_development():
        logger.info("Authentication disabled in development mode (DISABLE_AUTH=true)")
        return "dev_mode_bypass"
    
    # In development, allow access without API key
    if settings.is_development() and not api_key:
        logger.warning("No API key provided in development mode - allowing access")
        return None
    
    # In production, API key is required
    if settings.is_production() and not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Validate API key if provided
    if api_key:
        # Check against configured API keys
        valid_keys = settings.API_KEYS
        if not valid_keys:
            logger.error("No API keys configured - rejecting all requests")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API key validation not configured"
            )
        
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
    
    return api_key

# Configure CORS with proper security
# In development, allow all origins to avoid CORS issues
if settings.is_development():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )
else:
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
        expose_headers=["X-Request-ID"],
        max_age=3600,
    )

# Handle CORS preflight requests
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str) -> Response:
    """Handle CORS preflight requests"""
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, PUT, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type, X-API-Key'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

# Add rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add request/response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses"""
    import time
    import uuid
    
    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Log request
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Log response
        duration = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] Completed in {duration:.2f}ms - Status: {response.status_code}")
        
        # Record metrics
        metrics_collector.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_time_ms=duration,
            request_id=request_id
        )
        record_prometheus_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration/1000
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.2f}ms"
        
        return response
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Failed after {duration:.2f}ms - Error: {str(e)}")
        raise

# Initialize components
orchestrator = UnifiedOrchestratorV3()
type_converter = TypeConverter()

# Initialize circuit breakers
prediction_circuit_breaker = CircuitBreaker(
    name="prediction",
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=Exception
)

# Include authentication routes
app.include_router(auth_router)

# Include LLM endpoints if available
if LLM_ENDPOINTS_AVAILABLE:
    app.include_router(llm_router)
    logger.info("LLM endpoints enabled")

# Include Framework Intelligence endpoints
try:
    from api_framework_endpoints import framework_router
    app.include_router(framework_router)
    logger.info("Framework Intelligence endpoints enabled")
except ImportError as e:
    logger.warning(f"Could not import Framework endpoints: {e}")


def calculate_camp_scores(features: Dict[str, Any]) -> Dict[str, float]:
    """Calculate normalized CAMP scores from features (0-1 range)"""
    import numpy as np
    from feature_config import CAPITAL_FEATURES, ADVANTAGE_FEATURES, MARKET_FEATURES, PEOPLE_FEATURES
    
    # Define feature normalization rules
    MONETARY_FEATURES = [
        'total_capital_raised_usd', 'cash_on_hand_usd', 'monthly_burn_usd',
        'tam_size_usd', 'sam_size_usd', 'som_size_usd', 'annual_revenue_run_rate',
        'customer_count', 'team_size_full_time', 'founders_count', 'advisors_count',
        'competitors_named_count'
    ]
    
    PERCENTAGE_FEATURES = [
        'market_growth_rate_percent', 'user_growth_rate_percent', 
        'net_dollar_retention_percent', 'customer_concentration_percent',
        'team_diversity_percent', 'gross_margin_percent', 'revenue_growth_rate_percent'
    ]
    
    SCORE_FEATURES = [  # Already 1-5 scale
        'tech_differentiation_score', 'switching_cost_score', 'brand_strength_score',
        'scalability_score', 'board_advisor_experience_score', 'competition_intensity'
    ]
    
    # Normalize features first
    normalized = {}
    for key, value in features.items():
        if value is None:
            continue  # Skip missing values instead of defaulting
        # Skip string fields - they don't need normalization
        elif isinstance(value, str):
            continue
        elif key in MONETARY_FEATURES:
            # Log scale for monetary values
            if value > 0:
                # Map log scale to 0-1 (assumes $1K to $1B range)
                normalized[key] = np.clip(np.log10(value + 1) / 9, 0, 1)
            else:
                normalized[key] = 0
        elif key in PERCENTAGE_FEATURES:
            # Percentages: -100% to 200% mapped to 0-1
            normalized[key] = np.clip((value + 100) / 300, 0, 1)
        elif key in SCORE_FEATURES:
            # 1-5 scores mapped to 0-1
            normalized[key] = (value - 1) / 4 if value >= 1 else 0
        elif key == 'runway_months':
            # Runway: 0-24 months mapped to 0-1
            normalized[key] = np.clip(value / 24, 0, 1)
        elif key == 'burn_multiple':
            # Burn multiple: inverse (lower is better), 0-10 range
            normalized[key] = np.clip(1 - (value / 10), 0, 1)
        elif key == 'ltv_cac_ratio':
            # LTV/CAC: 0-5 mapped to 0-1
            normalized[key] = np.clip(value / 5, 0, 1)
        elif key in ['patent_count', 'prior_startup_experience_count', 'prior_successful_exits_count']:
            # Counts: 0-10 mapped to 0-1
            normalized[key] = np.clip(value / 10, 0, 1)
        elif key in ['years_experience_avg', 'domain_expertise_years_avg']:
            # Years: 0-20 mapped to 0-1
            normalized[key] = np.clip(value / 20, 0, 1)
        elif key in ['product_retention_30d', 'product_retention_90d']:
            # Retention percentages
            normalized[key] = np.clip(value / 100, 0, 1)
        elif key == 'dau_mau_ratio':
            # DAU/MAU already 0-1
            normalized[key] = np.clip(value, 0, 1)
        elif key == 'customer_concentration_percent':
            # Customer concentration: 0-100% (lower is better)
            normalized[key] = np.clip(1 - (value / 100), 0, 1)
        elif key in ['payback_period_months', 'time_to_revenue_months']:
            # Time periods: 0-24 months (lower is better)
            normalized[key] = np.clip(1 - (value / 24), 0, 1)
        else:
            # Binary features or unknown - keep as is
            try:
                normalized[key] = np.clip(float(value), 0, 1)
            except:
                continue  # Skip invalid values
    
    scores = {}
    
    # Calculate CAMP scores from normalized features
    # Capital score
    capital_features = [f for f in CAPITAL_FEATURES if f in normalized]
    if capital_features:
        scores['capital'] = np.mean([normalized[f] for f in capital_features])
    else:
        scores['capital'] = 0.0
    
    # Advantage score
    advantage_features = [f for f in ADVANTAGE_FEATURES if f in normalized]
    if advantage_features:
        scores['advantage'] = np.mean([normalized[f] for f in advantage_features])
    else:
        scores['advantage'] = 0.0
    
    # Market score
    market_features = [f for f in MARKET_FEATURES if f in normalized]
    if market_features:
        scores['market'] = np.mean([normalized[f] for f in market_features])
    else:
        scores['market'] = 0.0
    
    # People score
    people_features = [f for f in PEOPLE_FEATURES if f in normalized]
    if people_features:
        scores['people'] = np.mean([normalized[f] for f in people_features])
    else:
        scores['people'] = 0.0
    
    # Ensure all scores are bounded between 0 and 1
    for key in scores:
        scores[key] = max(0.0, min(1.0, scores[key]))
    
    # Return scores in 0-1 scale (frontend components expect this format)
    return scores


class StartupData(BaseModel):
    """Input model for startup data - matches 45 feature configuration"""
    
    # Capital Features (7)
    total_capital_raised_usd: Optional[float] = None
    cash_on_hand_usd: Optional[float] = None
    monthly_burn_usd: Optional[float] = None
    runway_months: Optional[float] = None
    burn_multiple: Optional[float] = None
    investor_tier_primary: Optional[str] = None
    has_debt: Optional[bool] = None
    
    # Advantage Features (8)
    patent_count: Optional[int] = None
    network_effects_present: Optional[bool] = None
    has_data_moat: Optional[bool] = None
    regulatory_advantage_present: Optional[bool] = None
    tech_differentiation_score: Optional[int] = Field(None, ge=1, le=5)
    switching_cost_score: Optional[int] = Field(None, ge=1, le=5)
    brand_strength_score: Optional[int] = Field(None, ge=1, le=5)
    scalability_score: Optional[int] = Field(None, ge=1, le=5)
    
    # Market Features (11)
    sector: Optional[str] = None
    tam_size_usd: Optional[float] = None
    sam_size_usd: Optional[float] = None
    som_size_usd: Optional[float] = None
    market_growth_rate_percent: Optional[float] = None
    customer_count: Optional[int] = None
    customer_concentration_percent: Optional[float] = Field(None, ge=0, le=100)
    user_growth_rate_percent: Optional[float] = None
    net_dollar_retention_percent: Optional[float] = None
    competition_intensity: Optional[int] = Field(None, ge=1, le=5)
    competitors_named_count: Optional[int] = None
    
    # People Features (10)
    founders_count: Optional[int] = None
    team_size_full_time: Optional[int] = None
    years_experience_avg: Optional[float] = None
    domain_expertise_years_avg: Optional[float] = None
    prior_startup_experience_count: Optional[int] = None
    prior_successful_exits_count: Optional[int] = None
    board_advisor_experience_score: Optional[int] = Field(None, ge=1, le=5)
    advisors_count: Optional[int] = None
    team_diversity_percent: Optional[float] = Field(None, ge=0, le=100)
    key_person_dependency: Optional[bool] = None
    
    # Product Features (9)
    product_stage: Optional[str] = None
    product_retention_30d: Optional[float] = Field(None, ge=0, le=1)
    product_retention_90d: Optional[float] = Field(None, ge=0, le=1)
    dau_mau_ratio: Optional[float] = Field(None, ge=0, le=1)
    annual_revenue_run_rate: Optional[float] = None
    revenue_growth_rate_percent: Optional[float] = None
    gross_margin_percent: Optional[float] = Field(None, ge=-100, le=100)
    ltv_cac_ratio: Optional[float] = None
    funding_stage: Optional[str] = None
    
    # Additional fields for calculations
    monthly_revenue: Optional[float] = None
    monthly_cogs: Optional[float] = None
    arpu: Optional[float] = None
    monthly_churn_rate: Optional[float] = None
    customer_acquisition_cost: Optional[float] = None
    
    # Frontend-specific fields (ignored)
    startup_name: Optional[str] = None
    hq_location: Optional[str] = None
    vertical: Optional[str] = None
    
    @field_validator('funding_stage', 'product_stage', 'sector', 'investor_tier_primary')
    @classmethod
    def lowercase_string_fields(cls, v):
        """Ensure string fields are lowercase"""
        if v and isinstance(v, str):
            # Keep hyphens as is for sectors like 'ai-ml'
            return v.lower().replace(' ', '_')
        return v
    
    @field_validator('*', mode='before')
    @classmethod
    def empty_strings_to_none(cls, v):
        """Convert empty strings to None"""
        if v == '':
            return None
        return v
    
    class Config:
        """Pydantic config"""
        # Allow extra fields that aren't in the model
        extra = 'allow'
        # Validate on assignment
        validate_assignment = True


class StartupDataStrict(StartupData):
    """Strict version of StartupData that forbids extra fields"""
    
    class Config:
        """Pydantic config for strict validation"""
        # Don't allow extra fields
        extra = 'forbid'
        # Validate on assignment
        validate_assignment = True


def transform_response_for_frontend(response: Dict) -> Dict:
    """Transform backend response to match frontend expectations"""
    
    # Check if orchestrator returned an error verdict
    if response.get('verdict') == 'ERROR':
        # Handle error response
        error_msg = response.get('error', 'Unknown error occurred')
        raise ValueError(f"Model returned error: {error_msg}")
    
    # Calculate verdict from probability
    prob = response['success_probability']
    confidence = response.get('confidence_score', 0.7)
    
    if prob >= 0.7:
        verdict = "PASS"
        strength_level = "Strong" if prob >= 0.8 else "Moderate"
    elif prob >= 0.5:
        verdict = "CONDITIONAL PASS"
        strength_level = "Moderate" if prob >= 0.6 else "Weak"
    else:
        verdict = "FAIL"
        strength_level = "Weak"
    
    # Transform the response
    transformed = {
        'success_probability': response['success_probability'],
        'confidence_interval': {
            'lower': max(0, response['success_probability'] - (1 - confidence) * 0.2),
            'upper': min(1, response['success_probability'] + (1 - confidence) * 0.2)
        },
        'verdict': verdict,
        'strength_level': strength_level,
        'pillar_scores': response.get('pillar_scores', {}),
        'risk_factors': response.get('interpretation', {}).get('risks', []),
        'success_factors': response.get('interpretation', {}).get('strengths', []),
        'processing_time_ms': response.get('processing_time_ms', 0),
        'timestamp': response.get('timestamp', datetime.now().isoformat()),
        'model_version': response.get('model_version', 'orchestrator_v3')
    }
    
    # Add pattern insights if available
    if 'pattern_analysis' in response and response['pattern_analysis']:
        transformed['pattern_insights'] = response['pattern_analysis'].get('pattern_insights', [])
        transformed['primary_patterns'] = response['pattern_analysis'].get('primary_patterns', [])
    
    return transformed


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FLASH API Server",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/predict_enhanced",
            "/predict/batch",
            "/features",
            "/patterns",
            "/system_info",
            "/health",
            "/metrics",
            "/metrics/summary",
            "/tasks",
            "/report/generate"
        ]
    }


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(
    request: Request, 
    data: Annotated[StartupData, Body()],  # Explicitly mark as body parameter
    current_user: CurrentUser = Depends(get_current_user_flexible)  # Allow API key or JWT
):
    """Standard prediction endpoint - accepts frontend data format"""
    try:
        # Log incoming data stats
        data_dict = data.model_dump()
        
        # Validation: Check if we received user data by mistake
        user_fields = {'user_id', 'username', 'id'}
        if any(field in data_dict for field in user_fields):
            # Check if this looks like user auth data
            if 'user_id' in data_dict and 'username' in data_dict and len(data_dict) < 10:
                logger.error(f"Received user data instead of startup data: {list(data_dict.keys())}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid request format",
                        "message": "Received authentication data instead of startup data. Please check your request body.",
                        "received_fields": list(data_dict.keys()),
                        "expected_type": "StartupData with fields like total_capital_raised_usd, funding_stage, sector, etc."
                    }
                )
        
        non_null_fields = sum(1 for v in data_dict.values() if v is not None)
        logger.info(f"Received prediction request with {non_null_fields} non-null fields")
        
        # Sanitize input data
        try:
            sanitized_data = sanitize_startup_data(data_dict)
        except Exception as e:
            logger.error(f"Sanitization error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Invalid input data format"
            )
        
        # Comprehensive validation
        is_valid, validation_errors, validated_data = data_validator.validate(sanitized_data)
        if not is_valid:
            logger.error(f"Validation failed: {validation_errors}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Data validation failed",
                    "validation_errors": validation_errors[:5],  # Return first 5 errors
                    "total_errors": len(validation_errors)
                }
            )
        
        # Use validated data
        sanitized_data = validated_data
        
        # Check cache first
        cache_key_data = {k: v for k, v in sanitized_data.items() if k != 'startup_name'}
        cached_result = redis_cache.get_prediction(cache_key_data)
        if cached_result:
            logger.info("Cache hit - returning cached prediction")
            metrics_collector.record_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=200,
                response_time_ms=10,  # Cache hits are fast
                request_id="cached"
            )
            return cached_result
        
        # Convert data for backend
        features = type_converter.convert_frontend_to_backend(sanitized_data)
        logger.info(f"After conversion: {len(features)} features")
        
        # Import feature config to filter to canonical features
        from feature_config import ALL_FEATURES
        
        # Filter to only the 45 canonical features expected by models
        # Use None for missing features instead of 0 to avoid type mismatches
        canonical_features = {k: features.get(k, None) for k in ALL_FEATURES}
        
        # Count how many features are actually provided
        provided_features = sum(1 for v in canonical_features.values() if v is not None)
        logger.info(f"Filtered to {len(canonical_features)} canonical features, {provided_features} non-null")
        
        # Fill in missing features with appropriate defaults
        for feature in ALL_FEATURES:
            if canonical_features[feature] is None:
                # Use type-appropriate defaults
                if feature in ['has_debt', 'network_effects_present', 'has_data_moat', 
                              'regulatory_advantage_present', 'key_person_dependency']:
                    canonical_features[feature] = False
                elif feature in ['funding_stage', 'sector', 'product_stage', 'investor_tier_primary']:
                    # String fields - use sensible defaults
                    if feature == 'funding_stage':
                        canonical_features[feature] = 'seed'
                    elif feature == 'sector':
                        canonical_features[feature] = 'saas'
                    elif feature == 'product_stage':
                        canonical_features[feature] = 'mvp'
                    elif feature == 'investor_tier_primary':
                        canonical_features[feature] = 'tier_2'
                else:
                    # Numeric fields - use 0
                    canonical_features[feature] = 0
        
        # Get prediction with circuit breaker protection
        try:
            result = prediction_circuit_breaker.call(
                orchestrator.predict,
                canonical_features
            )
        except Exception as e:
            logger.error(f"Orchestrator prediction failed: {str(e)}")
            record_prometheus_error("prediction_failed", "/predict")
            raise ModelError(
                "Failed to generate prediction",
                model_name="orchestrator",
                details={"error": str(e)}
            )
        
        # Calculate CAMP pillar scores if not provided
        if 'pillar_scores' not in result or not result.get('pillar_scores'):
            camp_scores = calculate_camp_scores(canonical_features)
            logger.info(f"Calculated CAMP scores: {camp_scores}")
            result['pillar_scores'] = camp_scores
        
        # Validate result
        if not result:
            raise ValueError("No prediction result received from model")
        if 'success_probability' not in result:
            raise ValueError("Model did not return success probability")
        if not 0 <= result['success_probability'] <= 1:
            raise ValueError(f"Invalid probability value: {result['success_probability']}")
        
        # Transform for frontend
        response = transform_response_for_frontend(result)
        
        # Add additional fields expected by frontend
        response['risk_level'] = 'high' if result['success_probability'] < 0.3 else ('medium' if result['success_probability'] < 0.7 else 'low')
        response['investment_recommendation'] = response['verdict']
        response['verdict_strength'] = response.get('strength_level', 'Moderate')
        response['camp_scores'] = result.get('pillar_scores', {})
        # Frontend expects pillar_scores, not camp_scores
        response['pillar_scores'] = result.get('pillar_scores', {})
        # Include funding stage for stage-specific display
        response['funding_stage'] = canonical_features.get('funding_stage', 'seed')
        
        # Log prediction summary
        logger.info(f"Prediction complete: {result['success_probability']:.1%} probability, "
                   f"verdict: {response['verdict']}")
        
        # Record prediction metrics
        metrics_collector.record_prediction(
            success_probability=result['success_probability'],
            confidence_score=result.get('confidence_score', 0.7),
            verdict=response['verdict'],
            processing_time_ms=response.get('processing_time_ms', 0),
            model_version=response.get('model_version', 'orchestrator_v3')
        )
        record_prometheus_prediction(
            verdict=response['verdict'],
            probability=result['success_probability'],
            model_version=response.get('model_version', 'orchestrator_v3')
        )
        
        # Cache the result
        redis_cache.set_prediction(cache_key_data, response)
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail={
                "error": "Validation Error",
                "message": str(e),
                "fields_received": non_null_fields
            }
        )
    except KeyError as e:
        logger.error(f"Missing required field: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing Required Field", 
                "message": f"Required field not found: {str(e)}",
                "hint": "Ensure all 45 features are provided"
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction Failed",
                "message": str(e),
                "type": type(e).__name__
            }
        )


@app.post("/predict_simple")
@limiter.limit("10/minute")
async def predict_simple(
    request: Request, 
    data: Annotated[StartupData, Body()],  # Explicitly mark as body parameter
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Alias for /predict for frontend compatibility"""
    return await predict(request, data, current_user)


@app.post("/predict_enhanced")
@limiter.limit("10/minute")
async def predict_enhanced(
    request: Request, 
    data: Annotated[StartupData, Body()],  # Explicitly mark as body parameter
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Enhanced prediction with full pattern analysis"""
    try:
        # Sanitize input data
        sanitized_data = sanitize_startup_data(data.model_dump())
        
        # Convert data
        features = type_converter.convert_frontend_to_backend(sanitized_data)
        
        # Import feature config to filter to canonical features
        from feature_config import ALL_FEATURES
        
        # Filter to only the 45 canonical features expected by models
        canonical_features = {k: features.get(k, 0) for k in ALL_FEATURES}
        
        # Get enhanced prediction
        result = orchestrator.predict(canonical_features)
        
        # Calculate CAMP pillar scores if not provided
        if 'pillar_scores' not in result or not result.get('pillar_scores'):
            camp_scores = calculate_camp_scores(canonical_features)
            logger.info(f"Calculated CAMP scores: {camp_scores}")
            result['pillar_scores'] = camp_scores
        
        # Extract verdict string from verdict object if needed
        if isinstance(result.get('verdict'), dict):
            verdict_str = result['verdict'].get('verdict')
            if not verdict_str:
                # Calculate verdict based on probability if missing
                prob = result.get('success_probability', 0.5)
                if prob >= 0.7:
                    verdict_str = "PASS"
                elif prob >= 0.5:
                    verdict_str = "CONDITIONAL PASS"
                else:
                    verdict_str = "FAIL"
            result['verdict'] = verdict_str
        
        # Add funding stage for frontend
        result['funding_stage'] = canonical_features.get('funding_stage', 'seed')
        
        # Return full result with pattern analysis
        return result
        
    except Exception as e:
        logger.error(f"Enhanced prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_advanced")
@limiter.limit("10/minute")
async def predict_advanced(
    request: Request, 
    data: Annotated[StartupData, Body()],  # Explicitly mark as body parameter
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Alias for /predict_enhanced"""
    return await predict_enhanced(request, data, current_user)


@app.post("/validate")
@limiter.limit("30/minute")
async def validate_data(request: Request, data: Annotated[StartupData, Body()]):
    """Validate startup data without making prediction"""
    try:
        data_dict = data.model_dump()
        
        # Count fields
        total_expected = len(ALL_FEATURES)
        non_null_fields = sum(1 for k, v in data_dict.items() if v is not None and k in ALL_FEATURES)
        missing_fields = [f for f in ALL_FEATURES if f not in data_dict or data_dict.get(f) is None]
        
        # Check field types
        type_errors = []
        for field in ALL_FEATURES:
            if field in data_dict and data_dict[field] is not None:
                value = data_dict[field]
                if field in ['funding_stage', 'investor_tier_primary', 'product_stage', 'sector']:
                    if not isinstance(value, str):
                        type_errors.append(f"{field} should be string, got {type(value).__name__}")
                elif field in TypeConverter.BOOLEAN_FIELDS:
                    if not isinstance(value, (bool, int)):
                        type_errors.append(f"{field} should be boolean, got {type(value).__name__}")
                else:
                    try:
                        float(value)
                    except:
                        type_errors.append(f"{field} should be numeric, got {type(value).__name__}")
        
        is_valid = len(missing_fields) == 0 and len(type_errors) == 0
        
        return {
            "valid": is_valid,
            "fields_received": non_null_fields,
            "fields_expected": total_expected,
            "missing_fields": missing_fields,
            "type_errors": type_errors,
            "completeness": f"{non_null_fields}/{total_expected}",
            "message": "Data is valid and ready for prediction" if is_valid else "Data validation failed"
        }
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
async def get_features():
    """Get feature documentation"""
    return {
        "total_features": len(ALL_FEATURES),
        "features": ALL_FEATURES,
        "categories": {
            "capital": 7,
            "advantage": 8,
            "market": 11,
            "people": 10,
            "product": 9
        }
    }


@app.get("/patterns")
async def get_patterns():
    """Get available patterns"""
    if orchestrator.pattern_classifier:
        patterns = orchestrator.pattern_classifier.get_all_patterns()
        return {
            "total_patterns": len(patterns),
            "patterns": patterns
        }
    else:
        return {"total_patterns": 0, "patterns": []}


@app.get("/patterns/{pattern_name}")
async def get_pattern_details(pattern_name: str):
    """Get details for a specific pattern"""
    if orchestrator.pattern_classifier:
        pattern = orchestrator.pattern_classifier.get_pattern(pattern_name)
        if pattern:
            return pattern
        else:
            raise HTTPException(status_code=404, detail=f"Pattern '{pattern_name}' not found")
    else:
        raise HTTPException(status_code=503, detail="Pattern system not available")


@app.post("/analyze_pattern")
@limiter.limit("20/minute")
async def analyze_pattern(
    request: Request, 
    data: Annotated[StartupData, Body()],  # Explicitly mark as body parameter
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Analyze patterns for a startup"""
    try:
        features = type_converter.convert_frontend_to_backend(data.model_dump())
        
        if orchestrator.pattern_classifier:
            result = orchestrator.pattern_classifier.predict(features)
            return result
        else:
            raise HTTPException(status_code=503, detail="Pattern system not available")
            
    except Exception as e:
        logger.error(f"Pattern analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_startup(
    request: Request,
    data: Annotated[StartupData, Body()],  # Explicitly mark as body parameter
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """
    Provide comprehensive, dynamic analysis for a startup
    
    Returns real benchmarks, percentiles, personalized recommendations,
    and actionable insights based on actual startup data.
    """
    if not ENHANCED_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Enhanced analysis not available. Please check server configuration."
        )
    
    try:
        # Sanitize input data
        sanitized_data = sanitize_startup_data(data.model_dump())
        
        # Convert frontend data to backend format
        features = type_converter.convert_frontend_to_backend(sanitized_data)
        
        # Filter to canonical features
        canonical_features = {k: features.get(k, 0) for k in ALL_FEATURES}
        
        # Get enhanced analysis
        analysis = await analyze_enhanced(canonical_features)
        
        # If we have a recent prediction, add it to the analysis
        cache_key_data = {
            "endpoint": "predict",
            "data": sanitized_data
        }
        cached_prediction = redis_cache.get_prediction(cache_key_data)
        if cached_prediction:
            analysis['current_prediction'] = {
                'success_probability': cached_prediction.get('success_probability'),
                'verdict': cached_prediction.get('verdict'),
                'confidence_interval': cached_prediction.get('confidence_interval')
            }
        
        # Record metrics
        metrics_collector.record_request(
            endpoint="/analyze",
            method="POST",
            status_code=200,
            response_time_ms=0  # Will be set by middleware
        )
        
        return {
            "status": "success",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    except ValueError as e:
        logger.error(f"Validation error in analysis: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Enhanced analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/investor_profiles")
async def get_investor_profiles():
    """Get sample investor profiles"""
    return [
        {
            "id": 1,
            "name": "TechVentures Capital",
            "type": "VC",
            "focus": ["B2B SaaS", "AI/ML", "Enterprise"],
            "stage": ["Series A", "Series B"],
            "typical_investment": "$5M - $20M",
            "portfolio_size": 45,
            "notable_investments": ["DataCo", "AIStartup", "CloudTech"]
        },
        {
            "id": 2,
            "name": "Innovation Partners",
            "type": "VC",
            "focus": ["Consumer Tech", "Marketplace", "FinTech"],
            "stage": ["Seed", "Series A"],
            "typical_investment": "$1M - $10M",
            "portfolio_size": 72,
            "notable_investments": ["PayApp", "MarketPlace", "FinanceAI"]
        },
        {
            "id": 3,
            "name": "Growth Equity Fund",
            "type": "PE",
            "focus": ["Late Stage", "Growth", "Scale-ups"],
            "stage": ["Series C+"],
            "typical_investment": "$20M+",
            "portfolio_size": 28,
            "notable_investments": ["ScaleUp", "GrowthCo", "MarketLeader"]
        }
    ]


@app.get("/config/stage-weights")
async def get_stage_weights():
    """Get stage-specific CAMP weights"""
    if CONFIG_SERVICE_AVAILABLE:
        try:
            return config_service.get_stage_weights_config()
        except Exception as e:
            logger.error(f"Error fetching stage weights from config service: {e}")
    
    # Fallback to hardcoded values if config service unavailable
    return {
        "pre_seed": {
            "people": 0.40,
            "advantage": 0.30,
            "market": 0.20,
            "capital": 0.10
        },
        "seed": {
            "people": 0.30,
            "advantage": 0.30,
            "market": 0.25,
            "capital": 0.15
        },
        "series_a": {
            "market": 0.30,
            "people": 0.25,
            "advantage": 0.25,
            "capital": 0.20
        },
        "series_b": {
            "market": 0.35,
            "capital": 0.25,
            "advantage": 0.20,
            "people": 0.20
        },
        "series_c": {
            "capital": 0.35,
            "market": 0.30,
            "people": 0.20,
            "advantage": 0.15
        },
        "growth": {
            "capital": 0.35,
            "market": 0.30,
            "people": 0.20,
            "advantage": 0.15
        }
    }


@app.get("/config/model-performance")
async def get_model_performance():
    """Get model performance metrics"""
    if CONFIG_SERVICE_AVAILABLE:
        try:
            return config_service.get_model_performance_config()
        except Exception as e:
            logger.error(f"Error fetching model performance from config service: {e}")
    
    # Fallback to hardcoded values if config service unavailable
    performance = {
        "dna_analyzer": {
            "name": "DNA Pattern Analyzer",
            "accuracy": 0.7711  # Should come from model metadata
        },
        "temporal_predictor": {
            "name": "Temporal Predictor", 
            "accuracy": 0.7736
        },
        "industry_model": {
            "name": "Industry-Specific Model",
            "accuracy": 0.7717
        },
        "ensemble_model": {
            "name": "Ensemble Model",
            "accuracy": 0.7401
        },
        "pattern_matcher": {
            "name": "Pattern Matcher",
            "accuracy": 0.7700
        },
        "meta_learner": {
            "name": "Meta Learner",
            "accuracy": 0.7636
        },
        "overall_accuracy": 0.7636,
        "dataset_size": "100k"
    }
    return performance


@app.get("/config/company-examples")
async def get_company_examples():
    """Get company examples for each stage"""
    if CONFIG_SERVICE_AVAILABLE:
        try:
            return config_service.get_test_data_config().get("company_examples", {})
        except Exception as e:
            logger.error(f"Error fetching company examples from config service: {e}")
    
    # Fallback to hardcoded values if config service unavailable
    return {
        "pre_seed": {
            "company": "Airbnb",
            "story": "Airbnb's founders were rejected by many VCs, but their persistence and execution skills turned a simple idea into a $75B company."
        },
        "seed": {
            "company": "Stripe",
            "story": "Stripe succeeded because the Collison brothers (team) built dramatically better payment APIs (advantage) than existing solutions."
        },
        "series_a": {
            "company": "Uber",
            "story": "Uber raised Series A after proving the ride-sharing market was massive and their model could scale beyond San Francisco."
        },
        "series_b": {
            "company": "DoorDash",
            "story": "DoorDash's Series B focused on their path to market leadership and improving delivery economics."
        },
        "series_c": {
            "company": "Spotify",
            "story": "Spotify's later rounds focused heavily on improving gross margins and reducing customer acquisition costs."
        },
        "growth": {
            "company": "Canva",
            "story": "Canva maintained high growth while achieving profitability, making it attractive for growth investors."
        }
    }


@app.get("/config/business-logic")
async def get_business_logic_config():
    """Get all business logic thresholds and rules"""
    if CONFIG_SERVICE_AVAILABLE:
        try:
            return config_service.get_business_logic_config()
        except Exception as e:
            logger.error(f"Error fetching business logic config: {e}")
    
    # Return empty dict as fallback - frontend will use its defaults
    return {}


@app.get("/config/ui")
async def get_ui_config():
    """Get UI configuration (colors, animations, etc.)"""
    if CONFIG_SERVICE_AVAILABLE:
        try:
            return config_service.get_ui_config()
        except Exception as e:
            logger.error(f"Error fetching UI config: {e}")
    
    # Return empty dict as fallback - frontend will use its defaults
    return {}


@app.get("/config/time-periods")
async def get_time_periods_config():
    """Get time period definitions"""
    if CONFIG_SERVICE_AVAILABLE:
        try:
            return config_service.get_time_periods_config()
        except Exception as e:
            logger.error(f"Error fetching time periods config: {e}")
    
    # Return empty dict as fallback - frontend will use its defaults
    return {}


@app.get("/config/default-values")
async def get_default_values_config():
    """Get default values for missing data"""
    if CONFIG_SERVICE_AVAILABLE:
        try:
            return config_service.get_default_values_config()
        except Exception as e:
            logger.error(f"Error fetching default values config: {e}")
    
    # Return empty dict as fallback - frontend will use its defaults
    return {}


@app.get("/config/all")
async def get_all_config():
    """Get all configuration in one call"""
    if CONFIG_SERVICE_AVAILABLE:
        try:
            return config_service.get_all_config()
        except Exception as e:
            logger.error(f"Error fetching all config: {e}")
    
    # Return minimal config as fallback
    return {
        "business_logic": {},
        "model_performance": {},
        "stage_weights": {},
        "ui": {},
        "test_data": {},
        "time_periods": {},
        "default_values": {},
        "environment": "production",
        "last_updated": datetime.now().isoformat()
    }


@app.get("/system_info")
async def get_system_info(current_user: CurrentUser = Depends(get_current_active_user)):
    """Get system information"""
    cache_stats = redis_cache.get_stats()
    return {
        "api_version": "1.0.0",
        "model_version": "orchestrator_v3_with_patterns",
        "feature_count": len(ALL_FEATURES),
        "pattern_count": 31 if hasattr(orchestrator, 'pattern_classifier') and orchestrator.pattern_classifier else 0,
        "models_loaded": list(orchestrator.models.keys()),
        "weights": orchestrator.weights,
        "cache": cache_stats,
        "status": "operational"
    }


@app.post("/explain")
@limiter.limit("10/minute")
async def explain_prediction(
    request: Request, 
    data: Annotated[StartupData, Body()],  # Explicitly mark as body parameter
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Generate explanations for a prediction"""
    try:
        # Get prediction first
        features = type_converter.convert_frontend_to_backend(data.model_dump())
        result = orchestrator.predict(features)
        
        # Generate explanations
        explanations = {
            'feature_importance': {},
            'decision_factors': [],
            'improvement_suggestions': [],
            'confidence_breakdown': {}
        }
        
        # Feature importance based on CAMP scores
        if 'pillar_scores' in result:
            explanations['feature_importance'] = {
                'Capital': result['pillar_scores'].get('capital', 0.5),
                'Advantage': result['pillar_scores'].get('advantage', 0.5),
                'Market': result['pillar_scores'].get('market', 0.5),
                'People': result['pillar_scores'].get('people', 0.5)
            }
        
        # Decision factors
        if result['success_probability'] > 0.7:
            explanations['decision_factors'].append("Strong overall fundamentals across CAMP dimensions")
        if result.get('pillar_scores', {}).get('market', 0) > 0.7:
            explanations['decision_factors'].append("Excellent market opportunity and growth potential")
        if result.get('pillar_scores', {}).get('people', 0) > 0.7:
            explanations['decision_factors'].append("Experienced team with proven track record")
        
        # Risk factors
        if result.get('risk_factors'):
            explanations['decision_factors'].extend([f"Risk: {risk}" for risk in result['risk_factors']])
        
        # Success factors
        if result.get('success_factors'):
            explanations['decision_factors'].extend([f"Strength: {factor}" for factor in result['success_factors']])
        
        # Improvement suggestions
        if result.get('pillar_scores', {}).get('capital', 0) < 0.5:
            explanations['improvement_suggestions'].append("Improve capital efficiency and extend runway")
        if result.get('pillar_scores', {}).get('market', 0) < 0.5:
            explanations['improvement_suggestions'].append("Strengthen market positioning and growth metrics")
        if result.get('pillar_scores', {}).get('people', 0) < 0.5:
            explanations['improvement_suggestions'].append("Build stronger team with domain expertise")
        if result.get('pillar_scores', {}).get('advantage', 0) < 0.5:
            explanations['improvement_suggestions'].append("Develop stronger competitive moats and differentiation")
        
        # Confidence breakdown
        explanations['confidence_breakdown'] = {
            'model_agreement': result.get('model_agreement', 0),
            'pattern_confidence': result.get('pattern_analysis', {}).get('pattern_score', 0.5),
            'overall_confidence': result.get('confidence_score', 0.7)
        }
        
        return {
            'prediction': result,
            'explanations': explanations,
            'methodology': "CAMP framework analysis with pattern recognition",
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        models_ok = len(orchestrator.models) > 0
        patterns_ok = hasattr(orchestrator, 'pattern_classifier') and orchestrator.pattern_classifier is not None
        
        if models_ok:
            return {
                "status": "healthy",
                "models_loaded": len(orchestrator.models),
                "patterns_available": patterns_ok,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="Models not loaded")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/cache/clear")
@limiter.limit("1/minute")
async def clear_cache(
    request: Request,
    cache_type: str = "all",
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Clear cache - requires authentication"""
    if not redis_cache.enabled:
        return {"status": "disabled", "message": "Redis cache not available"}
    
    cleared = 0
    if cache_type == "predictions" or cache_type == "all":
        cleared += redis_cache.invalidate_all_predictions()
    
    if cache_type == "patterns" or cache_type == "all":
        cleared += redis_cache.invalidate_all_patterns()
    
    return {
        "status": "success",
        "cleared": cleared,
        "cache_type": cache_type,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict/batch")
@limiter.limit("1/minute")
async def predict_batch(
    request: Request,
    startups: List[StartupData],
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Submit batch prediction job"""
    if len(startups) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 startups"
        )
    
    # Convert to dict format
    startup_dicts = [s.dict() for s in startups]
    
    # Submit background task
    task_id = task_manager.submit_task(
        async_batch_predict,
        startup_dicts
    )
    
    return {
        "task_id": task_id,
        "status": "submitted",
        "batch_size": len(startups),
        "message": "Batch prediction job submitted. Check /tasks/{task_id} for status."
    }


@app.post("/report/generate")
@limiter.limit("5/hour")
async def generate_report(
    request: Request,
    startup_id: str,
    include_patterns: bool = True,
    include_comparisons: bool = True,
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Generate comprehensive startup report (async)"""
    # Submit background task
    task_id = task_manager.submit_task(
        async_generate_report,
        startup_id,
        include_patterns=include_patterns,
        include_comparisons=include_comparisons
    )
    
    return {
        "task_id": task_id,
        "status": "submitted",
        "message": "Report generation started. Check /tasks/{task_id} for status."
    }


@app.get("/tasks")
async def get_all_tasks(
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Get all background tasks"""
    return task_manager.get_all_tasks()


@app.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Get status of a specific task"""
    task_result = task_manager.get_task_status(task_id)
    if not task_result:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return task_result.to_dict()


@app.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Cancel a pending task"""
    if task_manager.cancel_task(task_id):
        return {"status": "cancelled", "task_id": task_id}
    else:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be cancelled (not pending or not found)"
        )


@app.post("/tasks/cleanup")
@limiter.limit("1/hour")
async def cleanup_tasks(
    request: Request,
    older_than_hours: int = 24,
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Clean up old completed tasks"""
    cleaned = task_manager.cleanup_completed_tasks(
        older_than_seconds=older_than_hours * 3600
    )
    
    return {
        "status": "success",
        "cleaned": cleaned,
        "message": f"Removed {cleaned} tasks older than {older_than_hours} hours"
    }


@app.get("/metrics")
async def get_metrics(current_user: CurrentUser = Depends(get_current_user_flexible)):  # Allow API key or JWT
    """Get metrics in Prometheus format"""
    if PROMETHEUS_ENABLED:
        # Update system metrics before returning
        update_prometheus_system_metrics()
        return Response(content=get_prometheus_metrics(), media_type="text/plain")
    else:
        # Return JSON format metrics if Prometheus not available
        return metrics_collector.get_summary()


@app.get("/metrics/summary")
async def get_metrics_summary(current_user: CurrentUser = Depends(get_current_user_flexible)):  # Allow API key or JWT
    """Get human-readable metrics summary"""
    return metrics_collector.get_summary()


@app.post("/metrics/export")
@limiter.limit("1/minute")
async def export_metrics(
    request: Request, 
    filepath: str = "metrics_export.json",
    current_user: CurrentUser = Depends(get_current_active_user)
):
    """Export metrics to file"""
    success = metrics_collector.export_metrics(filepath)
    if success:
        return {"status": "success", "message": f"Metrics exported to {filepath}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to export metrics")


if __name__ == "__main__":
    logger.info("Starting FLASH API Server...")
    logger.info(f"Loaded {len(orchestrator.models)} models")
    logger.info(f"Pattern system: {'Available' if hasattr(orchestrator, 'pattern_classifier') and orchestrator.pattern_classifier else 'Not available'}")
    
    # Set up periodic system metrics collection
    import threading
    import time
    
    def collect_system_metrics():
        """Collect system metrics every 30 seconds"""
        while True:
            try:
                metrics_collector.record_system_metrics()
                if PROMETHEUS_ENABLED:
                    update_prometheus_system_metrics()
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
            time.sleep(30)
    
    # Start metrics collection in background
    metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
    metrics_thread.start()
    logger.info("Started system metrics collection")
    
    # Add shutdown handler
    @app.on_event("shutdown")
    async def shutdown():
        """Clean up resources on shutdown"""
        if LLM_ENDPOINTS_AVAILABLE:
            logger.info("Shutting down LLM engine...")
            await shutdown_llm_engine()
        logger.info("Shutdown complete")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8001)