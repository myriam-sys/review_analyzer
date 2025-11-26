"""
Central configuration for Review Analyzer Pipeline
Handles all environment variables, paths, and settings
"""
import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================
# Base Paths - New Architecture
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# New folder structure
CONFIG_DIR = DATA_DIR / "00_config"
RAW_DIR = DATA_DIR / "01_raw"
INTERIM_DIR = DATA_DIR / "02_interim"
PROCESSED_DIR = DATA_DIR / "03_processed"
ANALYSIS_DIR = DATA_DIR / "04_analysis"
ARCHIVE_DIR = DATA_DIR / "99_archive"

# Legacy aliases for backward compatibility
DEFAULT_DIR = CONFIG_DIR  # Old "default" → new "00_config"
INPUT_DIR = RAW_DIR  # Old "input" → new "01_raw"
OUTPUT_DIR = PROCESSED_DIR  # Old "output" → new "03_processed"

# Subdirectories
RAW_DISCOVERY_DIR = RAW_DIR / "discovery"
RAW_REVIEWS_DIR = RAW_DIR / "reviews" / "GoogleMaps"
INTERIM_DISCOVERY_DIR = INTERIM_DIR / "discovery"
INTERIM_COLLECTION_DIR = INTERIM_DIR / "collection"
INTERIM_TRANSFORM_DIR = INTERIM_DIR / "transform"
CHECKPOINTS_DIR = INTERIM_COLLECTION_DIR / "checkpoints"
CACHE_DIR = INTERIM_DISCOVERY_DIR / "cache"

LOGS_DIR = PROJECT_ROOT / "logs"

# Processed classified files directories
PROC_CLASSIFIED_DIR = PROCESSED_DIR / "classification"
PROC_CLASSIFIED_LATEST = PROC_CLASSIFIED_DIR / "latest"


# Ensure critical directories exist
for dir_path in [
    CONFIG_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, 
    ANALYSIS_DIR, LOGS_DIR, CHECKPOINTS_DIR, CACHE_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =========================
# API Configuration
# =========================
# SerpAPI for scraping
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
if not SERPAPI_KEY:
    raise RuntimeError(
        "SERPAPI_API_KEY not found. Please set it in .env file or environment variables."
    )

# OpenAI for classification
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Please set it in .env file or environment variables."
    )

# =========================
# SerpAPI Settings
# =========================
SERPAPI_CONFIG = {
    "api_key": SERPAPI_KEY,
    "hl": "fr",  # Language
    "gl": "ma",  # Geo-location (Morocco)
    "delay_seconds": 1.2,  # Delay between requests
    "max_retries": 5,
    "retry_min_wait": 1,
    "retry_max_wait": 20,
    "timeout": 60,
}

# =========================
# OpenAI Settings
# =========================
OPENAI_CONFIG = {
    "api_key": OPENAI_API_KEY,
    "model": "gpt-4.1-mini",
    "temperature": 0,
    "max_retries": 5,
    "retry_min_wait": 1,
    "retry_max_wait": 20,
}

# =========================
# Classification Categories
# =========================
CATEGORIES: List[str] = [
    # POSITIF
    "Accueil chaleureux et personnel attentionné (expérience humaine positive, sentiment d'être bien accueilli)",
    "Service client réactif et à l'écoute (problèmes résolus rapidement, vraie disponibilité)",
    "Conseil personnalisé et professionnalisme des équipes (expertise perçue, accompagnement individualisé)",
    "Efficacité et rapidité de traitement (fluidité, peu d'attente, processus clairs)",
    "Accessibilité et proximité des services (agences, guichets, présence locale, simplicité d'accès)",
    "Satisfaction sans détails spécifiques (le client indique que le service ou l'agence est bien sans explication )",
    "Expérience digitale et services en ligne pratiques (application fluide, opérations faciles à distance )",

    # NÉGATIF
    "Attente interminable et lenteur en agence (files d'attente, effectifs insuffisants)",
    "Service client injoignable ou non réactif (téléphone, e-mail, promesses de rappel non tenues)",
    "Réclamations ignorées ou mal suivies (absence de retour, sentiment d'abandon)",
    "Incidents techniques et erreurs récurrentes (cartes bloquées, pannes système, erreurs de compte)",
    "Frais bancaires jugés abusifs ou non justifiés (perception de déséquilibre prix/service)",
    "Insatisfaction sans détails spécifiques (le client indique que le service ou l'agence est nul sans explication )",
    "Manque de considération ou attitude peu professionnelle (accueil froid, ton condescendant, sentiment de mépris)",

    # NEUTRE
    "Hors-sujet ou contenu non pertinent (ex. « Bon restau », « Je cherche du travail »)",

    # AUTRE
    "Autre (positif)",
    "Autre (négatif)"
]

# Backward compatibility alias
REVIEW_CATEGORIES = CATEGORIES

# Threshold for category assignment
CONF_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = CONF_THRESHOLD  # Backward compatibility

# =========================
# Processing Settings
# =========================
PROCESSING_CONFIG = {
    "batch_size": 100,  # Reviews per batch for classification
    "pause_seconds": 0.5,  # Pause between batches
    "checkpoint_interval": 10,  # Save checkpoint every N operations
    "max_workers": 3,  # For parallel processing (future)
}

# =========================
# Default Files - New Paths
# =========================
DEFAULT_PLACES_FILE = CONFIG_DIR / "templates" / "banks_template.csv"
CITY_ALIASES_FILE = CONFIG_DIR / "cities" / "aliases.json"
OSM_CITY_FILE = CONFIG_DIR / "cities" / "coordinates.ndjson"
REGIONS_FILE = CONFIG_DIR / "cities" / "regions.geojson"

# Legacy aliases
DEFAULT_MAP_CENTERS_FILE = CITY_ALIASES_FILE  # Deprecated

# Sample map centers for Moroccan cities
DEFAULT_MAP_CENTERS = {
    "Rabat": "@34.020882,-6.841650,12z",
    "Casablanca": "@33.589886,-7.603869,12z",
    "Marrakech": "@31.629472,-7.981084,12z",
    "Fes": "@34.033333,-5.000000,12z",
    "Tangier": "@35.759465,-5.833954,12z",
    "Agadir": "@30.427755,-9.598107,12z",
    "Meknes": "@33.893791,-5.547080,12z",
    "Oujda": "@34.680000,-1.910000,12z",
    "Kenitra": "@34.261000,-6.580000,12z",
    "Tetouan": "@35.578780,-5.368470,12z",
    "Safi": "@32.299400,-9.237200,12z",
    "Mohammedia": "@33.686000,-7.383000,12z",
    "Beni Mellal": "@32.337300,-6.359900,12z",
}


# =========================
# Validation Settings
# =========================
VALID_PLACE_ID_PATTERN = r"^ChIJ[0-9A-Za-z_-]+$"  # Canonical place_id format
MIN_REVIEWS_PER_PLACE = 0  # Minimum expected reviews (0 = allow empty)
MAX_RATING = 5
MIN_RATING = 1

# =========================
# Logging Configuration
# =========================
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "pipeline.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "review_analyzer": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        }
    },
    "root": {"level": "INFO", "handlers": ["console", "file"]},
}

# =========================
# Output Modes
# =========================
class OutputMode:
    """Valid output modes for review collection"""
    JSON_PER_CITY = "json-per-city"
    JSON_PER_BUSINESS = "json-per-business"
    SINGLE_JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    
    # Backward compatibility alias (deprecated)
    JSON_PER_BANK = "json-per-business"  # Deprecated


# =========================
# Helper Functions
# =========================
def get_output_path(
    step: str,
    filename: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Get standardized output path for a pipeline step
    
    Args:
        step: Pipeline step name ('discovery', 'collection', 'classification')
        filename: Output filename
        output_dir: Optional custom output directory
        
    Returns:
        Path object for output file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    step_dir = output_dir / step
    step_dir.mkdir(parents=True, exist_ok=True)
    
    return step_dir / filename


def validate_place_id(place_id: str) -> bool:
    """
    Validate if a place_id matches the canonical format
    
    Args:
        place_id: Google Maps place_id
        
    Returns:
        True if valid canonical format
    """
    import re
    return bool(re.match(VALID_PLACE_ID_PATTERN, place_id))


# =========================
# Environment Info
# =========================
def print_config_summary():
    """Print configuration summary for debugging"""
    print("=" * 60)
    print("Review Analyzer Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print(f"\nSerpAPI Key: {'✓ Set' if SERPAPI_KEY else '✗ Missing'}")
    print(f"OpenAI Key: {'✓ Set' if OPENAI_API_KEY else '✗ Missing'}")
    print("\nProcessing:")
    print(f"  Batch Size: {PROCESSING_CONFIG['batch_size']}")
    print(f"  Max Workers: {PROCESSING_CONFIG['max_workers']}")
    print(f"  Checkpoint Interval: {PROCESSING_CONFIG['checkpoint_interval']}")
    print(f"\nCategories: {len(REVIEW_CATEGORIES)}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
