"""
Data models and schema definitions for Review Analyzer database
Uses dataclasses for Python-side representation, DuckDB for storage
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class Sentiment(str, Enum):
    """Review sentiment classification"""
    POSITIF = "Positif"
    NEGATIF = "Négatif"
    NEUTRE = "Neutre"
    ERROR = "Error"


class ProcessingStatus(str, Enum):
    """Status of processing runs"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class Place:
    """
    Represents a discovered business location
    Primary entity linking all reviews
    """
    place_id: str  # Canonical Google place_id (ChIJ...)
    business: str  # Business/brand name (e.g., "Attijariwafa Bank")
    business_type: Optional[str] = None  # bank, hotel, restaurant, etc.
    name: str = ""  # Location name from Google Maps
    address: str = ""
    city: str = ""
    region: Optional[str] = None
    zipcode: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    google_rating: Optional[float] = None
    google_reviews_count: Optional[int] = None
    data_id: Optional[str] = None  # CID-like identifier
    resolve_status: Optional[str] = None  # How place_id was resolved
    cluster_name: Optional[str] = None  # Cluster assignment
    potential: Optional[float] = None  # Business potential score
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "place_id": self.place_id,
            "business": self.business,
            "business_type": self.business_type,
            "name": self.name,
            "address": self.address,
            "city": self.city,
            "region": self.region,
            "zipcode": self.zipcode,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "phone": self.phone,
            "website": self.website,
            "google_rating": self.google_rating,
            "google_reviews_count": self.google_reviews_count,
            "data_id": self.data_id,
            "resolve_status": self.resolve_status,
            "cluster_name": self.cluster_name,
            "potential": self.potential,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Review:
    """
    Represents a single customer review
    Links to a Place and optionally has Classifications
    """
    id: Optional[int] = None  # Auto-generated
    place_id: str = ""  # FK to Place
    author: str = ""
    rating: int = 0  # 1-5 stars
    text: str = ""
    review_date: Optional[str] = None  # Original date string from Google
    review_date_parsed: Optional[datetime] = None  # Parsed date
    language: Optional[str] = None  # Detected language (fr, ar, en)
    source: str = "google_maps"  # Review source
    created_at: datetime = field(default_factory=datetime.now)
    
    # Denormalized fields for convenience
    business: Optional[str] = None
    city: Optional[str] = None
    agency_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "place_id": self.place_id,
            "author": self.author,
            "rating": self.rating,
            "text": self.text,
            "review_date": self.review_date,
            "review_date_parsed": self.review_date_parsed,
            "language": self.language,
            "source": self.source,
            "created_at": self.created_at,
            "business": self.business,
            "city": self.city,
            "agency_name": self.agency_name,
        }


@dataclass
class Classification:
    """
    AI classification result for a review
    Stores sentiment, categories, and confidence scores
    """
    id: Optional[int] = None  # Auto-generated
    review_id: int = 0  # FK to Review
    sentiment: str = ""  # Positif, Négatif, Neutre
    categories_json: str = "[]"  # JSON array of {label, confidence}
    rationale: str = ""  # AI explanation
    model_version: str = ""  # Model used (e.g., gpt-4.1-mini)
    processing_run_id: Optional[int] = None  # FK to ProcessingRun
    confidence_threshold: float = 0.55
    created_at: datetime = field(default_factory=datetime.now)
    
    # Wide format category flags (for analysis)
    # These are populated from categories_json
    category_flags: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "review_id": self.review_id,
            "sentiment": self.sentiment,
            "categories_json": self.categories_json,
            "rationale": self.rationale,
            "model_version": self.model_version,
            "processing_run_id": self.processing_run_id,
            "confidence_threshold": self.confidence_threshold,
            "created_at": self.created_at,
        }


@dataclass
class ProcessingRun:
    """
    Tracks pipeline execution runs for audit trail
    """
    id: Optional[int] = None
    run_type: str = ""  # discovery, collection, classification
    status: str = ProcessingStatus.PENDING.value
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    config_json: str = "{}"  # Pipeline configuration used
    stats_json: str = "{}"  # Run statistics
    error_message: Optional[str] = None
    
    # Counts
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_type": self.run_type,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "config_json": self.config_json,
            "stats_json": self.stats_json,
            "error_message": self.error_message,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
        }


# =========================
# Schema SQL Definitions
# =========================
SCHEMA_SQL = """
-- Places table: Discovered business locations
CREATE TABLE IF NOT EXISTS places (
    place_id VARCHAR PRIMARY KEY,  -- Canonical Google place_id
    business VARCHAR NOT NULL,
    business_type VARCHAR,
    name VARCHAR,
    address VARCHAR,
    city VARCHAR,
    region VARCHAR,
    zipcode VARCHAR,
    latitude DOUBLE,
    longitude DOUBLE,
    phone VARCHAR,
    website VARCHAR,
    google_rating DOUBLE,
    google_reviews_count INTEGER,
    data_id VARCHAR,
    resolve_status VARCHAR,
    cluster_name VARCHAR,
    potential DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_places_business ON places(business);
CREATE INDEX IF NOT EXISTS idx_places_city ON places(city);
CREATE INDEX IF NOT EXISTS idx_places_region ON places(region);
CREATE INDEX IF NOT EXISTS idx_places_business_city ON places(business, city);

-- Reviews table: Customer reviews
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    place_id VARCHAR NOT NULL REFERENCES places(place_id),
    author VARCHAR,
    rating INTEGER CHECK (rating >= 0 AND rating <= 5),
    text VARCHAR,
    review_date VARCHAR,  -- Original string
    review_date_parsed DATE,
    language VARCHAR,
    source VARCHAR DEFAULT 'google_maps',
    business VARCHAR,  -- Denormalized
    city VARCHAR,  -- Denormalized
    agency_name VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent duplicate reviews
    UNIQUE (place_id, author, text)
);

-- Create sequence for review IDs
CREATE SEQUENCE IF NOT EXISTS seq_reviews_id START 1;

-- Indexes for reviews
CREATE INDEX IF NOT EXISTS idx_reviews_place_id ON reviews(place_id);
CREATE INDEX IF NOT EXISTS idx_reviews_business ON reviews(business);
CREATE INDEX IF NOT EXISTS idx_reviews_city ON reviews(city);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date_parsed);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);

-- Classifications table: AI classification results
CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY,
    review_id INTEGER NOT NULL REFERENCES reviews(id),
    sentiment VARCHAR CHECK (sentiment IN ('Positif', 'Négatif', 'Neutre', 'Error')),
    categories_json VARCHAR,  -- JSON array
    rationale VARCHAR,
    model_version VARCHAR,
    processing_run_id INTEGER,
    confidence_threshold DOUBLE DEFAULT 0.55,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- One classification per review (can be updated)
    UNIQUE (review_id)
);

-- Create sequence for classification IDs  
CREATE SEQUENCE IF NOT EXISTS seq_classifications_id START 1;

-- Indexes for classifications
CREATE INDEX IF NOT EXISTS idx_class_review_id ON classifications(review_id);
CREATE INDEX IF NOT EXISTS idx_class_sentiment ON classifications(sentiment);
CREATE INDEX IF NOT EXISTS idx_class_run_id ON classifications(processing_run_id);

-- Processing runs table: Audit trail
CREATE TABLE IF NOT EXISTS processing_runs (
    id INTEGER PRIMARY KEY,
    run_type VARCHAR NOT NULL,
    status VARCHAR DEFAULT 'pending',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    config_json VARCHAR,
    stats_json VARCHAR,
    error_message VARCHAR,
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0
);

-- Create sequence for run IDs
CREATE SEQUENCE IF NOT EXISTS seq_runs_id START 1;

-- Category flags table: Wide format for analysis
-- Stores binary flags for each category per review
CREATE TABLE IF NOT EXISTS review_categories (
    review_id INTEGER PRIMARY KEY REFERENCES reviews(id),
    
    -- Positive categories
    cat_accueil_chaleureux INTEGER DEFAULT 0,
    cat_service_reactif INTEGER DEFAULT 0,
    cat_conseil_personnalise INTEGER DEFAULT 0,
    cat_efficacite_rapidite INTEGER DEFAULT 0,
    cat_accessibilite_proximite INTEGER DEFAULT 0,
    cat_satisfaction_generale INTEGER DEFAULT 0,
    cat_experience_digitale INTEGER DEFAULT 0,
    
    -- Negative categories
    cat_attente_lenteur INTEGER DEFAULT 0,
    cat_service_injoignable INTEGER DEFAULT 0,
    cat_reclamations_ignorees INTEGER DEFAULT 0,
    cat_incidents_techniques INTEGER DEFAULT 0,
    cat_frais_abusifs INTEGER DEFAULT 0,
    cat_insatisfaction_generale INTEGER DEFAULT 0,
    cat_manque_consideration INTEGER DEFAULT 0,
    
    -- Neutral/Other
    cat_hors_sujet INTEGER DEFAULT 0,
    cat_autre_positif INTEGER DEFAULT 0,
    cat_autre_negatif INTEGER DEFAULT 0
);

-- Views for common queries
CREATE OR REPLACE VIEW v_reviews_full AS
SELECT 
    r.id AS review_id,
    r.place_id,
    r.author,
    r.rating,
    r.text,
    r.review_date,
    r.review_date_parsed,
    r.language,
    r.business,
    r.city,
    r.agency_name,
    p.region,
    p.latitude,
    p.longitude,
    p.cluster_name,
    p.potential,
    c.sentiment,
    c.categories_json,
    c.rationale,
    c.model_version
FROM reviews r
LEFT JOIN places p ON r.place_id = p.place_id
LEFT JOIN classifications c ON r.id = c.review_id;

-- Aggregation view by business
CREATE OR REPLACE VIEW v_business_stats AS
SELECT 
    r.business,
    COUNT(*) AS total_reviews,
    AVG(r.rating) AS avg_rating,
    COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) AS positive_count,
    COUNT(CASE WHEN c.sentiment = 'Négatif' THEN 1 END) AS negative_count,
    COUNT(CASE WHEN c.sentiment = 'Neutre' THEN 1 END) AS neutral_count,
    ROUND(COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) * 100.0 / COUNT(*), 2) AS positive_rate
FROM reviews r
LEFT JOIN classifications c ON r.id = c.review_id
GROUP BY r.business;

-- Aggregation view by city
CREATE OR REPLACE VIEW v_city_stats AS
SELECT 
    r.city,
    COUNT(*) AS total_reviews,
    AVG(r.rating) AS avg_rating,
    COUNT(DISTINCT r.place_id) AS place_count,
    COUNT(DISTINCT r.business) AS business_count,
    COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) AS positive_count,
    COUNT(CASE WHEN c.sentiment = 'Négatif' THEN 1 END) AS negative_count
FROM reviews r
LEFT JOIN classifications c ON r.id = c.review_id
GROUP BY r.city;

-- Temporal aggregation view
CREATE OR REPLACE VIEW v_temporal_stats AS
SELECT 
    DATE_TRUNC('month', r.review_date_parsed) AS month,
    r.business,
    COUNT(*) AS review_count,
    AVG(r.rating) AS avg_rating,
    COUNT(CASE WHEN c.sentiment = 'Positif' THEN 1 END) AS positive_count,
    COUNT(CASE WHEN c.sentiment = 'Négatif' THEN 1 END) AS negative_count
FROM reviews r
LEFT JOIN classifications c ON r.id = c.review_id
WHERE r.review_date_parsed IS NOT NULL
GROUP BY DATE_TRUNC('month', r.review_date_parsed), r.business
ORDER BY month, r.business;
"""

# Category column mapping (short name -> full category)
CATEGORY_COLUMN_MAP = {
    "cat_accueil_chaleureux": "Accueil chaleureux et personnel attentionné (expérience humaine positive, sentiment d'être bien accueilli)",
    "cat_service_reactif": "Service client réactif et à l'écoute (problèmes résolus rapidement, vraie disponibilité)",
    "cat_conseil_personnalise": "Conseil personnalisé et professionnalisme des équipes (expertise perçue, accompagnement individualisé)",
    "cat_efficacite_rapidite": "Efficacité et rapidité de traitement (fluidité, peu d'attente, processus clairs)",
    "cat_accessibilite_proximite": "Accessibilité et proximité des services (agences, guichets, présence locale, simplicité d'accès)",
    "cat_satisfaction_generale": "Satisfaction sans détails spécifiques (le client indique que le service ou l'agence est bien sans explication )",
    "cat_experience_digitale": "Expérience digitale et services en ligne pratiques (application fluide, opérations faciles à distance )",
    "cat_attente_lenteur": "Attente interminable et lenteur en agence (files d'attente, effectifs insuffisants)",
    "cat_service_injoignable": "Service client injoignable ou non réactif (téléphone, e-mail, promesses de rappel non tenues)",
    "cat_reclamations_ignorees": "Réclamations ignorées ou mal suivies (absence de retour, sentiment d'abandon)",
    "cat_incidents_techniques": "Incidents techniques et erreurs récurrentes (cartes bloquées, pannes système, erreurs de compte)",
    "cat_frais_abusifs": "Frais bancaires jugés abusifs ou non justifiés (perception de déséquilibre prix/service)",
    "cat_insatisfaction_generale": "Insatisfaction sans détails spécifiques (le client indique que le service ou l'agence est nul sans explication )",
    "cat_manque_consideration": "Manque de considération ou attitude peu professionnelle (accueil froid, ton condescendant, sentiment de mépris)",
    "cat_hors_sujet": "Hors-sujet ou contenu non pertinent (ex. « Bon restau », « Je cherche du travail »)",
    "cat_autre_positif": "Autre (positif)",
    "cat_autre_negatif": "Autre (négatif)",
}

# Reverse mapping
CATEGORY_TO_COLUMN_MAP = {v: k for k, v in CATEGORY_COLUMN_MAP.items()}

