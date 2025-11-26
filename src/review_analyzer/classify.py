"""
Step 3: Review Classification with OpenAI
Classify reviews into predefined categories using GPT models
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import OpenAI

from . import config, utils

logger = logging.getLogger(__name__)


# =========================
# Few-Shot Examples
# =========================
def get_fewshot_examples() -> List[Dict[str, str]]:
    """Return few-shot examples for better classification"""
    return [
        # (1) Mixed: wait (main) + positive welcome
        {
            "role": "user",
            "content": "Avis: 'Beaucoup d'attente mais le personnel reste aimable, merci.'"
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "sentiment": "Négatif",
                "categories": [
                    {
                        "label": "Attente interminable et lenteur en agence (files d'attente, effectifs insuffisants)",
                        "confidence": 0.85
                    },
                    {
                        "label": "Accueil chaleureux et personnel attentionné (expérience humaine positive, sentiment d'être bien accueilli)",
                        "confidence": 0.55
                    }
                ],
                "language": "fr",
                "rationale": "Grief principal = attente; l'amabilité est secondaire."
            }, ensure_ascii=False)
        },
        # (2) Digital positive
        {
            "role": "user",
            "content": "Avis: 'Application claire, virements rapides, je gagne du temps 👍'"
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "sentiment": "Positif",
                "categories": [
                    {
                        "label": "Expérience digitale et services en ligne pratiques (application fluide, opérations faciles à distance )",
                        "confidence": 0.89
                    },
                    {
                        "label": "Efficacité et rapidité de traitement (fluidité, peu d'attente, processus clairs)",
                        "confidence": 0.62
                    }
                ],
                "language": "fr",
                "rationale": "App fluide et processus rapides."
            }, ensure_ascii=False)
        },
        # (3) Generic dissatisfaction
        {
            "role": "user",
            "content": "Avis: 'Nul nul nul service nul.'"
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "sentiment": "Négatif",
                "categories": [
                    {
                        "label": "Insatisfaction sans détails spécifiques (le client indique que le service ou l'agence est nul sans explication )",
                        "confidence": 0.90
                    }
                ],
                "language": "fr",
                "rationale": "Jugement négatif sans cause."
            }, ensure_ascii=False)
        },
        # (4) Off-topic
        {
            "role": "user",
            "content": "Avis: 'Bon restau à côté, je recommande le tajine.'"
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "sentiment": "Neutre",
                "categories": [
                    {
                        "label": "Hors-sujet ou contenu non pertinent (ex. « Bon restau », « Je cherche du travail »)",
                        "confidence": 0.95
                    }
                ],
                "language": "fr",
                "rationale": "Non lié à la banque."
            }, ensure_ascii=False)
        },
    ]


# =========================
# Custom Exceptions
# =========================
class RateLimitError(Exception):
    """OpenAI rate limit exceeded"""
    pass


# =========================
# Prompts and Schema
# =========================
SYSTEM_PROMPT = (
    "Tu es un analyste d'avis clients pour des agences bancaires au Maroc. "
    "Lis chaque avis (français, anglais, arabe/darija, emojis) "
    "et classe-le selon les catégories suivantes :\n\n"

    "CATÉGORIES POSITIVES:\n"
    "- Accueil chaleureux et personnel attentionné (expérience humaine positive, sentiment d'être bien accueilli)\n"
    "- Service client réactif et à l'écoute (problèmes résolus rapidement, vraie disponibilité)\n"
    "- Conseil personnalisé et professionnalisme des équipes (expertise perçue, accompagnement individualisé)\n"
    "- Efficacité et rapidité de traitement (fluidité, peu d'attente, processus clairs)\n"
    "- Accessibilité et proximité des services (agences, guichets, présence locale, simplicité d'accès)\n"
    "- Satisfaction sans détails spécifiques (le client indique que le service ou l'agence est bien sans explication )\n"
    "- Expérience digitale et services en ligne pratiques (application fluide, opérations faciles à distance )\n\n"

    "CATÉGORIES NÉGATIVES:\n"
    "- Attente interminable et lenteur en agence (files d'attente, effectifs insuffisants)\n"
    "- Service client injoignable ou non réactif (téléphone, e-mail, promesses de rappel non tenues)\n"
    "- Réclamations ignorées ou mal suivies (absence de retour, sentiment d'abandon)\n"
    "- Incidents techniques et erreurs récurrentes (cartes bloquées, pannes système, erreurs de compte)\n"
    "- Frais bancaires jugés abusifs ou non justifiés (perception de déséquilibre prix/service)\n"
    "- Insatisfaction sans détails spécifiques (le client indique que le service ou l'agence est nul sans explication )\n"
    "- Manque de considération ou attitude peu professionnelle (accueil froid, ton condescendant, sentiment de mépris)\n\n"

    "CATÉGORIES NEUTRES:\n"
    "- Hors-sujet ou contenu non pertinent (ex. « Bon restau », « Je cherche du travail »)\n\n"

    "CATÉGORIES AUTRES:\n"
    "- Autre (positif)\n"
    "- Autre (négatif)\n\n"


    "RÈGLES DE CLASSIFICATION:\n"
    "- Si le client parle d'une longue file, peu de personnel → 'Attente interminable et lenteur en agence'\n"
    "- Si le client critique la politesse → 'Manque de considération ou attitude peu professionnelle'\n"
    "- Si le problème est technique → 'Incidents techniques et erreurs récurrentes'\n"
    "- Si le client parle d'une application/banque en ligne fluide → 'Expérience digitale et services en ligne pratiques'\n"
    "- Si le service client est injoignable/ne rappelle pas → 'Service client injoignable ou non réactif'\n"
    "- Si les coûts/frais sont jugés abusifs ou trop importants par rapport au montant du retrait → 'Frais bancaires jugés abusifs ou non justifiés'\n"
    "- Si l'avis est mixte/mitigé, descriptif neutre, hors sujet, ou comparaison non justifiée → choisir les catégories NEUTRES correspondantes\n"
    "- Il est possible de choisir plusieurs catégories pour un même avis\n"
    "- Le maximum de catégories qu'un avis peut avoir est 3 catégories\n\n"
    "- Si aucune catégorie ne corresond, il faut assigner la catégorie Autre en fonction du sentiment\n\n"

    "POLITIQUE D'ABSTENTION:\n"
    "- Ne force JAMAIS une catégorie spécifique si aucune n'est clairement justifiée par le texte\n"
    "- Si le ton est POSITIF mais qu'aucune catégorie spécifique ne s'applique → choisir « Autre (positif) ».\n"
    "- Si le ton est NÉGATIF mais qu'aucune catégorie spécifique ne s'applique → choisir « Autre (négatif) ».\n"
    "- Si le contenu n'est pas lié à la banque → « Hors-sujet ou contenu non pertinent ».\n\n"

    "CONTRAINTES:\n"
    "- Les catégories doivent être triées par pertinence (1 = plus pertinente)\n"
    "- N'ajoute AUCUNE catégorie hors liste\n"
    "- Renvoie un JSON strict avec les clés : sentiment, categories (liste d'objets {label, confidence}), language, rationale\n"
)


def get_json_schema() -> Dict[str, Any]:
    """OpenAI structured output schema"""
    return {
        "name": "review_multi_class",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["Positif", "Neutre", "Négatif"]
                },
                "categories": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": config.CATEGORIES
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            }
                        },
                        "required": ["label", "confidence"]
                    }
                },
                "language": {"type": "string"},
                "rationale": {"type": "string", "maxLength": 300}
            },
            "required": ["sentiment", "categories", "language", "rationale"]
        },
        "strict": True
    }


# =========================
# Classifier
# =========================
class ReviewClassifier:
    """Classify reviews using OpenAI API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize classifier
        
        Args:
            api_key: OpenAI API key (uses config if None)
            model: Model name (uses config if None)
            debug: Enable debug logging
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_CONFIG["model"]
        self.client = OpenAI(api_key=self.api_key)
        self.schema = get_json_schema()
        self.debug = debug
        
        logger.info(f"Classifier initialized with model: {self.model}")
    
    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(RateLimitError),
    )
    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call OpenAI API with retry logic"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,  # Deterministic output
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": self.schema
                }
            )
            
            # Try multiple extraction methods
            msg = response.choices[0].message
            result = self._extract_from_message(msg)
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate_limit" in error_msg or "429" in error_msg:
                logger.warning("Rate limit hit, retrying...")
                raise RateLimitError("OpenAI rate limit exceeded")
            raise
    
    def _extract_from_message(self, msg) -> Dict[str, Any]:
        """Extract JSON from OpenAI message with multiple fallbacks"""
        # 1) Try structured outputs (parsed attribute)
        parsed = getattr(msg, "parsed", None)
        if parsed is not None:
            return parsed if isinstance(parsed, dict) else parsed.model_dump()
        
        # 2) Try content as string JSON
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return json.loads(content.strip())
        
        # 3) Try content as list with text items
        if isinstance(content, list) and content:
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        texts.append(item["text"])
                else:
                    if getattr(item, "type", None) == "text":
                        texts.append(getattr(item, "text", ""))
            
            if texts:
                return json.loads("".join(texts).strip())
        
        # 4) Try model_dump()
        as_dict = getattr(msg, "model_dump", None)
        if callable(as_dict):
            d = msg.model_dump()
            if isinstance(d.get("content"), str):
                return json.loads(d["content"].strip())
            if isinstance(d.get("content"), list):
                texts = [
                    it.get("text", "") for it in d["content"]
                    if it.get("type") == "text"
                ]
                if texts:
                    return json.loads("".join(texts).strip())
        
        raise ValueError(
            "Unable to extract structured JSON from OpenAI message"
        )
    
    def classify_review(
        self,
        review_text: str,
        rating: int
    ) -> Dict[str, Any]:
        """
        Classify a single review with few-shot examples
        
        Args:
            review_text: Review content
            rating: Star rating (1-5)
            
        Returns:
            Classification result with sentiment, categories, etc.
        """
        user_content = f"Avis: {review_text}\nNote (si disponible): {rating}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *get_fewshot_examples(),  # Add few-shot examples
            {"role": "user", "content": user_content}
        ]
        
        result = self._call_api(messages)
        
        # Apply threshold + fallback logic
        filtered_cats = self._apply_threshold_and_fallback(result)
        result["categories"] = filtered_cats
        
        return result
    
    def _apply_threshold_and_fallback(
        self, result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter categories by confidence threshold and apply fallback logic
        
        If no categories pass threshold, assign Autre (±) based on sentiment
        
        Args:
            result: Raw classification result from API
            
        Returns:
            Filtered list of categories with confidence >= threshold
        """
        raw = result.get("categories", []) or []
        filtered = []
        
        for c in raw:
            lbl = c.get("label")
            conf = c.get("confidence", 0.0) or 0.0
            if lbl in config.CATEGORIES and conf >= config.CONF_THRESHOLD:
                filtered.append({"label": lbl, "confidence": float(conf)})
        
        # Fallback if no categories retained
        if not filtered:
            sent = (result.get("sentiment") or "").lower()
            if "positif" in sent:
                filtered = [{"label": "Autre (positif)", "confidence": 0.6}]
            elif "négatif" in sent or "negatif" in sent:
                filtered = [{"label": "Autre (négatif)", "confidence": 0.6}]
            # If 'Neutre' + off-topic, model should have returned neutral cat
        
        return filtered
    
    def classify_batch(
        self,
        df: pd.DataFrame,
        review_col: Optional[str] = None,
        rating_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Classify all reviews in DataFrame
        
        Auto-detects column names from new collection format (text, rating)
        or falls back to old format (review_snippet, review_rating)
        
        Args:
            df: DataFrame with reviews
            review_col: Review text column name (auto-detected if None)
            rating_col: Rating column name (auto-detected if None)
            
        Returns:
            DataFrame with classification columns added
        """
        # Auto-detect column names
        if review_col is None:
            if "text" in df.columns:
                review_col = "text"
            elif "review_snippet" in df.columns:
                review_col = "review_snippet"
            else:
                raise ValueError("No review text column found (expected 'text' or 'review_snippet')")
        
        if rating_col is None:
            if "rating" in df.columns:
                rating_col = "rating"
            elif "review_rating" in df.columns:
                rating_col = "review_rating"
            else:
                raise ValueError("No rating column found (expected 'rating' or 'review_rating')")
        
        logger.info(f"Classifying {len(df)} reviews...")
        logger.info(f"Using columns: review='{review_col}', rating='{rating_col}'")
        
        # Checkpoint manager
        checkpoint_mgr = utils.CheckpointManager(
            config.LOGS_DIR / "classification_checkpoint.json"
        )
        checkpoint = checkpoint_mgr.load()
        start_idx = checkpoint.get("last_index", 0)
        
        # Initialize columns
        if "sentiment" not in df.columns:
            df["sentiment"] = None
            df["categories_json"] = None
            df["language"] = None
            df["rationale"] = None
        
        progress = utils.create_progress_bar(
            len(df) - start_idx,
            desc="Classifying"
        )
        
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            review_text = str(row[review_col])
            rating = int(row[rating_col]) if pd.notna(row[rating_col]) else 0
            
            # Skip empty
            if not review_text or review_text == "nan":
                if progress:
                    progress.update(1)
                continue
            
            try:
                result = self.classify_review(review_text, rating)
                
                df.at[idx, "sentiment"] = result.get("sentiment")
                df.at[idx, "categories_json"] = json.dumps(
                    result.get("categories", [])
                )
                df.at[idx, "language"] = result.get("language")
                df.at[idx, "rationale"] = result.get("rationale")
                
                if self.debug and idx % 10 == 0:
                    logger.debug(f"Processed {idx}/{len(df)}")
                
                # Checkpoint
                if (idx + 1) % config.PROCESSING_CONFIG["checkpoint_interval"] == 0:
                    checkpoint_mgr.save({"last_index": idx + 1})
                
                time.sleep(config.OPENAI_CONFIG.get("delay_seconds", 0.5))
                
            except Exception as e:
                logger.error(f"Failed to classify row {idx}: {e}")
                df.at[idx, "sentiment"] = "Error"
            
            if progress:
                progress.update(1)
        
        if progress:
            progress.close()
        
        checkpoint_mgr.clear()
        
        return df
    
    def convert_to_wide_format(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert categories JSON to wide format (one column per category)
        
        Args:
            df: DataFrame with categories_json column
            
        Returns:
            DataFrame with category columns (0/1 flags)
        """
        logger.info("Converting to wide format...")
        
        # Create binary columns for each category
        for category in config.CATEGORIES:
            df[category] = 0
        
        for idx, row in df.iterrows():
            cats_json = row.get("categories_json")
            if not cats_json or pd.isna(cats_json):
                continue
            
            try:
                categories = json.loads(cats_json)
                for cat in categories:
                    label = cat.get("label")
                    if label in config.CATEGORIES:
                        df.at[idx, label] = 1
            except Exception as e:
                logger.warning(f"Failed to parse categories at row {idx}: {e}")
        
        return df


# =========================
# CLI Interface
# =========================
def main():
    """CLI entry point for classification"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Classify reviews using OpenAI"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV with reviews"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV with classifications"
    )
    parser.add_argument(
        "--review-col",
        default=None,
        help="Review text column name (auto-detected: 'text' or 'review_snippet')"
    )
    parser.add_argument(
        "--rating-col",
        default=None,
        help="Rating column name (auto-detected: 'rating' or 'review_rating')"
    )
    parser.add_argument(
        "--wide-format",
        action="store_true",
        help="Convert categories to wide format"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Read data
    logger.info(f"Reading: {args.input}")
    df = pd.read_csv(args.input)
    
    # Classify
    classifier = ReviewClassifier(debug=args.debug)
    df = classifier.classify_batch(
        df,
        review_col=args.review_col,
        rating_col=args.rating_col
    )
    
    # Wide format
    if args.wide_format:
        df = classifier.convert_to_wide_format(df)
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Saved: {args.output}")
    
    print(f"\n Classification complete!")
    print(f"   Reviews processed: {len(df)}")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
