"""
Step 1: Place Discovery
Discover business locations on Google Maps and resolve canonical place_ids
"""
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Any
import re
import pandas as pd

from . import config, utils, geo

logger = logging.getLogger(__name__)

# Canonical Place ID pattern
_CANONICAL_RX = re.compile(r"^ChIJ[0-9A-Za-z_-]+$")


def _is_canonical_place_id(pid: Optional[str]) -> bool:
    return bool(pid and _CANONICAL_RX.match(pid))


class DiscoveryEngine:
    """
    Discovers business locations across cities and resolves canonical place_ids
    """

    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        """
        Initialize discovery engine
        """
        self.client = utils.SerpAPIClient(api_key=api_key, debug=debug)
        self.debug = debug

    def discover_branches(
        self,
        businesses: List[str],
        cities: List[str],
        business_type: Optional[str] = None,
        map_centers: Optional[Dict[str, str]] = None,
        brand_filter: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Discover locations for businesses across cities
        """
        logger.info(
            f"Starting discovery: {len(businesses)} businesses × {len(cities)} cities"
        )

        if map_centers is None:
            # Prefer OSM-derived centers if available; fallback to defaults
            map_centers = geo.load_map_centers()

        brand_pattern = re.compile(brand_filter, re.I) if brand_filter else None

        all_results: List[Dict[str, Any]] = []
        progress = utils.create_progress_bar(
            len(businesses) * len(cities), desc="Discovering locations"
        )

        # Resolve provided city names via alias index to canonical keys
        resolved_cities = [geo.resolve_city_name(c) for c in cities]

        for business in businesses:
            for city in resolved_cities:
                logger.info(f"Discovering: {business} in {city}")
                center_ll = map_centers.get(city)
                results = self._discover_for_city_and_business(
                    business=business,
                    city=city,
                    business_type=business_type,
                    center_ll=center_ll,
                    brand_pattern=brand_pattern,
                )
                all_results.extend(results)
                if progress:
                    progress.update(1)

        if progress:
            progress.close()

        # Deduplicate and resolve
        df = self._deduplicate_results(all_results)
        df = self._resolve_canonical_ids(df)

        logger.info(f"Discovery complete: {len(df)} unique locations found")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")

        return df

    # ---------- internals ----------
    def _build_search_queries(
        self, business: str, city: str, business_type: Optional[str]
    ) -> List[str]:
        base = [f"{business} {city}"]
        if business_type:
            base.append(f"{business} {business_type} {city}")

            if business_type in ["bank", "banque"]:
                base += [f"{business} agence {city}", f"{business} succursale {city}"]
            elif business_type in ["hotel", "hôtel"]:
                base += [f"{business} hotel {city}", f"{business} resort {city}"]
            elif business_type in ["restaurant"]:
                base += [f"{business} restaurant {city}", f"{business} cuisine {city}"]

        # A couple of broad variants help recall
        base += [f"{business} {city} Morocco", f"{business} near {city} Morocco"]
        return base

    def _normalize_places(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # SerpApi maps search returns results under 'local_results';
        # some older samples show 'places' so we fall back just in case.
        return data.get("local_results") or data.get("places") or []

    def _discover_for_city_and_business(
        self,
        business: str,
        city: str,
        business_type: Optional[str],
        center_ll: Optional[str],
        brand_pattern: Optional[Any],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen_keys = set()

        def add_place(p: Dict[str, Any], engine_tag: str):
            title = p.get("title") or ""
            if brand_pattern and not brand_pattern.search(title):
                return

            # canonical place_id only
            pid = p.get("place_id")
            if pid and not _is_canonical_place_id(pid):
                pid = None

            address = p.get("address") or ""
            key = pid or f"{title}-{address}"
            if key in seen_keys:
                return
            seen_keys.add(key)

            gps = p.get("gps_coordinates") or {}
            rating = p.get("rating") or p.get("reviews_rating") or p.get("stars")
            reviews = (
                p.get("reviews") or p.get("reviews_count") or p.get("number_of_reviews")
            )

            # numeric business id if exposed
            data_id = p.get("data_id") or p.get("data_cid") or p.get("cid")

            results.append(
                {
                    "business": business,
                    "business_type": business_type,
                    "name": title,
                    "address": address,
                    "city": city,
                    "lat": gps.get("latitude"),
                    "lng": gps.get("longitude"),
                    "place_id": pid,  # canonical only
                    "data_id": data_id,  # numeric CID-like
                    "type": p.get("type") or p.get("category"),
                    "rating": rating,
                    "reviews_count": reviews,
                    "phone": p.get("phone"),
                    "website": p.get("website"),
                    "_engine": engine_tag,
                }
            )

        queries = self._build_search_queries(business, city, business_type)

        # 1) Maps search WITH center, if provided (paginate)
        if center_ll:
            for q in queries:
                try:
                    token = None
                    while True:
                        data = self.client.search_google_maps(
                            query=q, ll=center_ll, next_page_token=token
                        )
                        places = self._normalize_places(data)
                        for p in places:
                            add_place(p, "maps_ll")
                        token = (data.get("serpapi_pagination") or {}).get(
                            "next_page_token"
                        )
                        if not token:
                            break
                        time.sleep(config.SERPAPI_CONFIG["delay_seconds"])
                    time.sleep(config.SERPAPI_CONFIG["delay_seconds"])
                except Exception as e:
                    logger.warning(f"Maps search failed (ll) for '{q}' in {city}: {e}")

        # 2) Maps search WITHOUT center if still empty
        if not results:
            for q in queries[:3]:
                try:
                    data = self.client.search_google_maps(query=q)
                    places = self._normalize_places(data)
                    for p in places:
                        add_place(p, "maps_noll")
                    time.sleep(config.SERPAPI_CONFIG["delay_seconds"])
                except Exception as e:
                    logger.warning(f"Maps search (noll) failed for '{q}': {e}")

        # 3) Optional: vary zooms around center for stubborn cases
        if not results and center_ll:
            try:
                lat, lng, _z = center_ll.strip("@").split(",")
                for z in ("10z", "12z", "14z", "16z"):
                    ll = f"@{lat},{lng},{z}"
                    data = self.client.search_google_maps(
                        query=f"{business} {city} Morocco", ll=ll
                    )
                    places = self._normalize_places(data)
                    for p in places:
                        add_place(p, f"maps_ll_{z}")
                    time.sleep(config.SERPAPI_CONFIG["delay_seconds"])
            except Exception as e:
                logger.debug(f"Zoom sweep skipped: {e}")

        return results

    def _deduplicate_results(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["_dedup_key"] = df.apply(
            lambda r: r.get("place_id")
            or r.get("data_id")
            or f"{r.get('name')}-{r.get('address')}",
            axis=1,
        )
        before = len(df)
        df = df.drop_duplicates(subset=["_dedup_key"], keep="first").drop(
            columns=["_dedup_key"]
        )
        logger.info(f"Deduplication: {before} → {len(df)} unique places")
        return df

    def _resolve_canonical_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df["canonical_place_id"] = None
            df["resolve_status"] = "empty"
            return df

        logger.info("Resolving canonical place_ids...")

        cache_path = config.OUTPUT_DIR / "place_id_cache.json"
        cache = utils.read_json(cache_path)
        canonical_ids, statuses = [], []

        progress = utils.create_progress_bar(len(df), desc="Resolving place_ids")

        for _, row in df.iterrows():
            pid = (row.get("place_id") or "") or ""
            did = row.get("data_id") or ""
            cache_key = did or pid

            if cache_key in cache:
                canonical_ids.append(cache[cache_key])
                statuses.append("cache")
                if progress:
                    progress.update(1)
                continue

            if _is_canonical_place_id(pid) or config.validate_place_id(pid):
                canonical_ids.append(pid)
                statuses.append("already_canonical")
                cache[cache_key] = pid
                if progress:
                    progress.update(1)
                continue

            try:
                resolved = self.client.get_place_details(
                    place_id=pid if pid else None, data_id=did if did else None
                )
                if resolved:
                    canonical_ids.append(resolved)
                    statuses.append("api_resolved")
                    cache[cache_key] = resolved
                else:
                    canonical_ids.append(None)
                    statuses.append("unresolved")
                time.sleep(config.SERPAPI_CONFIG["delay_seconds"])
            except Exception as e:
                logger.warning(f"Failed to resolve {cache_key}: {e}")
                canonical_ids.append(None)
                statuses.append("error")

            if progress:
                progress.update(1)

        if progress:
            progress.close()

        df["canonical_place_id"] = canonical_ids
        df["resolve_status"] = statuses
        utils.write_json(cache_path, cache)

        resolved_count = sum(1 for s in statuses if s != "unresolved")
        logger.info(f"Place ID resolution: {resolved_count}/{len(df)} resolved")
        return df


# =========================
# CLI Interface (unchanged)
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discover business locations on Google Maps")
    parser.add_argument("--businesses", required=True, help="Comma-separated business names")
    parser.add_argument("--business-type", help="Business type (e.g., 'bank', 'hotel', 'restaurant')")
    parser.add_argument("--cities", required=True, help="Comma-separated city names")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--map-centers", type=Path, help="JSON file with city center coordinates")
    parser.add_argument("--brand-filter", help="Regex pattern to filter by brand name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO))

    businesses = [b.strip() for b in args.businesses.split(",")]
    cities = [c.strip() for c in args.cities.split(",")]
    business_type = getattr(args, "business_type", None)

    map_centers = utils.read_json(args.map_centers) if args.map_centers else None

    engine = DiscoveryEngine(debug=args.debug)
    df = engine.discover_branches(
        businesses=businesses,
        cities=cities,
        business_type=business_type,
        map_centers=map_centers,
        brand_filter=args.brand_filter,
        output_path=args.output,
    )

    print("\n✅ Discovery complete!")
    print(f"   Found: {len(df)} locations")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
