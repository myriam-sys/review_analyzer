"""
Geospatial helpers for city centers and aliases.
Builds and loads map centers and alias index from OSM NDJSON.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional

from . import config, utils


def _normalize_name(value: str) -> str:
    if not isinstance(value, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", value)
    ascii_fold = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return " ".join(ascii_fold.lower().strip().split())


def _zoom_for_population(population: Optional[int]) -> str:
    if not population:
        return "14z"
    if population >= 800_000:
        return "11z"
    if population >= 200_000:
        return "12z"
    if population >= 50_000:
        return "13z"
    return "14z"


def _iter_osm_entries(ndjson_path: Path) -> Iterable[dict]:
    if not ndjson_path.exists():
        return []
    with ndjson_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _extract_aliases(entry: dict) -> Tuple[str, Iterable[str]]:
    """Returns (canonical_name, aliases_iterable)."""
    aliases = set()
    base = entry.get("name")
    if base:
        aliases.add(base)
    other = entry.get("other_names") or {}
    for _, v in (other.items() if isinstance(other, dict) else []):
        if isinstance(v, str) and v:
            aliases.add(v)
    addr_city = (entry.get("address") or {}).get("city")
    if isinstance(addr_city, str) and addr_city:
        aliases.add(addr_city)

    # Choose canonical by preference: French, English, otherwise base
    preferred = base or ""
    if isinstance(other, dict):
        if isinstance(other.get("name:fr"), str):
            preferred = other["name:fr"]
        elif isinstance(other.get("name:en"), str):
            preferred = other["name:en"]
    if not preferred and aliases:
        preferred = sorted(aliases)[0]

    return preferred, sorted(a for a in aliases if a)


def build_from_osm(
    osm_ndjson: Path,
    include_towns: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build (map_centers, alias_index) from OSM NDJSON.
    map_centers: city -> '@lat,lng,zoom'
    alias_index: normalized(alias) -> city
    """
    centers: Dict[str, str] = {}
    aliases: Dict[str, str] = {}

    valid_types = {"city"}
    if include_towns:
        valid_types.add("town")

    for e in _iter_osm_entries(osm_ndjson):
        addr = e.get("address") or {}
        if addr.get("country_code") != "ma":
            continue
        typ = e.get("type") or e.get("inferred_type")
        if typ not in valid_types:
            continue
        loc = e.get("location") or []
        if not (isinstance(loc, list) and len(loc) == 2):
            continue
        lng, lat = float(loc[0]), float(loc[1])
        pop = e.get("population")
        z = _zoom_for_population(pop)

        canonical, all_aliases = _extract_aliases(e)
        if not canonical:
            continue
        centers[canonical] = f"@{lat:.6f},{lng:.6f},{z}"
        for a in all_aliases:
            aliases[_normalize_name(a)] = canonical

    return centers, aliases


def ensure_osm_derived_files(
    include_towns: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Ensure map_centers.json and city_aliases.json exist (build from OSM if possible).
    Returns (map_centers, aliases) in-memory dicts.
    """
    centers_path = config.DEFAULT_MAP_CENTERS_FILE
    aliases_path = config.CITY_ALIASES_FILE
    osm_path = config.OSM_CITY_FILE

    # Try to load existing
    centers = utils.read_json(centers_path)
    aliases = utils.read_json(aliases_path)
    if centers and aliases:
        return centers, aliases

    # Build from OSM if available
    if osm_path.exists():
        centers, aliases = build_from_osm(osm_path, include_towns=include_towns)
        if centers:
            utils.write_json(centers_path, centers)
        if aliases:
            utils.write_json(aliases_path, aliases)
        return centers or {}, aliases or {}

    # Fall back to hardcoded defaults
    return dict(config.DEFAULT_MAP_CENTERS), {}


def load_map_centers() -> Dict[str, str]:
    centers, _ = ensure_osm_derived_files()
    if centers:
        return centers
    return dict(config.DEFAULT_MAP_CENTERS)


def resolve_city_name(input_city: str) -> str:
    """
    Resolve an input city to canonical key using alias index; fall back to input.
    """
    if not input_city:
        return input_city
    _, aliases = ensure_osm_derived_files()
    if not aliases:
        return input_city
    norm = _normalize_name(input_city)
    return aliases.get(norm, input_city)


