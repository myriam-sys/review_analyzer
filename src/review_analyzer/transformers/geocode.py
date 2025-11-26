"""
Geocoding transformer - Assign Moroccan regions to reviews

Uses a GeoJSON of the 12 Moroccan regions.
Path: data/00_config/cities/regions.geojson

If regions file is missing, falls back to city→region mapping.

Moroccan Regions (12):
1. Tanger-Tétouan-Al Hoceïma
2. L'Oriental
3. Fès-Meknès
4. Rabat-Salé-Kénitra
5. Béni Mellal-Khénifra
6. Casablanca-Settat
7. Marrakech-Safi
8. Drâa-Tafilalet
9. Souss-Massa
10. Guelmim-Oued Noun
11. Laâyoune-Sakia El Hamra
12. Dakhla-Oued Ed-Dahab
"""

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

# Fallback city→region mapping (approximate)
CITY_REGION_MAPPING = {
    # Tanger-Tétouan-Al Hoceïma
    "tanger": "Tanger-Tétouan-Al Hoceïma",
    "tangier": "Tanger-Tétouan-Al Hoceïma",
    "tetouan": "Tanger-Tétouan-Al Hoceïma",
    "al hoceima": "Tanger-Tétouan-Al Hoceïma",
    
    # L'Oriental
    "oujda": "L'Oriental",
    "nador": "L'Oriental",
    
    # Fès-Meknès
    "fes": "Fès-Meknès",
    "fez": "Fès-Meknès",
    "meknes": "Fès-Meknès",
    
    # Rabat-Salé-Kénitra
    "rabat": "Rabat-Salé-Kénitra",
    "sale": "Rabat-Salé-Kénitra",
    "kenitra": "Rabat-Salé-Kénitra",
    "temara": "Rabat-Salé-Kénitra",
    
    # Béni Mellal-Khénifra
    "beni mellal": "Béni Mellal-Khénifra",
    "khenifra": "Béni Mellal-Khénifra",
    
    # Casablanca-Settat
    "casablanca": "Casablanca-Settat",
    "casa": "Casablanca-Settat",
    "settat": "Casablanca-Settat",
    "mohammedia": "Casablanca-Settat",
    "el jadida": "Casablanca-Settat",
    
    # Marrakech-Safi
    "marrakech": "Marrakech-Safi",
    "marrakesh": "Marrakech-Safi",
    "safi": "Marrakech-Safi",
    "essaouira": "Marrakech-Safi",
    
    # Drâa-Tafilalet
    "errachidia": "Drâa-Tafilalet",
    "ouarzazate": "Drâa-Tafilalet",
    
    # Souss-Massa
    "agadir": "Souss-Massa",
    "tiznit": "Souss-Massa",
    "taroudant": "Souss-Massa",
    
    # Guelmim-Oued Noun
    "guelmim": "Guelmim-Oued Noun",
    "tan-tan": "Guelmim-Oued Noun",
    
    # Laâyoune-Sakia El Hamra
    "laayoune": "Laâyoune-Sakia El Hamra",
    
    # Dakhla-Oued Ed-Dahab
    "dakhla": "Dakhla-Oued Ed-Dahab",
}


def _ensure_shapely():
    try:
        import shapely  # noqa: F401
        return True
    except Exception:
        return False

def load_regions_geojson(path: str | Path):
    import shapely.geometry as geom
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    regions = []
    for f in feats:
        props = f.get("properties", {})
        name = props.get("name") or props.get("NAME_1") or props.get("NAME")
        shape = geom.shape(f.get("geometry"))
        if name and not shape.is_empty:
            regions.append((name, shape, shape.centroid))
    return regions

def build_spatial_index(regions):
    # regions: list[(name, polygon, centroid)]
    from shapely.strtree import STRtree
    polys = [r[1] for r in regions]
    index = STRtree(polys)
    return index, polys

def assign_region_row(lat: float, lng: float, regions, index, polys):
    import shapely.geometry as geom
    pt = geom.Point(float(lng), float(lat))  # shapely uses (x=lon, y=lat)
    # candidate polygons via index
    candidates = index.query(pt)
    for poly in candidates:
        if poly.contains(pt):
            idx = polys.index(poly)
            return regions[idx][0]
    # fallback: nearest centroid
    best = None
    best_d = 1e18
    for name, poly, centroid in regions:
        d = pt.distance(centroid)
        if d < best_d:
            best, best_d = name, d
    return best

def add_region(df: pd.DataFrame, regions_path: str | Path) -> pd.DataFrame:
    """
    Adds a 'region' column to df using lat/lng coordinates.
    
    Args:
        df: DataFrame with 'lat' and 'lng' columns
        regions_path: Path to regions GeoJSON file
        
    Returns:
        DataFrame with added 'region' column
        
    Raises:
        RuntimeError: If shapely is not installed
        ValueError: If required columns are missing
        FileNotFoundError: If regions file doesn't exist
        
    Note:
        - Rows with missing/invalid coordinates get None for region
        - Uses spatial index for fast polygon lookup
        - Falls back to nearest centroid if point not in any polygon
    """
    if not _ensure_shapely():
        raise RuntimeError(
            "Shapely required for region assignment. "
            "Install: pip install shapely"
        )

    # Check for required columns (support alternative names)
    lat_col = None
    lng_col = None
    
    for col in ["lat", "latitude"]:
        if col in df.columns:
            lat_col = col
            break
    
    for col in ["lng", "longitude", "lon"]:
        if col in df.columns:
            lng_col = col
            break
    
    if not lat_col or not lng_col:
        raise ValueError(
            "DataFrame must have lat/lng columns. "
            f"Found columns: {list(df.columns)}"
        )

    # Validate regions file exists
    regions_path = Path(regions_path)
    if not regions_path.exists():
        raise FileNotFoundError(
            f"Regions file not found: {regions_path}"
        )

    # Load regions and build spatial index
    regions = load_regions_geojson(regions_path)
    if not regions:
        raise ValueError(
            f"No valid regions found in {regions_path}"
        )
    
    index, polys = build_spatial_index(regions)

    def _assign(row):
        """Assign region with error handling"""
        try:
            lat = row[lat_col]
            lng = row[lng_col]
            
            # Skip if coordinates are missing or invalid
            if pd.isna(lat) or pd.isna(lng):
                return None
            
            # Convert to float (handle string coordinates)
            lat = float(lat)
            lng = float(lng)
            
            # Basic validation (Morocco bounds approx)
            if not (-17 <= lng <= -1 and 21 <= lat <= 36):
                return None
            
            return assign_region_row(lat, lng, regions, index, polys)
        except (ValueError, TypeError, AttributeError):
            return None

    out = df.copy()
    out["region"] = out.apply(_assign, axis=1)
    
    # Fallback: use city name for unassigned regions
    if "city" in out.columns:
        unassigned = out["region"].isna()
        if unassigned.any():
            def map_city_to_region(city):
                if pd.isna(city):
                    return None
                city_norm = str(city).lower().strip()
                return CITY_REGION_MAPPING.get(city_norm)
            
            out.loc[unassigned, "region"] = (
                out.loc[unassigned, "city"].apply(map_city_to_region)
            )
    
    # Log statistics
    total = len(out)
    assigned = out["region"].notna().sum()
    pct = assigned / total * 100 if total > 0 else 0
    print(f"Region assignment: {assigned}/{total} rows ({pct:.1f}%)")
    
    return out


def add_region_by_city(df: pd.DataFrame, city_col: str = "city") -> pd.DataFrame:
    """
    Fallback method: assign region based on city name only
    
    Args:
        df: DataFrame with city column
        city_col: Name of city column (default: "city")
        
    Returns:
        DataFrame with added 'region' column
        
    Note:
        Uses CITY_REGION_MAPPING for common Moroccan cities
        More approximate than GeoJSON-based assignment
    """
    if city_col not in df.columns:
        raise ValueError(f"Column '{city_col}' not found in DataFrame")
    
    def map_city(city):
        if pd.isna(city):
            return None
        city_norm = str(city).lower().strip()
        return CITY_REGION_MAPPING.get(city_norm)
    
    out = df.copy()
    out["region"] = out[city_col].apply(map_city)
    
    assigned = out["region"].notna().sum()
    total = len(out)
    pct = assigned / total * 100 if total > 0 else 0
    print(f"Region assignment (city-based): {assigned}/{total} ({pct:.1f}%)")
    
    return out
