"""Approximate home-arena coordinates (lat, lon) per franchise, for travel distance.
City-level precision is enough: the feature is distance between consecutive game sites."""

import math

ARENAS: dict[int, tuple[float, float]] = {
    1610612737: (33.757, -84.396),  # ATL
    1610612738: (42.366, -71.062),  # BOS
    1610612739: (41.496, -81.688),  # CLE
    1610612740: (29.949, -90.082),  # NOP
    1610612741: (41.881, -87.674),  # CHI
    1610612742: (32.790, -96.810),  # DAL
    1610612743: (39.749, -105.008),  # DEN
    1610612744: (37.768, -122.388),  # GSW
    1610612745: (29.751, -95.362),  # HOU
    1610612746: (34.043, -118.267),  # LAC
    1610612747: (34.043, -118.267),  # LAL
    1610612748: (25.781, -80.188),  # MIA
    1610612749: (43.045, -87.917),  # MIL
    1610612750: (44.979, -93.276),  # MIN
    1610612751: (40.683, -73.975),  # BKN
    1610612752: (40.750, -73.993),  # NYK
    1610612753: (28.539, -81.384),  # ORL
    1610612754: (39.764, -86.155),  # IND
    1610612755: (39.901, -75.172),  # PHI
    1610612756: (33.446, -112.071),  # PHX
    1610612757: (45.532, -122.667),  # POR
    1610612758: (38.580, -121.500),  # SAC
    1610612759: (29.427, -98.437),  # SAS
    1610612760: (35.463, -97.515),  # OKC
    1610612761: (43.643, -79.379),  # TOR
    1610612762: (40.768, -111.901),  # UTA
    1610612763: (35.138, -90.051),  # MEM
    1610612764: (38.898, -77.021),  # WAS
    1610612765: (42.341, -83.055),  # DET
    1610612766: (35.225, -80.839),  # CHA
}

SEATTLE = (47.622, -122.354)
OKLAHOMA_CITY = (35.463, -97.515)
ORLANDO = (28.539, -81.384)

# (team_id, season) -> site, for seasons a franchise played its home games elsewhere.
RELOCATIONS: dict[tuple[int, int], tuple[float, float]] = {
    **{(1610612760, season): SEATTLE for season in range(2004, 2008)},  # SuperSonics
    (1610612740, 2005): OKLAHOMA_CITY,  # Hornets after Hurricane Katrina
    (1610612740, 2006): OKLAHOMA_CITY,
}


def home_site(team_id: int, season: int) -> tuple[float, float]:
    return RELOCATIONS.get((team_id, season), ARENAS[team_id])


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(h))
