import requests

BASE_URL = "https://api.inaturalist.org/v1"
AMBROSIA_TAXON_ID = 75712
TIMEOUT_S = 5


def get_species_info() -> dict:
    try:
        r = requests.get(
            f"{BASE_URL}/taxa/{AMBROSIA_TAXON_ID}",
            params={"locale": "fr"},
            timeout=TIMEOUT_S
        )
        r.raise_for_status()
        d = r.json()["results"][0]
        return {
            "nom_commun": d.get("preferred_common_name", "Herbe à poux"),
            "nom_scientifique": d["name"],
            "description": (d.get("wikipedia_summary") or "")[:400],
            "nb_observations": d.get("observations_count", 0),
            "photo_url": d["default_photo"]["medium_url"]
                         if d.get("default_photo") else None,
        }
    except Exception as e:
        print(f"[iNaturalist] erreur: {e}")
        return {}


def get_nearby_observations(lat: float, lng: float, rayon_km: int = 10) -> list:
    try:
        r = requests.get(
            f"{BASE_URL}/observations",
            params={
                "taxon_id": AMBROSIA_TAXON_ID,
                "lat": lat,
                "lng": lng,
                "radius": rayon_km,
                "quality_grade": "research",
                "per_page": 10,
                "order_by": "observed_on",
                "order": "desc",
            },
            timeout=TIMEOUT_S
        )
        r.raise_for_status()
        obs_list = []
        for obs in r.json().get("results", []):
            coords = obs.get("geojson", {}).get("coordinates", [None, None])
            obs_list.append({
                "id": obs.get("id"),
                "date": obs.get("observed_on"),
                "lieu": obs.get("place_guess"),
                "latitude": coords[1] if coords[0] else None,
                "longitude": coords[0] if coords[0] else None,
            })
        return obs_list
    except Exception as e:
        print(f"[iNaturalist] erreur: {e}")
        return []
