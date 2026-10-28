"""
ophalen.py - haalt de CBS-data op voor casus 7 (Nabijheid: wat is een voorziening waard?)

Dit script draai je EEN keer. Het haalt de tabellen op via de nieuwe open API van het CBS
(OData v4) en slaat ze op als CSV in de map data/. Het dashboard leest daarna alleen
die CSV's en roept zelf geen API aan.

De data komt in LANG formaat: een rij per buurt per meting, met codes in plaats van namen.
Het pivoteren en het toepassen van het codeboek gebeurt in de volgende stap.
"""

import json
import time
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

# ---------------------------------------------------------------
# Instellingen
# ---------------------------------------------------------------
BASIS_URL = "https://datasets.cbs.nl/odata/v1/CBS"
DATA_MAP = Path("data")
PEILDATUM = date.today().isoformat()      # bv. "2026-10-26", komt in de bestandsnaam

# Welke metingen (codes) we per tabel nodig hebben. Gevonden in het codeboek (MeasureCodes).
NABIJHEID_CODES = [
    "D000025",    # Afstand tot grote supermarkt (km)
    "A045790_3",  # Aantal grote supermarkten binnen 1 km
    "A045791_3",  # Aantal grote supermarkten binnen 3 km
    "A045792_5",  # Aantal grote supermarkten binnen 5 km
]
KERNCIJFERS_CODES = [
    "M001642",    # Gemiddelde WOZ-waarde van woningen (x 1 000 euro)
    "ST0001",     # Mate van stedelijkheid (code 1 t/m 5)
    "ST0003",     # Omgevingsadressendichtheid (per km2)
    "ZW10290",    # Percentage eengezinswoningen
    "GM000C",     # Gemeentenaam
    "R00003",     # Soort regio (Land, Gemeente, Wijk, Buurt)
]

# Buurtgrenzen voor de kaart: PDOK (kaartendienst van de overheid), CBS-buurten vereenvoudigd
PDOK_URL = "https://api.pdok.nl/cbs/gebiedsindelingen/ogc/v1/collections/buurt_gegeneraliseerd/items"
GRENZEN_JAAR = 2024                      # zelfde jaar als de CBS-cijfers


# ---------------------------------------------------------------
# Functies
# ---------------------------------------------------------------
def maak_filter(codes):
    """Maakt van een lijst codes een filter: Measure eq 'A' or Measure eq 'B' ..."""
    stukjes = [f"Measure eq '{code}'" for code in codes]
    return " or ".join(stukjes)


def haal_alles(url, parameters=None):
    """Haalt alle rijen op. De API geeft ze in pagina's; we volgen de link naar de volgende pagina."""
    stukken = []
    totaal = 0
    while url is not None:
        antwoord = requests.get(url, params=parameters, timeout=60)
        if antwoord.status_code != 200:
            raise RuntimeError(f"Fout {antwoord.status_code} bij {antwoord.url}")
        inhoud = antwoord.json()
        stukken.append(pd.DataFrame(inhoud["value"]))
        totaal = totaal + len(inhoud["value"])
        print(f"   {totaal} rijen binnen")

        url = inhoud.get("@odata.nextLink")   # None als er geen volgende pagina is
        parameters = None                     # de volgende link bevat de filter al
        time.sleep(1)                         # even wachten, anders weigert de API ons
    return pd.concat(stukken, ignore_index=True)


def download(tabel_id, onderdeel, naam, parameters=None):
    """Haalt een onderdeel van een tabel op en slaat het op, tenzij het bestand er al is."""
    bestaande = list(DATA_MAP.glob(f"{naam}_{tabel_id}_*.csv"))
    if len(bestaande) > 0:
        print(f"Overgeslagen, staat er al: {bestaande[0]}")
        return

    print(f"Ophalen: {tabel_id} / {onderdeel}")
    url = f"{BASIS_URL}/{tabel_id}/{onderdeel}"
    df = haal_alles(url, parameters)

    bestand = DATA_MAP / f"{naam}_{tabel_id}_{PEILDATUM}.csv"
    df.to_csv(bestand, index=False)
    print(f"   Opgeslagen: {bestand}  ({df.shape[0]} rijen, {df.shape[1]} kolommen)")


def download_buurtgrenzen():
    """Haalt de buurtgrenzen (vlakken op de kaart) op bij PDOK en slaat ze op als GeoJSON."""
    bestaande = list(DATA_MAP.glob(f"buurtgrenzen_{GRENZEN_JAAR}_*.geojson"))
    if len(bestaande) > 0:
        print(f"Overgeslagen, staat er al: {bestaande[0]}")
        return

    print(f"Ophalen: buurtgrenzen {GRENZEN_JAAR} (PDOK)")
    parameters = {"f": "json", "jaarcode": GRENZEN_JAAR, "limit": 1000}
    buurten = []
    while True:
        antwoord = requests.get(PDOK_URL, params=parameters, timeout=60)
        if antwoord.status_code != 200:
            raise RuntimeError(f"Fout {antwoord.status_code} bij {antwoord.url}")
        inhoud = antwoord.json()
        buurten.extend(inhoud["features"])
        print(f"   {len(buurten)} buurten binnen")

        # Is er een volgende pagina? Dan staat er een link met rel = "next" en daarin
        # een 'cursor' (een soort bladwijzer). Die geven we mee bij het volgende verzoek.
        volgende = [link["href"] for link in inhoud["links"] if link["rel"] == "next"]
        if len(volgende) == 0:
            break
        parameters["cursor"] = parse_qs(urlparse(volgende[0]).query)["cursor"][0]
        time.sleep(1)

    # Voor de zekerheid: alleen buurten uit het goede jaar
    buurten = [b for b in buurten if b["properties"]["jaarcode"] == GRENZEN_JAAR]

    bestand = DATA_MAP / f"buurtgrenzen_{GRENZEN_JAAR}_{PEILDATUM}.geojson"
    with open(bestand, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": buurten}, f)
    print(f"   Opgeslagen: {bestand}  ({len(buurten)} buurten)")


# ---------------------------------------------------------------
# Hoofdprogramma
# ---------------------------------------------------------------
DATA_MAP.mkdir(exist_ok=True)

# 1. De data zelf (lang formaat), alleen de metingen die we nodig hebben
download("86134NED", "Observations", "nabijheid",
         {"$filter": maak_filter(NABIJHEID_CODES)})
download("85984NED", "Observations", "kerncijfers",
         {"$filter": maak_filter(KERNCIJFERS_CODES)})

# 2. De codeboeken: wat betekent elke code?
download("86134NED", "MeasureCodes", "nabijheid_codeboek")
download("85984NED", "MeasureCodes", "kerncijfers_codeboek")
download("85984NED", "WijkenEnBuurtenCodes", "regio_codeboek")   # regiocode -> naam

# 3. De buurtgrenzen voor de kaart
download_buurtgrenzen()

print("Klaar!")