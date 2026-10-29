"""
dashboard.py - Casus 7: Nabijheid, wat is een supermarkt waard?

Starten met:   streamlit run dashboard.py

Dit dashboard leest alleen de bestanden in de map data/ (gemaakt door ophalen.py).
Er wordt hier geen API aangeroepen.

Opbouw van dit bestand:
    DEEL 1 - DATA        : inlezen, opschonen, pivoteren, samenvoegen, kolommen afleiden
    DEEL 2 - STATISTIEK  : verschil in WOZ met marge, gecorrigeerd voor woningtype en gemeente
    DEEL 3 - HULPFUNCTIES: getallen netjes opschrijven, grafieken maken
    DEEL 4 - STREAMLIT   : wat je op het scherm ziet (titel, filters, KPI's, tabs)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import streamlit as st

st.set_page_config(page_title="Wat is een supermarkt waard?", layout="wide")


# =====================================================================
# DEEL 1 - DATA
# =====================================================================
DATA_MAP = Path("data")

# De bestanden die ophalen.py heeft gemaakt (de datum = de dag van ophalen)
BESTAND_NABIJHEID = DATA_MAP / "nabijheid_86134NED_2026-10-26.csv"
BESTAND_KERNCIJFERS = DATA_MAP / "kerncijfers_85984NED_2026-10-26.csv"
BESTAND_REGIO_CODEBOEK = DATA_MAP / "regio_codeboek_85984NED_2026-10-26.csv"
BESTAND_GRENZEN = DATA_MAP / "buurtgrenzen_2024_2026-10-26.geojson"

# Het codeboek: welke CBS-code wordt welke kolomnaam
CODES_NABIJHEID = {
    "D000025":   "afstand_supermarkt_km",     # gem. afstand tot grote supermarkt (km)
    "A045790_3": "supermarkten_binnen_1km",   # aantal grote supermarkten binnen 1 km
    "A045791_3": "supermarkten_binnen_3km",   # aantal grote supermarkten binnen 3 km
    "A045792_5": "supermarkten_binnen_5km",   # aantal grote supermarkten binnen 5 km
}
CODES_KERNCIJFERS = {
    "M001642": "woz_x1000",            # gem. WOZ-waarde woningen (x 1 000 euro)
    "ST0001":  "stedelijkheid_code",   # mate van stedelijkheid (1 t/m 5)
    "ST0003":  "adressen_per_km2",     # omgevingsadressendichtheid
    "ZW10290": "pct_eengezinswoning",  # percentage eengezinswoningen (%)
}
CODE_GEMEENTENAAM = "GM000C"

# Het codeboek voor stedelijkheid (uit de CBS-beschrijving van ST0001)
STEDELIJKHEID_NAMEN = {
    1: "1 · zeer sterk stedelijk",
    2: "2 · sterk stedelijk",
    3: "3 · matig stedelijk",
    4: "4 · weinig stedelijk",
    5: "5 · niet stedelijk",
}
# Onze eigen indeling in twee groepen (keuze die je moet kunnen verdedigen).
# Klasse 3 (matig stedelijk) gedroeg zich in de analyse hetzelfde als de stad.
GEBIED = {1: "Stad", 2: "Stad", 3: "Stad", 4: "Platteland", 5: "Platteland"}

# Afstandsbanden: grenzen in km en de namen van de banden
AFSTANDSBANDEN = [0, 0.5, 1, 2, 5, 100]
BAND_NAMEN = ["0 - 0,5 km", "0,5 - 1 km", "1 - 2 km", "2 - 5 km", "meer dan 5 km"]

# Aantal grote supermarkten binnen 1 km. Het CBS geeft een gemiddelde per buurt (bv. 1,4),
# dat ronden we af naar 0, 1, 2 of 3 of meer: [0 - 0,5) = 0, [0,5 - 1,5) = 1, enz.
AANTAL_GRENZEN = [0, 0.5, 1.5, 2.5, 1000]
AANTAL_NAMEN = ["0", "1", "2", "3 of meer"]

# Onder dit aantal buurten vinden we een vergelijking niet betrouwbaar genoeg
MINIMAAL_BUURTEN = 100
# Per gemeente zijn er veel minder buurten; daar is de grens lager (onderste grafiek van tab 2)
MINIMAAL_BUURTEN_GEMEENTE = 10
AANTAL_GEMEENTEN = 10          # hoeveel gemeenten de onderste grafiek van tab 2 laat zien

# Vaste kleuren, overal in het dashboard hetzelfde
KLEUREN = {"Stad": "#1f5f8b", "Platteland": "#3a8a5c", "Alle buurten": "#8c8c8c"}
KLEUR_VERSCHIL = "#6a4c93"     # paars: het effect van een supermarkt op loopafstand
KLEUR_VERSCHIL_LICHT = "#d4c6e6"  # lichtpaars: de andere kant van hetzelfde
# Gemeentegrafiek in tab 2: lichtpaars = alle buurten, donkerpaars = bij gelijk woningtype
KLEUREN_VERGELIJKING = {"Alle buurten vergeleken": KLEUR_VERSCHIL_LICHT,
                        "Bij gelijk woningtype": KLEUR_VERSCHIL}


def maak_breed(lang, codes):
    """Pivoteert van lang (rij per buurt per meting) naar breed (rij per buurt)."""
    lang = lang[lang["Measure"].isin(codes.keys())]
    breed = lang.pivot(index="buurtcode", columns="Measure", values="Value")
    breed = breed.rename(columns=codes)         # codes -> leesbare namen
    breed.columns.name = None
    return breed.reset_index()


@st.cache_data
def laad_data():
    """Leest de ruwe CBS-data in en maakt er een schone tabel van: een rij per buurt.
    Geeft ook 'stappen' terug: tussenresultaten voor het tabblad Over de data."""
    stappen = {}

    # --- Stap 1: inlezen (lang formaat) ---
    nabij_lang = pd.read_csv(BESTAND_NABIJHEID)
    kern_lang = pd.read_csv(BESTAND_KERNCIJFERS)
    regio_codeboek = pd.read_csv(BESTAND_REGIO_CODEBOEK)
    stappen["voorbeeld_lang"] = nabij_lang[["WijkenEnBuurten", "Measure", "Value"]].head(8)

    # --- Stap 2: codes opschonen en alleen buurten houden ---
    # Het CBS vult codes aan met spaties ("BU03630000  "). Die halen we weg,
    # anders koppelt "BU03630000  " niet aan "BU03630000".
    nabij_lang["buurtcode"] = nabij_lang["WijkenEnBuurten"].str.strip()
    kern_lang["buurtcode"] = kern_lang["WijkenEnBuurten"].str.strip()

    # Buurtcodes beginnen met BU (gemeenten met GM, wijken met WK, Nederland met NL)
    nabij_lang = nabij_lang[nabij_lang["buurtcode"].str.startswith("BU")]
    kern_lang = kern_lang[kern_lang["buurtcode"].str.startswith("BU")]
    stappen["buurten_nabijheid"] = nabij_lang["buurtcode"].nunique()
    stappen["buurten_kerncijfers"] = kern_lang["buurtcode"].nunique()

    # --- Stap 3: pivoteren (lang -> breed) en codeboek toepassen ---
    nabij = maak_breed(nabij_lang, CODES_NABIJHEID)
    kern = maak_breed(kern_lang, CODES_KERNCIJFERS)

    # De gemeentenaam is tekst, dus die staat in StringValue in plaats van Value
    gemeenten = kern_lang[kern_lang["Measure"] == CODE_GEMEENTENAAM]
    gemeenten = gemeenten[["buurtcode", "StringValue"]].rename(columns={"StringValue": "gemeente"})
    gemeenten["gemeente"] = gemeenten["gemeente"].str.strip()
    kern = kern.merge(gemeenten, on="buurtcode", how="left", validate="one_to_one")
    stappen["voorbeeld_breed"] = nabij.head(5)

    # --- Stap 4: samenvoegen op buurtcode ---
    # Eerst controleren: welke buurten zitten in beide tabellen, en welke maar in een?
    controle = nabij.merge(kern, on="buurtcode", how="outer", indicator=True)
    stappen["alleen_nabijheid"] = (controle["_merge"] == "left_only").sum()
    stappen["alleen_kerncijfers"] = (controle["_merge"] == "right_only").sum()

    # Dan echt samenvoegen. validate="one_to_one" geeft een foutmelding als een
    # buurtcode dubbel voorkomt (dan zouden rijen stilletjes verdubbelen).
    df = nabij.merge(kern, on="buurtcode", how="inner", validate="one_to_one")

    # Buurtnamen erbij uit het regio-codeboek
    namen = regio_codeboek[["Identifier", "Title"]].rename(
        columns={"Identifier": "buurtcode", "Title": "buurtnaam"})
    namen["buurtcode"] = namen["buurtcode"].str.strip()
    df = df.merge(namen, on="buurtcode", how="left", validate="one_to_one")
    stappen["na_samenvoegen"] = len(df)

    # --- Stap 5: ontbrekende waarden ---
    df["stedelijkheid"] = df["stedelijkheid_code"].map(STEDELIJKHEID_NAMEN)
    df["woz_ontbreekt"] = df["woz_x1000"].isna()
    stappen["missend_per_stedelijkheid"] = (
        df.groupby("stedelijkheid")["woz_ontbreekt"].mean() * 100
    ).reset_index(name="procent_zonder_woz")

    # Keuze: buurten zonder WOZ, afstand, stedelijkheid of woningtype laten we weg.
    # Invullen (bv. met het gemiddelde) zou een WOZ-waarde verzinnen die het CBS bewust niet geeft.
    df = df.dropna(subset=["afstand_supermarkt_km", "woz_x1000", "stedelijkheid_code",
                           "pct_eengezinswoning"])
    stappen["eindtotaal"] = len(df)

    # --- Stap 6: afgeleide kolommen ---
    df["woz_euro"] = df["woz_x1000"] * 1000
    df["gebied"] = df["stedelijkheid_code"].map(GEBIED)
    df["afstandsband"] = pd.cut(df["afstand_supermarkt_km"], bins=AFSTANDSBANDEN,
                                labels=BAND_NAMEN, include_lowest=True)
    df["aantal_supermarkten"] = pd.cut(df["supermarkten_binnen_1km"], bins=AANTAL_GRENZEN,
                                       labels=AANTAL_NAMEN, right=False)

    return df, stappen


@st.cache_resource
def laad_grenzen():
    """Leest de buurtgrenzen (GeoJSON van PDOK) voor de kaart.
    cache_resource in plaats van cache_data: het bestand is groot, en cache_data zou het bij
    elke klik opnieuw kopiëren. Dat maakte het dashboard traag."""
    with open(BESTAND_GRENZEN, encoding="utf-8") as f:
        return json.load(f)


def alle_hoekpunten(geometrie):
    """Alle hoekpunten (lon, lat) van de buitenrand van een buurt, als numpy-array.
    Een buurt is een Polygon (een vlak) of een MultiPolygon (meerdere losse vlakken)."""
    if geometrie["type"] == "Polygon":
        vlakken = [geometrie["coordinates"]]
    else:
        vlakken = geometrie["coordinates"]
    return np.concatenate([np.array(vlak[0]) for vlak in vlakken])   # vlak[0] = buitenrand


@st.cache_resource
def maak_kaartdata():
    """Zet de buurtgrenzen klaar voor de kaart:
    - vlakken: woordenboek buurtcode -> vlak (voor de kaart van één gemeente)
    - middelpunten: tabel met het middelpunt van elke buurt (voor de stippen van heel NL)"""
    vlakken = {}
    middelpunten = []
    for buurt in laad_grenzen()["features"]:
        code = buurt["properties"]["statcode"].strip()
        vlakken[code] = buurt
        punten = alle_hoekpunten(buurt["geometry"])
        middelpunten.append({"buurtcode": code,
                             "lon": punten[:, 0].mean(), "lat": punten[:, 1].mean()})
    return vlakken, pd.DataFrame(middelpunten)


def voeg_loopafstand_toe(df, loopafstand_km):
    """Afgeleide kolom: ligt er een grote supermarkt op loopafstand? (True/False)
    Staat buiten laad_data, zodat de grens met de slider instelbaar is."""
    df = df.copy()
    df["supermarkt_op_loopafstand"] = df["afstand_supermarkt_km"] <= loopafstand_km
    return df


# =====================================================================
# DEEL 2 - STATISTIEK
# =====================================================================
def bereken_verschil(df, groep_kolom, ja_nee_kolom, uitkomst_kolom):
    """Vergelijkt per groep de buurten met 'ja' tegen de buurten met 'nee'.
    Geeft per groep het verschil in de uitkomst in procenten, met een 95%-marge."""
    rijen = []
    for groep, deel in df.groupby(groep_kolom):
        ja = deel[deel[ja_nee_kolom]]
        nee = deel[~deel[ja_nee_kolom]]
        if len(ja) < 2 or len(nee) < 2:          # niets te vergelijken in deze groep
            continue

        # We rekenen op logschaal: dan wordt het verschil vanzelf een percentage.
        # (WOZ-waarden zijn scheef verdeeld: een paar hele dure buurten.)
        log_ja = np.log(ja[uitkomst_kolom])
        log_nee = np.log(nee[uitkomst_kolom])
        verschil = log_ja.mean() - log_nee.mean()

        # Standaardfout: hoe onzeker is dat verschil? Kleiner bij meer buurten.
        standaardfout = np.sqrt(log_ja.var() / len(ja) + log_nee.var() / len(nee))

        # 95%-marge: het verschil plus en min 1,96 keer de standaardfout
        laag = verschil - 1.96 * standaardfout
        hoog = verschil + 1.96 * standaardfout

        rijen.append({
            groep_kolom: groep,
            "buurten_ja": len(ja),
            "buurten_nee": len(nee),
            "verschil_pct": (np.exp(verschil) - 1) * 100,    # terug van log naar %
            "marge_laag_pct": (np.exp(laag) - 1) * 100,
            "marge_hoog_pct": (np.exp(hoog) - 1) * 100,
        })
    return pd.DataFrame(rijen)


def bereken_verschil_gecorrigeerd(df, groep_kolom, ja_nee_kolom, uitkomst_kolom, controles):
    """Zelfde vraag als bereken_verschil, maar nu met een regressie die de kolommen in
    'controles' constant houdt. Bijvoorbeeld: wat is het verschil tussen ja en nee,
    bij buurten met hetzelfde percentage eengezinswoningen, in dezelfde gemeente?"""
    rijen = []
    for groep, deel in df.groupby(groep_kolom):
        deel = deel.copy()
        deel["log_uitkomst"] = np.log(deel[uitkomst_kolom])   # logschaal -> verschil in %
        deel["ja"] = deel[ja_nee_kolom].astype(int)           # True/False -> 1/0
        if deel["ja"].sum() < 2 or (deel["ja"] == 0).sum() < 2:
            continue

        # De regressie: log(WOZ) verklaren uit 'ja' EN de controlevariabelen.
        # Het getal bij 'ja' is dan het verschil bij gelijke controlevariabelen.
        formule = "log_uitkomst ~ ja + " + " + ".join(controles)
        model = smf.ols(formule, data=deel).fit()
        verschil = model.params["ja"]
        laag, hoog = model.conf_int().loc["ja"]               # 95%-marge

        rijen.append({
            groep_kolom: groep,
            "buurten_ja": int(deel["ja"].sum()),
            "buurten_nee": int((deel["ja"] == 0).sum()),
            "verschil_pct": (np.exp(verschil) - 1) * 100,
            "marge_laag_pct": (np.exp(laag) - 1) * 100,
            "marge_hoog_pct": (np.exp(hoog) - 1) * 100,
            "r_kwadraat": model.rsquared,      # hoeveel van de WOZ-verschillen het model verklaart
        })
    return pd.DataFrame(rijen)


@st.cache_data
def bereken_opbouw(df):
    """Het verschil in WOZ (ja tegen nee), stap voor stap beter vergelijkbaar gemaakt.
    Elke stap houdt een ding meer constant. Dit is de basis van tab 2 en de KPI's."""
    alle = df.copy()
    alle["alle_buurten"] = "Alle buurten"

    stap0 = bereken_verschil(alle, "alle_buurten", "supermarkt_op_loopafstand", "woz_x1000")
    stap0 = stap0.rename(columns={"alle_buurten": "gebied"})
    stap1 = bereken_verschil(df, "gebied", "supermarkt_op_loopafstand", "woz_x1000")
    stap2 = bereken_verschil_gecorrigeerd(df, "gebied", "supermarkt_op_loopafstand",
                                          "woz_x1000", ["pct_eengezinswoning"])
    stap3 = bereken_verschil_gecorrigeerd(df, "gebied", "supermarkt_op_loopafstand",
                                          "woz_x1000", ["pct_eengezinswoning", "C(gemeente)"])

    stap0["stap"] = "1 · Niets"
    stap1["stap"] = "2 · + stad / platteland"
    stap2["stap"] = "3 · + woningtype"
    stap3["stap"] = "4 · + gemeente"

    opbouw = pd.concat([stap0, stap1, stap2, stap3], ignore_index=True)
    # Als er nergens iets te vergelijken valt (bv. alleen stad bij 4 km), zijn de tabellen leeg.
    # Dan toch alle kolommen aanmaken, zodat de rest van het dashboard niet vastloopt.
    opbouw = opbouw.reindex(columns=["gebied", "stap", "buurten_ja", "buurten_nee",
                                     "verschil_pct", "marge_laag_pct", "marge_hoog_pct",
                                     "r_kwadraat"])
    # Betrouwbaar als beide groepen (ja en nee) minstens MINIMAAL_BUURTEN buurten hebben
    opbouw["betrouwbaar"] = opbouw[["buurten_ja", "buurten_nee"]].min(axis=1) >= MINIMAAL_BUURTEN
    return opbouw


@st.cache_data
def bereken_per_groep(df, groep_kolom, groepen):
    """Per gebied: hoeveel hoger of lager is de WOZ in elke groep dan in de eerste groep
    (het vergelijkingspunt), in dezelfde gemeente en bij hetzelfde aandeel eengezinswoningen?
    Gebruikt in tab 3: groep_kolom = 'afstandsband' of 'aantal_supermarkten'."""
    rijen = []
    for gebied, deel in df.groupby("gebied"):
        deel = deel.copy()
        deel["log_woz"] = np.log(deel["woz_x1000"])

        # Alleen groepen met genoeg buurten doen mee
        aantallen = deel[groep_kolom].value_counts()
        mee = [g for g in groepen if aantallen.get(g, 0) >= MINIMAAL_BUURTEN]
        if groepen[0] not in mee or len(mee) < 2:
            continue                                # geen vergelijkingspunt
        deel = deel[deel[groep_kolom].isin(mee)].copy()

        # Een 0/1-kolom per groep (behalve de eerste: dat is het vergelijkingspunt)
        kolommen = []
        for nummer, groep in enumerate(mee[1:], start=1):
            kolom = f"groep_{nummer}"
            deel[kolom] = (deel[groep_kolom] == groep).astype(int)
            kolommen.append(kolom)

        formule = "log_woz ~ " + " + ".join(kolommen) + " + pct_eengezinswoning + C(gemeente)"
        model = smf.ols(formule, data=deel).fit()
        marges = model.conf_int()

        rijen.append({"gebied": gebied, groep_kolom: mee[0], "buurten": aantallen[mee[0]],
                      "verschil_pct": 0.0, "marge_laag_pct": 0.0, "marge_hoog_pct": 0.0,
                      "vergelijkingspunt": True})
        for kolom, groep in zip(kolommen, mee[1:]):
            rijen.append({
                "gebied": gebied,
                groep_kolom: groep,
                "vergelijkingspunt": False,
                "buurten": aantallen[groep],
                "verschil_pct": (np.exp(model.params[kolom]) - 1) * 100,
                "marge_laag_pct": (np.exp(marges.loc[kolom, 0]) - 1) * 100,
                "marge_hoog_pct": (np.exp(marges.loc[kolom, 1]) - 1) * 100,
            })
    return pd.DataFrame(rijen)


# =====================================================================
# DEEL 3 - HULPFUNCTIES
# =====================================================================
def getal(waarde):
    """12773 -> '12.773' (Nederlandse notatie)."""
    return f"{waarde:,.0f}".replace(",", ".")


def procent(waarde):
    """-10.64 -> '-10,6%'."""
    return f"{waarde:+.1f}%".replace(".", ",")


def hoger_of_lager(waarde):
    """-10.64 -> '10,6% lager', 4.2 -> '4,2% hoger'."""
    richting = "lager" if waarde < 0 else "hoger"
    return f"{abs(waarde):.1f}% {richting}".replace(".", ",")


def marge_tekst(laag, hoog):
    """De 95%-marge in woorden: (-12.3, -8.9) -> 'marge 8,9 tot 12,3% lager'."""
    if hoog < 0:
        tekst = f"marge {abs(hoog):.1f} tot {abs(laag):.1f}% lager"
    elif laag > 0:
        tekst = f"marge {laag:.1f} tot {hoog:.1f}% hoger"
    else:
        tekst = f"marge {laag:+.1f}% tot {hoog:+.1f}% (kan ook 0 zijn)"
    return tekst.replace(".", ",")


def puntengrafiek(tabel, x_kolom, x_volgorde, titel, ondertitel, x_titel):
    """Punten met een streepje (de 95%-marge), per gebied een eigen kleur.
    Stad en platteland staan iets naast elkaar, zodat de streepjes niet overlappen."""
    tabel = tabel.copy()
    verschuiving = {"Stad": -0.12, "Platteland": 0.12, "Alle buurten": 0.0}
    tabel["x"] = tabel[x_kolom].map({naam: i for i, naam in enumerate(x_volgorde)})
    tabel["x"] = tabel["x"] + tabel["gebied"].map(verschuiving)
    tabel["boven"] = tabel["marge_hoog_pct"] - tabel["verschil_pct"]
    tabel["onder"] = tabel["verschil_pct"] - tabel["marge_laag_pct"]
    tabel["label"] = tabel["verschil_pct"].apply(procent)
    tabel["marge"] = [marge_tekst(l, h) for l, h in zip(tabel["marge_laag_pct"], tabel["marge_hoog_pct"])]
    # Het vergelijkingspunt (tab 3) is geen uitkomst: geen getal bij het punt,
    # maar één keer "(vergelijkingspunt)" onder de as
    as_teksten = list(x_volgorde)
    if "vergelijkingspunt" in tabel.columns:
        tabel.loc[tabel["vergelijkingspunt"], "label"] = ""
        tabel.loc[tabel["vergelijkingspunt"], "marge"] = "–"
        as_teksten[0] = f"{x_volgorde[0]}<br>(vergelijkingspunt)"

    fig = px.scatter(
        tabel, x="x", y="verschil_pct", color="gebied", text="label",
        error_y="boven", error_y_minus="onder", color_discrete_map=KLEUREN,
        category_orders={"gebied": ["Alle buurten", "Stad", "Platteland"]},
        hover_data={"x": False, "verschil_pct": False, "boven": False, "onder": False,
                    "label": False, x_kolom: True, "marge": True, "buurten_tekst": True},
        labels={"gebied": "", "marge": "95%-marge", "buurten_tekst": "aantal buurten", x_kolom: ""},
    )
    fig.update_traces(marker_size=13, textposition="middle right", textfont_size=13)
    fig.add_hline(y=0, line_color="#999999", line_width=1)
    fig.update_xaxes(tickvals=list(range(len(x_volgorde))), ticktext=as_teksten,
                     title=x_titel, showgrid=False, range=[-0.5, len(x_volgorde) - 0.3])
    fig.update_yaxes(title="Verschil in WOZ-waarde", ticksuffix="%", zeroline=False)
    fig.update_layout(title=dict(text=f"{titel}<br><sup>{ondertitel}</sup>"),
                      legend=dict(orientation="h", y=-0.22), height=460, plot_bgcolor="white")
    return fig


def kies_zoom(lon, lat):
    """Kiest het zoomniveau van de kaart zo dat alle buurten (net) in beeld passen.
    Elke stap zoom = twee keer zo dichtbij; 360 graden past op zoom 0."""
    breedte = max(lon.max() - lon.min(), (lat.max() - lat.min()) * 1.6, 0.01)
    return float(np.clip(np.log2(360 / breedte) - 0.4, 6, 13))


def toon_ontbrekende_groepen(df, uitkomst, groep_kolom, groepen, gebieden):
    """Zet onder een grafiek van tab 3 welke gebieden en groepen niet getoond worden en waarom,
    bv. 'Niet getoond: platteland · 3 of meer (maar 12 buurten)'."""
    for gebied in gebieden:
        if len(uitkomst) == 0 or gebied not in uitkomst["gebied"].values:
            st.caption(f"{gebied}: te weinig buurten per groep om betrouwbaar te vergelijken, "
                       "daarom niet in deze grafiek.")
        aantallen = df[df["gebied"] == gebied][groep_kolom].value_counts()
        for groep in groepen:
            aantal = aantallen.get(groep, 0)
            if aantal < MINIMAAL_BUURTEN:
                st.caption(f"Niet getoond: {gebied.lower()} · {groep} "
                           f"(maar {getal(aantal)} buurten, minimaal {MINIMAAL_BUURTEN} nodig).")


def gemeente_grafiek(tabel, volgorde, titel):
    """Per gemeente twee liggende balken: alle buurten vergeleken (licht) en bij gelijk
    woningtype (donker). 'volgorde' bepaalt welke gemeente bovenaan staat."""
    tabel = tabel.copy()
    tabel["label"] = tabel["verschil_pct"].apply(hoger_of_lager)
    tabel["marge"] = [marge_tekst(l, h) for l, h in zip(tabel["marge_laag_pct"], tabel["marge_hoog_pct"])]

    fig = px.bar(
        tabel, x="verschil_pct", y="gemeente", color="soort", orientation="h",
        barmode="group", text="label", color_discrete_map=KLEUREN_VERGELIJKING,
        category_orders={"soort": list(KLEUREN_VERGELIJKING.keys())},
        hover_data={"verschil_pct": False, "label": False, "soort": False,
                    "marge": True, "buurten_tekst": True},
        labels={"verschil_pct": "Verschil in WOZ-waarde binnen de gemeente", "gemeente": "",
                "soort": "", "marge": "95%-marge", "buurten_tekst": "aantal buurten"},
        title=titel,
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.add_vline(x=0, line_color="#999999", line_width=1)
    # Plotly zet de eerste categorie onderaan; omdraaien zodat volgorde[0] bovenaan staat
    fig.update_yaxes(categoryorder="array", categoryarray=list(volgorde)[::-1])
    laagste = min(tabel["verschil_pct"].min(), 0)
    hoogste = max(tabel["verschil_pct"].max(), 0)
    fig.update_xaxes(ticksuffix="%", range=[laagste * 1.3 - 4, hoogste * 1.3 + 4])
    fig.update_layout(plot_bgcolor="white", height=120 + 55 * len(volgorde),
                      margin=dict(l=10), legend=dict(orientation="h", y=1.02, x=0,
                                                     yanchor="bottom", traceorder="normal"))
    return fig


# =====================================================================
# DEEL 4 - STREAMLIT
# =====================================================================
data, stappen = laad_data()

# ----- Filters (links) -----
st.sidebar.header("Filters")

loopafstand_km = st.sidebar.slider(
    "Grens voor 'op loopafstand'", min_value=0.5, max_value=5.0,
    value=1.0, step=0.1, format="%.1f km",
    help="De afstand is de gemiddelde afstand van de inwoners van een buurt "
         "tot de dichtstbijzijnde grote supermarkt.")
afstand_tekst = f"{loopafstand_km:.1f} km".replace(".", ",")     # 1.0 -> "1,0 km"
st.sidebar.caption(f"Buurten tot en met {afstand_tekst} tellen als 'op loopafstand'.")

gebied_keuze = st.sidebar.radio("Gebied", ["Beide", "Stad", "Platteland"], horizontal=True)

# Van de keuze een lijst maken: "Beide" betekent stad en platteland samen
if gebied_keuze == "Beide":
    gekozen_gebieden = ["Stad", "Platteland"]
else:
    gekozen_gebieden = [gebied_keuze]

# Eén gefilterde dataset, die alles hieronder gebruikt
selectie = voeg_loopafstand_toe(data, loopafstand_km)
selectie = selectie[selectie["gebied"].isin(gekozen_gebieden)]

# Samenvatting onder de filters
st.sidebar.divider()
st.sidebar.markdown(f"**{getal(len(selectie))}** buurten in de selectie")
st.sidebar.caption("Stad = CBS-stedelijkheid 1 t/m 3 · Platteland = 4 en 5.")

# De statistiek voor deze selectie (stap 4 = gecorrigeerd voor woningtype en gemeente)
opbouw = bereken_opbouw(selectie[["gebied", "gemeente", "woz_x1000", "pct_eengezinswoning",
                                   "supermarkt_op_loopafstand"]])
eindstap = opbouw[opbouw["stap"] == "4 · + gemeente"].set_index("gebied")

# ----- Titel -----
st.title("Wat is een supermarkt op loopafstand waard?")
st.write(
    f"**Voor gemeenten** die bepalen waar voorzieningen komen: hebben buurten met een grote "
    f"supermarkt binnen **{afstand_tekst}** een hogere WOZ-waarde, en is dat in de stad anders "
    "dan op het platteland? CBS-cijfers over alle Nederlandse buurten."
)

# ----- KPI's -----
kol1, kol2, kol3, kol4 = st.columns(4)

with kol1.container(border=True):
    st.metric("Buurten", getal(len(selectie)))
    if gebied_keuze == "Beide":
        st.caption("stad en platteland")
    else:
        st.caption(f"alleen {gebied_keuze.lower()}")

with kol2.container(border=True):
    aandeel = selectie["supermarkt_op_loopafstand"].mean() * 100
    st.metric("Supermarkt op loopafstand", f"{aandeel:.0f}% van de buurten")
    st.caption(f"binnen {afstand_tekst}")

# Kader 3 en 4: het WOZ-verschil per gebied
meldingen = []      # hierin verzamelen we de uitleg voor de gele balk(en)
for kolom, gebied in [(kol3, "Stad"), (kol4, "Platteland")]:
    with kolom.container(border=True):
        titel = f"WOZ-verschil {gebied.lower()}"

        if gebied not in gekozen_gebieden:
            st.metric(titel, "—")
            st.caption(f"{gebied} staat uit in het filter links.")

        elif gebied in eindstap.index and eindstap.loc[gebied, "betrouwbaar"]:
            rij = eindstap.loc[gebied]
            st.metric(titel, hoger_of_lager(rij["verschil_pct"]))
            st.caption(marge_tekst(rij["marge_laag_pct"], rij["marge_hoog_pct"]))

        else:
            st.metric(titel, "Geen getal")
            st.caption("Te weinig data, zie de gele melding hieronder.")
            meldingen.append(gebied)

st.caption(
    "**Lager** betekent: buurten met een supermarkt op loopafstand hebben een lagere WOZ-waarde "
    "dan vergelijkbare buurten zonder (in dezelfde gemeente, met hetzelfde aandeel "
    "eengezinswoningen). De marge is het 95%-betrouwbaarheidsinterval."
)

# ----- Gele melding: waarom staat er bij een gebied geen getal? -----
for gebied in meldingen:
    deel = selectie[selectie["gebied"] == gebied]
    totaal = len(deel)
    met = int(deel["supermarkt_op_loopafstand"].sum())
    zonder = totaal - met
    soort = f"{gebied.lower()}sbuurten"           # "stadsbuurten" / "plattelandsbuurten"

    if zonder < met:
        situatie = (f"Bij een grens van {afstand_tekst} liggen {getal(met)} van de "
                    f"{getal(totaal)} {soort} binnen die afstand van een supermarkt. "
                    f"Er blijven maar **{getal(zonder)} {soort} zónder** supermarkt op "
                    "loopafstand over.")
        tip = "Kies links een **kleinere** loopafstand."
    else:
        situatie = (f"Bij een grens van {afstand_tekst} liggen maar **{getal(met)} van de "
                    f"{getal(totaal)} {soort}** binnen die afstand van een supermarkt.")
        tip = "Kies links een **grotere** loopafstand."

    st.warning(
        f"**Te weinig data in het {gebied.lower()}sgedeelte.** {situatie}\n\n"
        f"Om betrouwbaar te vergelijken zijn in allebei de groepen (mét en zónder supermarkt) "
        f"minstens {MINIMAAL_BUURTEN} buurten nodig. Bij minder buurten kunnen een paar "
        f"uitzonderlijke buurten het gemiddelde bepalen, en zegt het getal niets meer. "
        f"Daarom staat bij {gebied.lower()} geen getal. {tip}"
    )

# =====================================================================
# PAGINA'S
# =====================================================================
# Elke pagina is hieronder een functie. Onderaan het bestand maakt st.navigation daar een
# menu van in de zijbalk. Alleen de gekozen pagina wordt uitgerekend, en Streamlit onthoudt
# welke pagina open staat (ook als er een gele melding bij komt of verdwijnt).
st.divider()

# Kleuren voor 'op loopafstand' tegen 'niet': donker- en lichtpaars (twee kanten van hetzelfde).
# Zo blijven blauw en groen altijd gereserveerd voor stad en platteland.
KLEUREN_LOOPAFSTAND = {"Op loopafstand": KLEUR_VERSCHIL, "Niet op loopafstand": KLEUR_VERSCHIL_LICHT}
VOLGORDE_GEBIED = ["Stad", "Platteland"]
VOLGORDE_LOOPAFSTAND = ["Op loopafstand", "Niet op loopafstand"]
# Tab 3 en 5 geven alleen deze kolommen door aan bereken_per_groep. Zonder de kolom
# 'op loopafstand' hangt de uitkomst niet af van de slider, dus haalt Streamlit hem uit de cache.
KOLOMMEN_PER_GROEP = ["gebied", "gemeente", "woz_x1000", "pct_eengezinswoning"]

# Een kopie van de selectie met een leesbare kolom voor de legenda's (True/False -> tekst)
buurten = selectie.copy()
buurten["loopafstand_tekst"] = buurten["supermarkt_op_loopafstand"].map(
    {True: "Op loopafstand", False: "Niet op loopafstand"})

# ---------------------------------------------------------------------
# TAB 1 · EERSTE BLIK
# ---------------------------------------------------------------------
def pagina_eerste_blik():
    st.subheader("Op het eerste gezicht lijkt een supermarkt dichtbij de WOZ te verlagen")
    st.write("Eerst vergelijken we alle buurten simpelweg met elkaar. "
             "Daarna kijken we waarom dat eerste antwoord misleidend is.")

    # ----- Rij 1: het eerste antwoord -----
    st.markdown("#### Op het eerste gezicht")
    links, rechts = st.columns(2)

    # A. Gemiddelde WOZ: op loopafstand tegen niet
    with links:
        gemiddelden = buurten.groupby("loopafstand_tekst")["woz_euro"].mean().reset_index()
        gemiddelden["label"] = "€ " + gemiddelden["woz_euro"].apply(getal)
        fig = px.bar(
            gemiddelden, x="loopafstand_tekst", y="woz_euro", text="label",
            color="loopafstand_tekst", color_discrete_map=KLEUREN_LOOPAFSTAND,
            category_orders={"loopafstand_tekst": VOLGORDE_LOOPAFSTAND},
            labels={"loopafstand_tekst": "", "woz_euro": "Gemiddelde WOZ-waarde (€)"},
            title="A · Gemiddelde WOZ-waarde per buurt",
        )
        fig.update_traces(textfont_size=13)
        fig.update_layout(plot_bgcolor="white", height=380, showlegend=False)
        st.plotly_chart(fig)

        eerste = opbouw[opbouw["stap"] == "1 · Niets"]
        if len(eerste) > 0:
            st.markdown(f"Buurten met een supermarkt binnen {afstand_tekst} hebben een "
                        f"**{hoger_of_lager(eerste['verschil_pct'].iloc[0])}** WOZ-waarde.")

    # B. Hoe ver ligt de supermarkt? Per afstandsband het % buurten, stad en platteland apart
    with rechts:
        verdeling = (buurten.groupby(["gebied", "afstandsband"], observed=False).size()
                     .reset_index(name="aantal"))
        # Percentage binnen elk gebied (zodat stad en platteland even zwaar tellen)
        verdeling["procent"] = (verdeling["aantal"]
                                / verdeling.groupby("gebied")["aantal"].transform("sum") * 100)
        verdeling["label"] = verdeling["procent"].round(0).astype(int).astype(str) + "%"
        fig = px.bar(
            verdeling, x="afstandsband", y="procent", text="label",
            color="gebied", color_discrete_map=KLEUREN, barmode="group",
            category_orders={"afstandsband": BAND_NAMEN, "gebied": VOLGORDE_GEBIED},
            labels={"afstandsband": "Afstand tot grote supermarkt", "procent": "% van de buurten",
                    "gebied": ""},
            title="B · Hoe ver ligt de supermarkt?",
        )
        fig.update_yaxes(ticksuffix="%")
        fig.update_traces(textfont_size=13)
        fig.update_layout(plot_bgcolor="white", height=380,
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig)

        st.caption("Stadsbuurten liggen bijna allemaal binnen 1 km van een supermarkt. "
                    "Plattelandsbuurten liggen veel verder weg, tot meer dan 5 km.")

    # ----- Rij 2: waarom dat misleidt -----
    st.markdown("#### Maar de buurten met en zonder supermarkt zijn niet vergelijkbaar")
    links, rechts = st.columns(2)

    # C. Aandeel buurten met supermarkt op loopafstand, per gebied
    with links:
        aandeel_gebied = (buurten.groupby("gebied")["supermarkt_op_loopafstand"].mean() * 100
                          ).reset_index(name="procent")
        aandeel_gebied["label"] = aandeel_gebied["procent"].round(0).astype(int).astype(str) + "%"
        fig = px.bar(
            aandeel_gebied, x="gebied", y="procent", text="label",
            color="gebied", color_discrete_map=KLEUREN,
            category_orders={"gebied": VOLGORDE_GEBIED},
            labels={"gebied": "", "procent": "% van de buurten"},
            title=f"C · Buurten met een supermarkt binnen {afstand_tekst}",
        )
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        fig.update_traces(textfont_size=13)
        fig.update_layout(plot_bgcolor="white", height=380, showlegend=False)
        st.plotly_chart(fig)

        st.caption("Op loopafstand betekent dus vooral: in de stad. "
                   "Wie 'op loopafstand' met 'niet' vergelijkt, vergelijkt grotendeels stad met platteland.")

    # D. Percentage eengezinswoningen: op loopafstand tegen niet, per gebied
    with rechts:
        woningtype = (buurten.groupby(["gebied", "loopafstand_tekst"])
                      .agg(pct_eengezinswoning=("pct_eengezinswoning", "mean"),
                           aantal=("pct_eengezinswoning", "count"))
                      .reset_index())
        # Groepen met te weinig buurten laten we niet zien (zelfde regel als bij de KPI's)
        te_klein = woningtype[woningtype["aantal"] < MINIMAAL_BUURTEN]
        woningtype = woningtype[woningtype["aantal"] >= MINIMAAL_BUURTEN].copy()
        woningtype["label"] = woningtype["pct_eengezinswoning"].round(0).astype(int).astype(str) + "%"
        fig = px.bar(
            woningtype, x="gebied", y="pct_eengezinswoning", text="label",
            color="loopafstand_tekst", color_discrete_map=KLEUREN_LOOPAFSTAND,
            category_orders={"loopafstand_tekst": VOLGORDE_LOOPAFSTAND, "gebied": VOLGORDE_GEBIED},
            barmode="group",
            labels={"gebied": "", "pct_eengezinswoning": "% eengezinswoningen",
                    "loopafstand_tekst": ""},
            title="D · Aandeel eengezinswoningen in de buurt",
        )
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        fig.update_traces(textfont_size=13)
        fig.update_layout(plot_bgcolor="white", height=380,
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig)

        st.caption("Dicht bij een supermarkt staan minder eengezinswoningen en dus meer "
                   "appartementen. Die zijn kleiner, en daardoor goedkoper.")
        for _, rij in te_klein.iterrows():
            st.caption(f"Niet getoond: {rij['gebied'].lower()} · {rij['loopafstand_tekst'].lower()} "
                       f"(maar {getal(rij['aantal'])} buurten).")

    # ----- Conclusie van deze tab -----
    st.info(
        "**Het eerste getal meet dus niet alleen de supermarkt.** Buurten op loopafstand liggen "
        "vaker in de stad en hebben vaker appartementen. In **tab 2** vergelijken we alleen vergelijkbare buurten: "
        "stad met stad, platteland met platteland, binnen dezelfde gemeente en bij hetzelfde "
        "aandeel eengezinswoningen."
    )

# ---------------------------------------------------------------------
# TAB 2 · VERGELIJKBARE BUURTEN
# ---------------------------------------------------------------------
def pagina_vergelijkbare_buurten():
    st.subheader("Vergelijkbare buurten: wat blijft er over van het verschil?")
    st.write("In tab 1 zag je twee redenen waarom het eerste getal misleidend is. Hier halen "
             "we die er stap voor stap uit. Bij elke stap houden we één ding meer gelijk.")

    # ----- Rij 1: de hoofdgrafiek -----
    tonen = opbouw[opbouw["betrouwbaar"]].copy()
    if len(tonen) == 0:
        st.info("Met deze instellingen zijn er te weinig buurten om betrouwbaar te vergelijken.")
    else:
        tonen["buurten_tekst"] = (tonen["buurten_ja"].apply(getal) + " op loopafstand · "
                                  + tonen["buurten_nee"].apply(getal) + " niet")
        stappen_volgorde = ["1 · Niets", "2 · + stad / platteland", "3 · + woningtype",
                            "4 · + gemeente"]
        fig = puntengrafiek(
            tonen, "stap", stappen_volgorde,
            titel="Verschil in WOZ-waarde: supermarkt op loopafstand tegen niet",
            ondertitel="Elk punt is het verschil; het streepje is de 95%-marge. "
                       "Raakt het streepje de 0 niet, dan is het verschil geen toeval.",
            x_titel="Wat houden we gelijk?",
        )
        st.plotly_chart(fig)
        st.caption("Stap 1: alle buurten op één hoop. Stap 2: stad met stad, platteland met "
                   "platteland. Stap 3: ook hetzelfde aandeel eengezinswoningen. "
                   "Stap 4: ook alleen buurten binnen dezelfde gemeente.")

    # ----- Rij 2: stap 4 in beeld, per gemeente (volle breedte) -----
    # 1. Alle buurten vergeleken: per gemeente het gemiddelde van 'op loopafstand' tegen 'niet'
    gewoon = bereken_verschil(buurten, "gemeente", "supermarkt_op_loopafstand", "woz_x1000")

    # 2. Alleen gemeenten met genoeg buurten in beide groepen; daarvan de grootste
    if len(gewoon) > 0:
        gewoon = gewoon[gewoon[["buurten_ja", "buurten_nee"]].min(axis=1)
                        >= MINIMAAL_BUURTEN_GEMEENTE].copy()
        gewoon["buurten"] = gewoon["buurten_ja"] + gewoon["buurten_nee"]
        gewoon = gewoon.nlargest(AANTAL_GEMEENTEN, "buurten")

    if len(gewoon) == 0:
        st.info("Te weinig buurten om per gemeente te vergelijken.")
    else:
        # 3. Bij gelijk woningtype: dezelfde gemeenten en dezelfde buurten, maar nu met een
        #    regressie die het percentage eengezinswoningen gelijk houdt (zoals stap 3,
        #    maar dan apart voor elke gemeente)
        zelfde_gemeenten = buurten[buurten["gemeente"].isin(gewoon["gemeente"])]
        eerlijk = bereken_verschil_gecorrigeerd(zelfde_gemeenten, "gemeente",
                                                "supermarkt_op_loopafstand", "woz_x1000",
                                                ["pct_eengezinswoning"])

        # 4. Beide onder elkaar in een tabel, met een kolom 'soort' voor de kleur
        gewoon["soort"] = "Alle buurten vergeleken"
        eerlijk["soort"] = "Bij gelijk woningtype"
        kolommen = ["gemeente", "soort", "buurten_ja", "buurten_nee",
                    "verschil_pct", "marge_laag_pct", "marge_hoog_pct"]
        beide = pd.concat([gewoon[kolommen], eerlijk[kolommen]], ignore_index=True)
        beide["buurten_tekst"] = (beide["buurten_ja"].apply(getal) + " op loopafstand · "
                                  + beide["buurten_nee"].apply(getal) + " niet")

        # Volgorde: de gemeente met het grootste verschil bij gelijk woningtype bovenaan
        volgorde = eerlijk.sort_values("verschil_pct")["gemeente"]
        fig = gemeente_grafiek(
            beide, volgorde,
            titel=f"Stap 4 in beeld: het verschil binnen de {len(volgorde)} grootste gemeenten",
        )
        st.plotly_chart(fig)

        # Het inzicht in woorden: in hoeveel gemeenten is het verschil bij gelijk woningtype zeker?
        zeker_lager = int((eerlijk["marge_hoog_pct"] < 0).sum())
        zeker_hoger = int((eerlijk["marge_laag_pct"] > 0).sum())
        onzeker = len(eerlijk) - zeker_lager - zeker_hoger
        voorbeeld = eerlijk.sort_values("verschil_pct").iloc[0]
        voorbeeld_gewoon = gewoon.set_index("gemeente").loc[voorbeeld["gemeente"], "verschil_pct"]

        st.caption(
            f"Lees zo: in **{voorbeeld['gemeente']}** zijn buurten met een supermarkt op "
            f"loopafstand **{hoger_of_lager(voorbeeld_gewoon)}** als je alle buurten vergelijkt. Vergelijk je "
            f"alleen buurten met hetzelfde aandeel eengezinswoningen, dan is het nog "
            f"**{hoger_of_lager(voorbeeld['verschil_pct'])}**. Het verschil tussen de lichte en "
            "de donkere balk komt dus door het woningtype (appartementen tegen huizen)."
        )
        uitkomst = f"In **{zeker_lager} van de {len(eerlijk)}** gemeenten is het verschil bij gelijk woningtype zeker lager"
        if zeker_hoger > 0:
            uitkomst += f", in **{zeker_hoger}** zeker hoger"
        if onzeker > 0:
            uitkomst += f", en in **{onzeker}** kan het ook toeval zijn (de marge raakt de 0)"
        st.caption(uitkomst + ". Beweeg de muis over een balk voor de marge. Getoond: de "
                   f"gemeenten met de meeste buurten, met minstens {MINIMAAL_BUURTEN_GEMEENTE} "
                   "buurten op loopafstand én niet.")

    # ----- Hoe is dit berekend? -----
    with st.expander("Hoe is dit berekend?"):
        st.markdown(
            "- **Procenten:** we rekenen met de logaritme van de WOZ-waarde. Zo wordt een verschil "
            "vanzelf een percentage, en tellen dure en goedkope buurten even zwaar mee.\n"
            "- **Stap 1 en 2** vergelijken gewoon de gemiddelden van twee groepen.\n"
            "- **Stap 3 en 4** gebruiken een regressie. Die houdt het percentage eengezinswoningen "
            "gelijk, en geeft elke gemeente haar eigen prijsniveau (`C(gemeente)`). Zo wordt het "
            "verschil gemeten tussen buurten in dezelfde gemeente, met hetzelfde woningtype.\n"
            "- **De marge** (95%-betrouwbaarheidsinterval): het bereik waarin het echte verschil "
            "met 95% zekerheid ligt. Hoe meer buurten, hoe smaller.\n"
            "- **R²**: hoeveel van de verschillen in WOZ tussen buurten het model verklaart "
            "(0 = niets, 1 = alles). Een lage R² betekent niet dat het verschil onzeker is; "
            "daarvoor kijk je naar de marge."
        )
        uitleg = opbouw[opbouw["betrouwbaar"]].copy()
        uitleg["verschil"] = uitleg["verschil_pct"].apply(procent)
        uitleg["95%-marge"] = [f"{procent(l)} tot {procent(h)}"
                               for l, h in zip(uitleg["marge_laag_pct"], uitleg["marge_hoog_pct"])]
        # Stap 1 en 2 hebben geen regressie, dus geen R²: daar een streepje
        uitleg["R²"] = uitleg["r_kwadraat"].apply(lambda r: "–" if pd.isna(r) else f"{r:.2f}".replace(".", ","))
        st.dataframe(uitleg[["stap", "gebied", "verschil", "95%-marge", "R²"]], hide_index=True)

    # ----- Conclusie van deze tab -----
    regels = []
    alle_lager = True
    for gebied in VOLGORDE_GEBIED:
        if gebied in eindstap.index and eindstap.loc[gebied, "betrouwbaar"]:
            regels.append(f"**{hoger_of_lager(eindstap.loc[gebied, 'verschil_pct'])}** in de "
                          f"{'stad' if gebied == 'Stad' else 'plattelandsbuurten'}")
            alle_lager = alle_lager and eindstap.loc[gebied, "verschil_pct"] < 0
    if regels:
        kop = ("zijn buurten met een supermarkt op loopafstand niet meer waard, maar juist "
               "goedkoper: " if alle_lager else "is het verschil: ")
        st.info("**Ook bij vergelijkbare buurten** (binnen dezelfde gemeente en bij hetzelfde woningtype) "
                + kop + " en ".join(regels) + ". Bij welke afstand begint dat verschil? "
                "Zie **tab 3**.")

# ---------------------------------------------------------------------
# TAB 3 · AFSTAND EN AANTAL
# ---------------------------------------------------------------------
def pagina_afstand():
    st.subheader("Hoe verder de supermarkt, hoe hoger de WOZ?")
    st.write("In tab 2 waren er twee groepen: op loopafstand of niet. Hier kijken we preciezer: "
             "eerst naar de afstand tot de supermarkt, daarna naar het aantal supermarkten dichtbij.")
    st.caption("Deze tab gebruikt vaste groepen en niet de slider. Het filter Gebied werkt wel. "
               f"Groepen met minder dan {MINIMAAL_BUURTEN} buurten worden niet getoond.")

    # ----- Rij 1: per afstandsband -----
    st.markdown("#### 1 · Afstand tot de dichtstbijzijnde grote supermarkt")
    banden = bereken_per_groep(buurten[KOLOMMEN_PER_GROEP + ["afstandsband"]],
                               "afstandsband", BAND_NAMEN)
    if len(banden) == 0:
        st.info("Te weinig buurten om de afstandsbanden betrouwbaar te vergelijken.")
    else:
        banden["buurten_tekst"] = banden["buurten"].apply(getal) + " buurten"
        fig = puntengrafiek(
            banden, "afstandsband", BAND_NAMEN,
            titel="Verschil in WOZ-waarde per afstandsband, bij vergelijkbare buurten",
            ondertitel="Vergeleken met buurten met een supermarkt binnen 0,5 km, in dezelfde "
                       "gemeente en bij hetzelfde aandeel eengezinswoningen.",
            x_titel="Afstand tot grote supermarkt",
        )
        st.plotly_chart(fig)
    toon_ontbrekende_groepen(buurten, banden, "afstandsband", BAND_NAMEN, gekozen_gebieden)

    # ----- Rij 2: per aantal supermarkten binnen 1 km -----
    st.markdown("#### 2 · Aantal grote supermarkten binnen 1 km")
    aantallen = bereken_per_groep(buurten[KOLOMMEN_PER_GROEP + ["aantal_supermarkten"]],
                                  "aantal_supermarkten", AANTAL_NAMEN)
    if len(aantallen) == 0:
        st.info("Te weinig buurten om het aantal supermarkten betrouwbaar te vergelijken.")
    else:
        aantallen["buurten_tekst"] = aantallen["buurten"].apply(getal) + " buurten"
        fig = puntengrafiek(
            aantallen, "aantal_supermarkten", AANTAL_NAMEN,
            titel="Verschil in WOZ-waarde naar aantal supermarkten binnen 1 km, bij vergelijkbare buurten",
            ondertitel="Vergeleken met buurten zonder grote supermarkt binnen 1 km, in dezelfde "
                       "gemeente en bij hetzelfde aandeel eengezinswoningen.",
            x_titel="Aantal grote supermarkten binnen 1 km",
        )
        st.plotly_chart(fig)
        st.caption("Het CBS geeft per buurt het gemiddelde aantal supermarkten binnen 1 km "
                   "(bijvoorbeeld 1,4). Dat is hier afgerond naar een heel getal.")
    toon_ontbrekende_groepen(buurten, aantallen, "aantal_supermarkten", AANTAL_NAMEN,
                             gekozen_gebieden)

    # ----- Hoe is dit berekend? -----
    with st.expander("Hoe is dit berekend?"):
        st.markdown(
            "- **Groepen:** met `pd.cut` delen we de buurten in groepen: vijf afstandsbanden, "
            "en het aantal supermarkten binnen 1 km (0, 1, 2, 3 of meer).\n"
            "- **Vergelijking:** dezelfde regressie als stap 4 in tab 2, maar met een 0/1-kolom "
            "per groep in plaats van één kolom 'op loopafstand'. Elk punt is het verschil met de "
            "eerste groep (het vergelijkingspunt), in dezelfde gemeente en bij hetzelfde aandeel "
            "eengezinswoningen.\n"
            "- **De marge** (het streepje): raakt die de 0 niet, dan is het verschil geen toeval."
        )

    # ----- Conclusie van deze tab -----
    regels = []
    verder_weg_duurder = True          # blijft True als overal een verdere band het duurst is
    for gebied in VOLGORDE_GEBIED:
        deel = banden[banden["gebied"] == gebied] if len(banden) > 0 else banden
        if len(deel) < 2:
            continue
        hoogste = deel.loc[deel["verschil_pct"].idxmax()]
        if hoogste["vergelijkingspunt"]:
            verder_weg_duurder = False
        regels.append(f"in de **{'stad' if gebied == 'Stad' else 'plattelandsbuurten'}** zijn "
                      f"buurten op **{hoogste['afstandsband']}** het duurst "
                      f"({hoger_of_lager(hoogste['verschil_pct'])} dan binnen 0,5 km)")
    if regels:
        kop = ("Een supermarkt vlakbij maakt een buurt niet meer waard:" if verder_weg_duurder
               else "Bij vergelijkbare buurten:")
        st.info(f"**{kop}** " + "; ".join(regels) + ". "
                "Grafiek 2 laat zien of méér supermarkten dichtbij nog verschil maakt.")

# ---------------------------------------------------------------------
# TAB 4 · KAART
# ---------------------------------------------------------------------
def pagina_kaart():
    st.subheader("Op de kaart: waar ligt de supermarkt dichtbij, en waar is het duur?")
    st.write("**Links:** welke buurten hebben een grote supermarkt op loopafstand (volgt de "
             "slider)? **Rechts:** hoe hoog is de WOZ-waarde? Leg de kaarten naast elkaar.")
    st.info("**Waar je op let:** liggen de paarse buurten (supermarkt dichtbij) vooral in het "
            "centrum, en zijn diezelfde buurten rechts lichter (goedkoper) dan de buurten aan de "
            "rand? Dat ruimtelijke patroon, een goedkoper centrum met een duurdere ring eromheen, "
            "zie je in geen enkele tabel. Schuif met de slider om te zien hoe het paarse gebied "
            "groeit, en kies een andere gemeente om te zien of het daar ook zo is.")

    vlakken, middelpunten = maak_kaartdata()

    # Leesbare teksten voor als je met de muis over een buurt gaat
    kaart = buurten[buurten["buurtcode"].isin(vlakken.keys())].copy()
    kaart["woz_tekst"] = "€ " + kaart["woz_euro"].apply(getal)
    kaart["afstand_tekst"] = kaart["afstand_supermarkt_km"].map(
        lambda km: f"{km:.1f} km".replace(".", ","))
    kaart["eengezins_tekst"] = kaart["pct_eengezinswoning"].round(0).astype(int).astype(str) + "%"
    muis = {"buurtcode": False, "loopafstand_tekst": False, "woz_euro": False,
            "gemeente": True, "woz_tekst": True, "afstand_tekst": True, "eengezins_tekst": True}
    namen = {"loopafstand_tekst": "", "woz_euro": "WOZ-waarde (€)", "gemeente": "gemeente",
             "woz_tekst": "WOZ-waarde", "afstand_tekst": "afstand tot supermarkt",
             "eengezins_tekst": "eengezinswoningen"}

    # Kleurenschaal voor de WOZ: de 5% goedkoopste en duurste buurten krijgen de uiterste kleur,
    # anders maken een paar hele dure buurten alle andere buurten even licht
    woz_bereik = [kaart["woz_euro"].quantile(0.05), kaart["woz_euro"].quantile(0.95)]

    # ----- Eén gemeente: de echte buurtgrenzen -----
    gemeentelijst = sorted(kaart["gemeente"].dropna().unique())
    standaard = gemeentelijst.index("Houten") if "Houten" in gemeentelijst else 0
    gemeente = st.selectbox("Kies een gemeente", gemeentelijst, index=standaard,
                            key="kaart_gemeente")
    deel = kaart[kaart["gemeente"] == gemeente]

    # Alleen de vlakken van deze gemeente, en de kaart daarop inzoomen
    geojson = {"type": "FeatureCollection",
               "features": [vlakken[code] for code in deel["buurtcode"]]}
    midden = middelpunten[middelpunten["buurtcode"].isin(deel["buurtcode"])]
    centrum = {"lat": midden["lat"].mean(), "lon": midden["lon"].mean()}
    zoom = kies_zoom(midden["lon"], midden["lat"])

    links, rechts = st.columns(2)
    with links:
        fig = px.choropleth_map(
            deel, geojson=geojson, locations="buurtcode", featureidkey="properties.statcode",
            color="loopafstand_tekst", color_discrete_map=KLEUREN_LOOPAFSTAND,
            category_orders={"loopafstand_tekst": VOLGORDE_LOOPAFSTAND},
            hover_name="buurtnaam", hover_data=muis, labels=namen,
            center=centrum, zoom=zoom, map_style="carto-positron", opacity=0.8,
            title=f"Supermarkt binnen {afstand_tekst}?",
        )
        # Legenda rechts van de kaart (net als de kleurbalk rechts), zodat beide kaarten even groot zijn
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0),
                          legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"))
        st.plotly_chart(fig)
    with rechts:
        fig = px.choropleth_map(
            deel, geojson=geojson, locations="buurtcode", featureidkey="properties.statcode",
            color="woz_euro", color_continuous_scale="Blues", range_color=woz_bereik,
            hover_name="buurtnaam", hover_data=muis, labels=namen,
            center=centrum, zoom=zoom, map_style="carto-positron", opacity=0.8,
            title="Gemiddelde WOZ-waarde",
        )
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig)

    # ----- Wat betekent de kaart voor deze gemeente? -----
    met = deel[deel["supermarkt_op_loopafstand"]]
    zonder = deel[~deel["supermarkt_op_loopafstand"]]
    genoeg = min(len(met), len(zonder)) >= MINIMAAL_BUURTEN_GEMEENTE

    st.markdown(f"#### Wat betekent dit voor {gemeente}?")
    k1, k2, k3 = st.columns(3)

    # Kader 1 en 2: gewoon de gemiddelde WOZ van de paarse en de lichtpaarse buurten
    with k1.container(border=True):
        waarde = f"€ {getal(met['woz_euro'].mean())}" if len(met) > 0 else "—"
        st.metric("Gem. WOZ mét supermarkt op loopafstand", waarde)
        st.caption(f"de {getal(len(met))} paarse buurten op de kaart")
    with k2.container(border=True):
        waarde = f"€ {getal(zonder['woz_euro'].mean())}" if len(zonder) > 0 else "—"
        st.metric("Gem. WOZ zónder supermarkt op loopafstand", waarde)
        st.caption(f"de {getal(len(zonder))} lichtpaarse buurten op de kaart")

    # Kader 3: het verschil bij gelijk woningtype (zelfde berekening als de donkere balk in tab 2)
    with k3.container(border=True):
        if genoeg:
            eerlijk_gem = bereken_verschil_gecorrigeerd(deel, "gemeente",
                                                        "supermarkt_op_loopafstand",
                                                        "woz_x1000",
                                                        ["pct_eengezinswoning"]).iloc[0]
            st.metric("Verschil bij gelijk woningtype",
                      hoger_of_lager(eerlijk_gem["verschil_pct"]))
            st.caption(marge_tekst(eerlijk_gem["marge_laag_pct"],
                                   eerlijk_gem["marge_hoog_pct"]))
        else:
            st.metric("Verschil bij gelijk woningtype", "Geen getal")
            st.caption("te weinig buurten, zie hieronder")

    # De uitleg in gewone woorden
    if genoeg:
        verschil_euro = met["woz_euro"].mean() - zonder["woz_euro"].mean()
        richting = "goedkoper" if verschil_euro < 0 else "duurder"
        st.info(
            f"**Zo lees je dit:** in {gemeente} zijn buurten met een supermarkt binnen "
            f"{afstand_tekst} gemiddeld **€ {getal(abs(verschil_euro))} {richting}** dan "
            "buurten zonder. Een deel daarvan komt doordat er bij de supermarkt meer "
            "appartementen staan. Vergelijk je alleen buurten met hetzelfde aandeel "
            f"eengezinswoningen, dan is het verschil "
            f"**{hoger_of_lager(eerlijk_gem['verschil_pct'])}**."
        )
    else:
        if len(zonder) < MINIMAAL_BUURTEN_GEMEENTE:
            situatie = (f"maar {getal(len(zonder))} buurt(en) zónder supermarkt binnen "
                        f"{afstand_tekst}. Kies links een kleinere loopafstand.")
        else:
            situatie = (f"maar {getal(len(met))} buurt(en) mét supermarkt binnen "
                        f"{afstand_tekst}. Kies links een grotere loopafstand.")
        st.warning(
            f"**Geen verschil bij gelijk woningtype voor {gemeente}:** er zijn {situatie} Om betrouwbaar te "
            f"vergelijken zijn minstens {MINIMAAL_BUURTEN_GEMEENTE} buurten mét én "
            f"{MINIMAAL_BUURTEN_GEMEENTE} zónder supermarkt op loopafstand nodig; met minder "
            "buurten bepalen een paar uitschieters het getal."
        )
    if gebied_keuze != "Beide":
        st.caption(f"Let op: het filter staat op {gebied_keuze.lower()}, dus alleen die "
                   "buurten van deze gemeente staan op de kaart.")

    st.caption("Klik op een kleur in de legenda om die groep aan of uit te zetten. Witte "
               "vlakken zijn buurten zonder WOZ-waarde (niet gepubliceerd door het CBS). "
               "Buurtgrenzen: CBS/PDOK, 2024.")

# ---------------------------------------------------------------------
# TAB 5 · CONCLUSIE
# ---------------------------------------------------------------------
def pagina_conclusie():
    st.subheader("Conclusie: wat is een supermarkt op loopafstand waard?")

    # ----- Het antwoord in één zin -----
    getoond = [g for g in VOLGORDE_GEBIED
               if g in eindstap.index and eindstap.loc[g, "betrouwbaar"]]
    alle_lager = all(eindstap.loc[g, "verschil_pct"] < 0 for g in getoond)
    if len(getoond) == 0:
        st.warning("Met deze instellingen is er te weinig data voor een conclusie. "
                   "Zet de slider op 1 km en het filter Gebied op Beide.")
    else:
        delen = [f"**{hoger_of_lager(eindstap.loc[g, 'verschil_pct'])}** in de "
                 f"{'stad' if g == 'Stad' else 'plattelandsbuurten'}" for g in getoond]
        kop = ("Een supermarkt op loopafstand maakt een buurt niet meer waard, maar juist "
               "minder." if alle_lager else "Het antwoord verschilt per gebied.")
        st.info(f"### {kop}\n\nVergelijk je buurten in dezelfde gemeente en met hetzelfde "
                f"aandeel eengezinswoningen, dan is de WOZ-waarde van buurten met een grote "
                f"supermarkt binnen {afstand_tekst} " + " en ".join(delen) + ".")

    # ----- Hoe we daar kwamen (alleen tekst) -----
    with st.container(border=True):
        st.markdown("#### Hoe we daar kwamen")
        eerste = opbouw[opbouw["stap"] == "1 · Niets"]
        if len(eerste) > 0:
            st.markdown(f"**1 · Eerste blik:** alle buurten samen zijn met een supermarkt op "
                        f"loopafstand **{hoger_of_lager(eerste['verschil_pct'].iloc[0])}**. "
                        "Maar die buurten liggen vaker in de stad en hebben vaker appartementen.")
        if len(getoond) > 0 and alle_lager:
            st.markdown("**2 · Vergelijkbare buurten:** halen we stad/platteland, woningtype en "
                        "gemeente eruit, dan wordt het verschil kleiner, maar het blijft negatief.")
        banden_conclusie = bereken_per_groep(buurten[KOLOMMEN_PER_GROEP + ["afstandsband"]],
                                             "afstandsband", BAND_NAMEN)
        zinnen = []
        for gebied in VOLGORDE_GEBIED:
            deel = (banden_conclusie[banden_conclusie["gebied"] == gebied]
                    if len(banden_conclusie) > 0 else banden_conclusie)
            if len(deel) < 2:
                continue
            hoogste = deel.loc[deel["verschil_pct"].idxmax()]
            if hoogste["vergelijkingspunt"]:
                continue                    # hier is 'naast de supermarkt' juist het duurst
            zinnen.append(f"{'in de stad' if gebied == 'Stad' else 'op het platteland'} op "
                          f"{hoogste['afstandsband']} "
                          f"({hoger_of_lager(hoogste['verschil_pct'])})")
        if zinnen:
            st.markdown("**3 · Afstand:** de duurste buurten liggen niet naast de supermarkt, "
                        "maar " + " en ".join(zinnen) + ", vergeleken met binnen 0,5 km.")

    # ----- Rij 2: advies en beperkingen -----
    links, rechts = st.columns(2)
    with links.container(border=True):
        st.markdown("#### Advies aan de gemeente")

        # De getallen voor in het advies (bij vergelijkbare buurten, stap 4); None = geen getal
        stad = eindstap.loc["Stad", "verschil_pct"] if "Stad" in getoond else None
        platteland = eindstap.loc["Platteland", "verschil_pct"] if "Platteland" in getoond else None

        advies = []
        cijfers = []
        if stad is not None:
            cijfers.append(f"{hoger_of_lager(stad)} in de stad")
        if platteland is not None:
            cijfers.append(f"{hoger_of_lager(platteland)} op het platteland")
        if cijfers and all(g is None or g < 0 for g in [stad, platteland]):
            advies.append("**Reken niet op een hogere woningwaarde door een supermarkt op "
                          "loopafstand.** Bij vergelijkbare buurten is de WOZ "
                          + " en ".join(cijfers) + ".")
        if zinnen:
            advies.append("**Plan woningen niet pal naast de supermarkt.** De buurten met de "
                          "hoogste WOZ liggen " + " en ".join(zinnen) + ". Een supermarkt "
                          "aan de rand van de kern, op fietsafstand, past daar beter bij.")
        if stad is not None and platteland is not None and stad < 0 and platteland < 0:
            factor = f"{platteland / stad:.1f}".replace(".", ",")
            advies.append(f"**Let extra op in dorpen.** Op het platteland is het verschil "
                          f"{factor} keer zo groot als in de stad. Wie daar woont, kiest vaak "
                          "voor rust en ruimte.")
        advies.append("**Weeg het af.** Dit onderzoek gaat alleen over woningwaarde. Voor "
                      "ouderen en mensen zonder auto blijft een supermarkt dichtbij belangrijk.")
        st.markdown("\n".join(f"{i}. {regel}" for i, regel in enumerate(advies, start=1)))
    with rechts.container(border=True):
        st.markdown("#### Waar je voorzichtig mee moet zijn")
        st.markdown(
            "- **Verband, geen oorzaak.** Supermarkten staan in drukke centra; misschien "
            "drukt die drukte de prijs en niet de winkel zelf.\n"
            "- **Woningtype is grof gemeten.** We kennen alleen het percentage "
            "eengezinswoningen, niet de grootte of het bouwjaar van de woningen.\n"
            "- **Gemiddelden per buurt.** De afstand en de WOZ zijn buurtgemiddelden, geen "
            "waarden per woning.\n"
            "- **Verschillende peildata.** De WOZ is van 1 januari 2023, de afstanden van 2024.\n"
            "- **Weggelaten buurten.** Buurten zonder WOZ-waarde (vaak kleine buurten met weinig "
            "woningen) doen niet mee. Zie **6 · Over de data**."
        )

# ---------------------------------------------------------------------
# TAB 6 · OVER DE DATA
# ---------------------------------------------------------------------
def pagina_over_de_data():
    st.subheader("Over de data: waar komt het vandaan en wat is ermee gebeurd?")
    st.caption("Dit onderdeel gaat over alle buurten; de filters links werken hier niet.")

    # ----- De bronnen -----
    st.markdown("#### De bronnen")
    st.dataframe(pd.DataFrame({
        "Bron": ["CBS · Nabijheid voorzieningen, buurtcijfers 2024",
                 "CBS · Kerncijfers wijken en buurten 2024",
                 "PDOK · CBS-gebiedsindelingen, buurten (vereenvoudigd)"],
        "Tabel": ["86134NED", "85984NED", "buurt_gegeneraliseerd, jaar 2024"],
        "Wat we eruit gebruiken": [
            "afstand tot grote supermarkt; aantal binnen 1, 3 en 5 km",
            "WOZ-waarde (peildatum 1 jan 2023), stedelijkheid, % eengezinswoningen, gemeente",
            "buurtgrenzen voor de kaart"],
        "Opgehaald": ["26-10-2026, CBS OData v4-API (ophalen.py)",
                      "26-10-2026, CBS OData v4-API (ophalen.py)",
                      "26-10-2026, PDOK OGC-API (ophalen.py)"],
    }), hide_index=True)
    st.caption("De sleutel is de buurtcode (bv. BU03630000). Die werkt niet vanzelf: het CBS "
               "zet er spaties achter, en de tabellen bevatten ook gemeenten (GM) en wijken (WK).")

    # ----- Rij 1: van ruwe data naar de dataset -----
    links, rechts = st.columns(2)
    with links:
        trechter = pd.DataFrame({
            "stap": ["Buurten in kerncijfers", "Buurten in nabijheid",
                     "In allebei (na samenvoegen)", "Met WOZ en woningtype (eindtotaal)"],
            "buurten": [stappen["buurten_kerncijfers"], stappen["buurten_nabijheid"],
                        stappen["na_samenvoegen"], stappen["eindtotaal"]],
        })
        trechter["label"] = trechter["buurten"].apply(getal)
        fig = px.bar(trechter, x="buurten", y="stap", orientation="h", text="label",
                     labels={"buurten": "Aantal buurten", "stap": ""},
                     title="Van ruwe data naar de dataset")
        fig.update_traces(marker_color=KLEUREN["Stad"], textposition="inside",
                          insidetextanchor="end", textfont_size=13)
        fig.update_yaxes(categoryorder="array", categoryarray=list(trechter["stap"])[::-1])
        fig.update_layout(plot_bgcolor="white", height=330)
        st.plotly_chart(fig)
        st.caption(f"Alleen in de kerncijfers: {getal(stappen['alleen_kerncijfers'])} buurten. "
                   f"Alleen in de nabijheidstabel: {getal(stappen['alleen_nabijheid'])}. "
                   "Die vallen weg bij het samenvoegen (inner join).")

    with rechts:
        missend = stappen["missend_per_stedelijkheid"].copy()
        missend["label"] = missend["procent_zonder_woz"].round(1).astype(str).str.replace(".", ",") + "%"
        fig = px.bar(missend, x="stedelijkheid", y="procent_zonder_woz", text="label",
                     labels={"stedelijkheid": "", "procent_zonder_woz": "% buurten zonder WOZ"},
                     title="Welke buurten hebben geen WOZ-waarde?")
        fig.update_traces(marker_color=KLEUREN["Alle buurten"], textfont_size=13)
        fig.update_yaxes(ticksuffix="%")
        fig.update_layout(plot_bgcolor="white", height=330)
        st.plotly_chart(fig)
        st.caption("Het CBS publiceert geen WOZ bij buurten met heel weinig woningen. Die laten "
                   "we weg in plaats van een waarde te verzinnen (bv. het gemiddelde): dan zouden "
                   "we prijzen invullen die niet bestaan.")

    # ----- Rij 2: wat er met de data is gebeurd -----
    st.markdown("#### Wat er met de data is gebeurd")
    links, rechts = st.columns(2)
    with links:
        st.markdown("**Zo komt de data binnen (lang):** één rij per buurt per meting, met codes.")
        st.dataframe(stappen["voorbeeld_lang"], hide_index=True)
    with rechts:
        st.markdown("**Na pivoteren (breed):** één rij per buurt, met leesbare kolomnamen.")
        st.dataframe(stappen["voorbeeld_breed"], hide_index=True)

    st.markdown("**Zelf afgeleide kolommen**")
    st.dataframe(pd.DataFrame({
        "Kolom": ["supermarkt_op_loopafstand", "gebied", "afstandsband",
                  "aantal_supermarkten", "woz_euro"],
        "Hoe gemaakt": [
            "afstand_supermarkt_km <= de grens van de slider (True/False)",
            "stedelijkheid 1-3 = Stad, 4-5 = Platteland",
            "pd.cut van de afstand in 5 banden",
            "pd.cut van het gemiddeld aantal binnen 1 km, afgerond naar 0/1/2/3+",
            "woz_x1000 × 1000",
        ],
        "Waar gebruikt": ["KPI's, tab 1, 2, 4, 5", "overal (stad blauw, platteland groen)",
                          "tab 1 en 3", "tab 3", "tab 1 en 4"],
    }), hide_index=True)

    with st.expander("Keuzes die we hebben gemaakt"):
        st.markdown(
            "- **Stad en platteland:** CBS-stedelijkheid 1 t/m 3 is stad, 4 en 5 is platteland. "
            "Klasse 3 (matig stedelijk) gedroeg zich in de analyse hetzelfde als de stad.\n"
            f"- **Minimaal {MINIMAAL_BUURTEN} buurten** per groep voor een vergelijking over heel "
            f"Nederland, en {MINIMAAL_BUURTEN_GEMEENTE} binnen één gemeente. Bij minder buurten "
            "bepalen een paar uitschieters het getal.\n"
            "- **Logaritme van de WOZ:** zo wordt een verschil een percentage, en tellen dure "
            "en goedkope buurten even zwaar mee.\n"
            "- **Geen invulling van missende waarden:** buurten zonder WOZ of woningtype laten "
            "we weg."
        )


# =====================================================================
# MENU LINKS
# =====================================================================
# Het menu in de zijbalk: elke regel koppelt een titel (en een icoontje) aan een functie hierboven.
menu = st.navigation([
    st.Page(pagina_eerste_blik, title="1 · Eerste blik", icon=":material/visibility:",
            default=True),
    st.Page(pagina_vergelijkbare_buurten, title="2 · Vergelijkbare buurten",
            icon=":material/balance:"),
    st.Page(pagina_afstand, title="3 · Afstand", icon=":material/straighten:"),
    st.Page(pagina_kaart, title="4 · Kaart", icon=":material/map:"),
    st.Page(pagina_conclusie, title="5 · Conclusie", icon=":material/flag:"),
    st.Page(pagina_over_de_data, title="6 · Over de data", icon=":material/database:"),
])
menu.run()          # voert de functie van de gekozen pagina uit