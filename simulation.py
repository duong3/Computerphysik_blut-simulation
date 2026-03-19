import numpy as np
import pyvista as pv
import time
import random
import sys

# -----------------------------------------------------------------------
# Zerebrale Blutbahn-Simulation
#
# Modelliert den Blutfluss durch das zerebrale Gefäßsystem:
# Karotiden / Vertebralarterien → Circulus Willisii → Hirnarterien
# → Kapillarbetten (O2-Abgabe) → venöse Sinus → Jugularvenen
#
# Physik:
#   v = I / (π·r²)       — Kontinuitätsgesetz (Kapillarbetten: 16-38× langsamer)
#   v = MURRAY_K × r     — Murray (1926): dickere Gefäße fließen schneller
#   C·dP/dt = Q(t) - P/R — Windkessel-Modell: arterieller Blutdruck
# -----------------------------------------------------------------------

# --- Simulations-Parameter ---
MAX_CELLS         = 80    # Gesamtzahl Zellen; bei Ruckeln auf 50 reduzieren
SPAWN_RATE        = 3     # Alle N Frames eine neue Zelle spawnen
TIME_STEP         = 0.04  # Simulationszeit pro Frame
ARTERIAL_FRACTION = 1.00  # alle Zellen arteriel – traversieren den kompletten Pfad Arterie→Kapillar→Vene

# Entry-Gefäße
ARTERIAL_ENTRIES = [0, 1, 2, 3]   # Karotiden (0,1) + Vertebralis (2,3)
VENOUS_ENTRIES   = [19]            # Sinus sagittalis superior

# --- Szenario-Presets ---
# (aggregation_prob, aspirin_level)
SCENARIOS = [
    ("Gesund",    0.15, 0.0),   # niedrige Thromboseneigung, kein Aspirin
    ("Rauchen",   0.85, 0.0),   # starke Thromboseneigung (COX-Überaktivierung)
    ("Diabetes",  0.65, 0.0),   # erhöhte Thromboseneigung (Thrombozyten-Hyperreaktivität)
    ("+ Aspirin", 0.50, 0.80),  # Prophylaxe: COX-1-Hemmung, Aggregation stark reduziert
]

# --- 3D-Beschriftungen: alle wichtigen Gefäße ---
VESSEL_LABELS = {
    # Halsgefäße
    0:  "Karotis dx.",
    1:  "Karotis sin.",
    2:  "Vertebralis dx.",
    3:  "Vertebralis sin.",
    # Schädelbasis
    4:  "ICA dx.",
    5:  "ICA sin.",
    8:  "Basilaris",
    # Circulus Willisii / ACA
    9:  "ACA-Bogen dx.",
    10: "ACA-Bogen sin.",
    15: "ACA dist. dx.",
    16: "ACA dist. sin.",
    # MCA
    13: "MCA dx. [Läsion]",
    14: "MCA sin.",
    # PCA
    17: "PCA dx.",
    18: "PCA sin.",
    # Kapillarbetten
    24: "Kap. MCA dx.",
    25: "Kap. MCA sin.",
    26: "Kap. ACA dx.",
    27: "Kap. ACA sin.",
    28: "Kap. PCA dx.",
    29: "Kap. PCA sin.",
    # Venen
    19: "Sinus sagittalis",
    20: "Sinus transv. dx.",
    21: "Sinus transv. sin.",
    22: "Jugularis dx.",
    23: "Jugularis sin.",
}

# Manuelle Ankerpunkte für Labels – seitlich versetzt, damit jedes Label
# eindeutig auf sein Gefäß zeigt (dx=rechts/+x, sin=links/-x).
VESSEL_LABEL_POS = {
    # Hals (unten)
    0:  [ 2.5, -7.0,  0.1],
    1:  [-2.5, -7.0,  0.1],
    2:  [ 1.5, -6.5, -1.0],
    3:  [-1.5, -6.5, -1.0],
    # Schädelbasis
    4:  [ 3.2, -0.5, -0.2],
    5:  [-3.2, -0.5, -0.2],
    8:  [ 0.8,  0.0, -1.1],
    # ACA-Bogen (Circulus Willisii) – dx rechts, sin links
    9:  [ 1.8,  1.6, -0.5],
    10: [-1.8,  1.6, -0.5],
    # MCA – weit nach außen
    13: [ 4.2,  2.2,  0.1],
    14: [-4.2,  2.2,  0.1],
    # ACA dist. – seitlich versetzt damit sie sich nicht überlappen
    15: [ 2.0,  4.5, -0.4],
    16: [-2.0,  4.5, -0.4],
    # PCA – nach hinten-außen
    17: [ 3.2,  0.6, -1.1],
    18: [-3.2,  0.6, -1.1],
    # Kapillarbetten – auf den jeweiligen Tube-Mittelpunkt versetzt
    24: [ 5.8,  5.8,  0.1],
    25: [-5.8,  5.8,  0.1],
    26: [ 1.8,  7.2, -0.2],
    27: [-1.8,  7.2, -0.2],
    28: [ 5.2,  1.5, -0.7],
    29: [-5.2,  1.5, -0.7],
    # Venen
    19: [ 1.2,  5.0, -0.5],
    20: [ 3.5,  0.5, -1.5],
    21: [-3.5,  0.5, -1.5],
    22: [ 3.8, -5.5,  0.3],
    23: [-3.8, -5.5,  0.3],
}

# --- Puls-Parameter ---
PULSE_AMPLITUDE = 0.10   # 10% Radius-Schwingung pro Herzschlag
PULSE_FREQ      = 1.2    # Herzfrequenz in Hz (≈ 72 Schläge/min)
PULSE_VESSELS   = {0, 1, 2, 3, 4, 5, 8}  # IDs der pulsierenden Gefäße

# --- Herz (stilisiert) ---
HEART_CENTER = np.array([0.0, -14.0, 0.0])
HEART_SCALE  = 0.20   # 16×0.20=3.2 Einh. Breite; obere Einbuchtung bei y≈-13

# --- Murray's Gesetz (Geschwindigkeit) ---
# v = MURRAY_K × r  →  dickere Gefäße fließen schneller (Murray 1926)
# Karotis (r=0.35): v = 12.95; ACA dist. (r=0.14): v = 5.18; Kapillaren: I/A-Formel
MURRAY_K = 37.0

# --- Thrombose-Parameter ---
PLATELET_RATIO           = 0.20  # 20 % der arteriellen Zellen sind Thrombozyten
ADHESION_PROB            = 0.80  # Erste Adhäsion an Läsion (aspirin-UNabhängig: GPIb/vWF-Mechanismus)
AGGREGATION_PROB         = 0.50  # Aggregation Plättchen→Thrombus (aspirin-ABHÄNGIG: COX-1/TXA2)
CLOT_GROWTH_PER_PLATELET = 0.006 # Lineares Wachstum pro Plättchen: ~9 nötig für Okklusion.

# --- Windkessel-Modell (arterieller Blutdruck) ---
# 2-Element-Windkessel: C·dP/dt = Q(t) - P/R
#   C  = arterielle Compliance (Gefäßwand als elastischer Speicher) [ml/mmHg]
#   R  = peripherer Widerstand (Kapillaren)                         [mmHg·s/ml]
#   Q(t) = pulsatiler Herzauswurf (positive Halbwelle des Sinus)    [ml/s]
# Numerisch gelöst per Euler-Verfahren, ein Schritt pro Frame.
WK_C  = 0.8    # arterielle Compliance
WK_R  = 0.70   # peripherer Widerstand (Ruhe)
WK_Q0 = 450.0  # Spitzendurchfluss Systole

# --- Fentanyl / Blut-Hirn-Schranke ---
# BHS_CENTER: geometrisches Zentrum des Hirnbereichs (Mittelpunkt aller Kapillarbetten)
BHS_CENTER              = np.array([0.0, 4.5, -0.2])
MAX_FENTANYL_PARTICLES  = 12     # Anzahl sichtbarer Wirkstoff-Partikel im Pool


# Globaler Simulations-Zustand (von UI-Callbacks gelesen/gesetzt)
SIM_STATE = {
    "aspirin_level":        0.0,    # 0 = kein Aspirin, 1 = maximale Hemmung
    "aggregation_prob":     0.50,   # Thromboseneigung-Slider (5b); entspricht AGGREGATION_PROB
    "clear_clot_signal":    False,  # True = Thrombus lösen (nach einem Frame reset)
    "mca_occluded":         False,  # True = MCA vollständig blockiert (5a)
    "lesion_active":        False,  # True = Läsion gesetzt, Thrombose-Check läuft
    "lesion_activate_time": None,   # time.time() beim Aktivieren → 1s Highlight
    "last_platelet_inject": 0.0,    # time.time() der letzten lokalen Thrombozyten-Injektion
    "ruptur_active":        False,  # True = Arterie gerissen (hämorrhagischer Schlaganfall)
    "rueckstau_active":     False,  # True = minimaler Restfluss durch teilweisen Rückstau
    "fentanyl_level":       0.0,    # 0 = kein Fentanyl, 1 = maximale Dosis
    "windkessel_p":         100.0,  # arterieller Druck [mmHg], Startwert nahe Ruhewert
}

# Läsion: A. cerebri media dx. (vessel id=13), Mitte des Gefäßes (t=0.5)
# Häufigste Lokalisation beim ischämischen Schlaganfall
LESION = {
    "vessel_id": 13,
    "t":         0.5,
    "pos_3d":    None,   # wird in setup_plotter() befüllt
}

TUBE_SIDES = 8
SPLINE_PTS = 30

COLOR = {
    "artery":    "firebrick",
    "brain":     "firebrick",    # Hirnarterien = Arterien → gleich rot
    "comm":      "firebrick",    # Circulus Willisii = Arterien → gleich rot
    "extern":    "firebrick",    # A. car. ext. = Arterie → gleich rot
    "capillary": "mediumseagreen",
    "vein":      "royalblue",
    "sinus":     "royalblue",    # Venöse Sinus = Venen → gleich blau
}

# Volumenstrom I pro Gefäß (willkürliche Einheiten)
# Konsistenzregel: I_eltern = Summe(I_kinder)  (Kontinuität an Verzweigungen)
#
# Gesamtfluss arteriell: 8.0
#   Karotiden:    2 × 3.2 = 6.4
#   Vertebralis:  2 × 0.8 = 1.6
#   Gesamt:       8.0
#
# Kapillarbett r_eff so gewählt, dass v_kap ≈ 1/20 * v_arterie:
#   r_eff = sqrt(I / (π * v_ziel))  mit v_ziel ≈ 0.5
#   Für I=1.2: r_eff ≈ 0.87
#   Für I=0.8: r_eff ≈ 0.71
VESSEL_FLOWS = {
    # --- ARTERIEN Hals ---
    0:  3.2,   # A. car. comm. dx.
    1:  3.2,   # A. car. comm. sin.
    2:  0.8,   # A. vertebralis dx.
    3:  0.8,   # A. vertebralis sin.
    # --- Schädelbasis ---
    4:  2.4,   # A. car. int. dx.        (75% der Karotis)
    5:  2.4,   # A. car. int. sin.
    6:  0.8,   # A. car. ext. dx.        (25% der Karotis)
    7:  0.8,   # A. car. ext. sin.
    8:  1.6,   # A. basilaris            (beide Vertebralis)
    # --- Circulus Willisii ---
    9:  1.2,   # A. cer. ant. dx. prox.  (50% der ICA)
    10: 1.2,   # A. cer. ant. sin. prox.
    11: 0.2,   # A. comm. post. dx.      (kollateral)
    12: 0.2,   # A. comm. post. sin.
    # --- Hirnarterien ---
    13: 1.2,   # A. cer. media dx.       (50% der ICA)
    14: 1.2,   # A. cer. media sin.
    15: 1.2,   # A. cer. ant. dx. dist.
    16: 1.2,   # A. cer. ant. sin. dist.
    17: 0.8,   # A. cer. post. dx.       (50% der Basilaris)
    18: 0.8,   # A. cer. post. sin.
    # --- Kapillarbetten ---
    # Repräsentieren den aggregierten Querschnitt aller echten Kapillaren
    # im jeweiligen Hirngebiet. r_eff >> r_arterie → v << v_arterie
    24: 1.2,   # cap_MCA_dx
    25: 1.2,   # cap_MCA_sin
    26: 1.2,   # cap_ACA_dx
    27: 1.2,   # cap_ACA_sin
    28: 0.8,   # cap_PCA_dx
    29: 0.8,   # cap_PCA_sin
    # --- Venöse Sinus (empfängt gesamten Gehirnabfluss: 1.2×4 + 0.8×2 = 6.4) ---
    19: 6.4,   # Sinus sagittalis sup.   (1.2×4 + 0.8×2 = 6.4)
    20: 3.2,   # Sinus transv. dx.
    21: 3.2,   # Sinus transv. sin.
    22: 3.2,   # V. jug. int. dx.
    23: 3.2,   # V. jug. int. sin.
}


# -----------------------------------------------------------------------
# GEOMETRIE-HILFSFUNKTIONEN
# -----------------------------------------------------------------------

def calculate_length(points):
    total = 0.0
    for i in range(len(points) - 1):
        total += np.linalg.norm(points[i + 1] - points[i])
    return total


def get_point_at_t(points, t):
    """3D-Punkt an Parameter t ∈ [0, 1] entlang der Kontrollpunktlinie."""
    if t <= 0:
        return points[0].copy()
    if t >= 1:
        return points[-1].copy()
    total_len = calculate_length(points)
    target_dist = t * total_len
    current_dist = 0.0
    for i in range(len(points) - 1):
        seg_len = np.linalg.norm(points[i + 1] - points[i])
        if current_dist + seg_len >= target_dist:
            local_t = (target_dist - current_dist) / seg_len
            return points[i] + local_t * (points[i + 1] - points[i])
        current_dist += seg_len
    return points[-1].copy()


# -----------------------------------------------------------------------
# GEFÄSSNETZ
# -----------------------------------------------------------------------

def build_vessel_network():
    vessels = []

    def add(vid, vtype, label, pts, radius, next_ids):
        vessels.append({
            "id":       vid,
            "type":     vtype,
            "label":    label,
            "points":   np.array(pts, dtype=float),
            "radius":   radius,
            "color":    COLOR[vtype],
            "next":     next_ids,
            "length":   0.0,
            "flow":     0.0,
            "velocity": 0.0,
        })

    # Hals – große Arterien
    add(0,  "artery", "A. car. comm. dx.",
        [[1.5,-10,0.0],[1.6,-7,0.0],[1.7,-4,0.1],[1.8,-2,0.2]], 0.35, [4, 6])
    add(1,  "artery", "A. car. comm. sin.",
        [[-1.5,-10,0.0],[-1.6,-7,0.0],[-1.7,-4,0.1],[-1.8,-2,0.2]], 0.35, [5, 7])
    add(2,  "artery", "A. vertebralis dx.",
        [[0.8,-10,-1.0],[0.7,-7,-1.0],[0.5,-4,-1.0],[0.3,-1.5,-1.0]], 0.20, [8])
    add(3,  "artery", "A. vertebralis sin.",
        [[-0.8,-10,-1.0],[-0.7,-7,-1.0],[-0.5,-4,-1.0],[-0.3,-1.5,-1.0]], 0.20, [8])

    # Schädelbasis
    add(4,  "artery", "A. car. int. dx.",
        [[1.8,-2,0.2],[2.0,-1,-0.1],[2.0,0,-0.4],[1.8,1,-0.5]], 0.28, [9, 13])
    add(5,  "artery", "A. car. int. sin.",
        [[-1.8,-2,0.2],[-2.0,-1,-0.1],[-2.0,0,-0.4],[-1.8,1,-0.5]], 0.28, [10, 14])
    add(6,  "extern", "A. car. ext. dx.",
        [[1.8,-2,0.2],[2.5,-0.5,0.5],[3.5,1,0.5],[4.2,2.5,0.3]], 0.22, [])
    add(7,  "extern", "A. car. ext. sin.",
        [[-1.8,-2,0.2],[-2.5,-0.5,0.5],[-3.5,1,0.5],[-4.2,2.5,0.3]], 0.22, [])
    add(8,  "artery", "A. basilaris",
        [[0,-1.5,-1.0],[0,-0.5,-1.0],[0,0.5,-1.0],[0,1.5,-1.0]], 0.22, [17, 18])

    # Circulus Willisii
    add(9,  "artery", "A. cer. ant. dx. (prox.)",
        [[1.8,1,-0.5],[1.0,1.8,-0.5],[0.0,2.2,-0.5]], 0.16, [15])
    # Vessel 10: Richtung umgekehrt – startet jetzt bei ICA sin. Ende [-1.8,1,-0.5],
    # läuft zum Bogenscheitel [0,2.2,-0.5]. Kein Sprung mehr von ICA sin. → vessel 10.
    add(10, "artery", "A. cer. ant. sin. (prox.)",
        [[-1.8,1,-0.5],[-1.0,1.8,-0.5],[0.0,2.2,-0.5]], 0.16, [16])
    add(11, "comm",   "A. comm. post. dx.",
        [[1.8,1,-0.5],[1.3,0.8,-0.8],[1.0,1.2,-1.0]], 0.12, [])
    add(12, "comm",   "A. comm. post. sin.",
        [[-1.8,1,-0.5],[-1.3,0.8,-0.8],[-1.0,1.2,-1.0]], 0.12, [])

    # Hirnarterien → leiten in Kapillarbetten weiter
    add(13, "brain", "A. cer. media dx.",
        [[1.8,1,-0.5],[3.0,1.8,-0.2],[4.5,3,0],[5.5,4.5,0.2]], 0.20, [24])
    add(14, "brain", "A. cer. media sin.",
        [[-1.8,1,-0.5],[-3.0,1.8,-0.2],[-4.5,3,0],[-5.5,4.5,0.2]], 0.20, [25])
    # ACA dist. startet am Bogenscheitel [0,2.2,-0.5] (nahtloser Übergang von vessel 9/10)
    add(15, "brain", "A. cer. ant. dx. (dist.)",
        [[0.0,2.2,-0.5],[0.4,3.5,-0.5],[0.6,5.0,-0.4],[0.6,7,-0.3]], 0.14, [26])
    add(16, "brain", "A. cer. ant. sin. (dist.)",
        [[0.0,2.2,-0.5],[-0.4,3.5,-0.5],[-0.6,5.0,-0.4],[-0.6,7,-0.3]], 0.14, [27])
    add(17, "brain", "A. cer. post. dx.",
        [[0,1.5,-1.0],[1.5,1.2,-1.1],[3,0.5,-1.1],[4.5,0,-1.0]], 0.15, [28])
    add(18, "brain", "A. cer. post. sin.",
        [[0,1.5,-1.0],[-1.5,1.2,-1.1],[-3,0.5,-1.1],[-4.5,0,-1.0]], 0.15, [29])

    # Kapillarbetten
    # r_eff = Physik-Gesamtquerschnitt aller Kapillaren im Gebiet (für Geschwindigkeit).
    # Pfad-Endpunkte bei [0,7.5,-0.3] = Eingang Sinus sagittalis sup. (id=19),
    # nahtloser Übergang in den Sinus-Tube (entspricht kortikalen Drainagevenen).
    add(24, "capillary", "Kap.-Bett MCA dx.",
        [[5.5,4.5,0.2],[6.0,5.5,0.0],[4.5,6.5,-0.15],[2.0,7.2,-0.25],[0,7.5,-0.3]], 0.87, [19])
    add(25, "capillary", "Kap.-Bett MCA sin.",
        [[-5.5,4.5,0.2],[-6.0,5.5,0.0],[-4.5,6.5,-0.15],[-2.0,7.2,-0.25],[0,7.5,-0.3]], 0.87, [19])
    add(26, "capillary", "Kap.-Bett ACA dx.",
        [[0.6,7,-0.3],[0.3,7.3,-0.3],[0,7.5,-0.3]], 0.87, [19])
    add(27, "capillary", "Kap.-Bett ACA sin.",
        [[-0.6,7,-0.3],[-0.3,7.3,-0.3],[0,7.5,-0.3]], 0.87, [19])
    # PCA endet nach rechts/links-unten ([±4.5,0,-1.0]).
    # Erster Kapillar-Punkt setzt diese Richtung fort (±5.7,-0.5,-0.85),
    # dann biegt die Kurve sanft nach oben ein → gleichmäßige Kurve ohne Knick.
    add(28, "capillary", "Kap.-Bett PCA dx.",
        [[4.5,0,-1.0],[5.7,-0.5,-0.85],[5.0,1.5,-0.7],[3.0,4.0,-0.4],[0,7.5,-0.3]], 0.71, [19])
    add(29, "capillary", "Kap.-Bett PCA sin.",
        [[-4.5,0,-1.0],[-5.7,-0.5,-0.85],[-5.0,1.5,-0.7],[-3.0,4.0,-0.4],[0,7.5,-0.3]], 0.71, [19])

    # Venöse Sinus
    # Radien vergrößert: Sinus sind weite Kanäle, die viel Blut langsam drainieren.
    # r=0.55 für Sinus sagittalis → v≈6.7 (statt v=26 mit r=0.28)
    add(19, "sinus", "Sinus sagittalis sup.",
        [[0,7.5,-0.3],[0,5.5,-0.5],[0,3,-0.7],[0,1.5,-1.0]], 0.55, [20, 21])
    add(20, "sinus", "Sinus transv. dx.",
        [[0,1.5,-1.0],[1.5,0.8,-1.5],[3.2,0,-1.4],[3,-1.5,0.3]], 0.40, [22])
    add(21, "sinus", "Sinus transv. sin.",
        [[0,1.5,-1.0],[-1.5,0.8,-1.5],[-3.2,0,-1.4],[-3,-1.5,0.3]], 0.40, [23])

    # V. jugularis interna
    add(22, "vein", "V. jug. int. dx.",
        [[3,-1.5,0.3],[2.8,-4,0.4],[2.7,-7,0.3],[2.7,-10,0.3]], 0.38, [])
    add(23, "vein", "V. jug. int. sin.",
        [[-3,-1.5,0.3],[-2.8,-4,0.4],[-2.7,-7,0.3],[-2.7,-10,0.3]], 0.38, [])

    # Länge + Physik + Spline-Pfad berechnen
    # spline_pts = geglätteter Pfad (SPLINE_PTS Punkte), identisch mit Tube-Rendering.
    # Zellen folgen spline_pts → exakte Übereinstimmung mit dem gerenderten Rohr.
    #
    # Geschwindigkeit:
    #   Arterien/Venen: Murray's Gesetz  v = MURRAY_K × r
    #     → dickere Gefäße fließen schneller (physiologisch korrekt; Murray 1926)
    #   Kapillarbetten: v = I / (π r²) mit r_eff >> r_arterie → sehr langsam
    for v in vessels:
        spline         = pv.Spline(v["points"], SPLINE_PTS)
        v["spline_pts"] = spline.points          # Nx3 ndarray
        v["length"]    = calculate_length(v["spline_pts"])
        I              = VESSEL_FLOWS.get(v["id"], 0.5)
        v["flow"]      = I
        if v["type"] == "capillary":
            v["velocity"] = I / (np.pi * v["radius"] ** 2)   # I/A für aggregierten Querschnitt
        else:
            v["velocity"] = MURRAY_K * v["radius"]            # Murray: v ∝ r

    return vessels


# -----------------------------------------------------------------------
# BLUTZELLEN
# -----------------------------------------------------------------------

class BloodCell:
    """
    Bewegt sich entlang des Gefäßgraphen.

    Bewegung:    Δprogress = velocity * dt / length  pro Frame
    Verzweigung: gewichtete Zufallswahl nach vessel["flow"]
    Thrombose:   Thrombozyten kleben bei Kontakt mit Läsion oder anderen
                 klebenden Plättchen (modelliert durch SIM_STATE + BASE_CLOT_PROB)
    """
    CELL_RADIUS = 0.13
    _CLOT_RES   = 16   # Sphere-Auflösung für Thrombus-Mesh (muss überall identisch sein)

    def __init__(self, vid_map, plotter, entry_id, cell_type, is_platelet=False):
        self.vid_map     = vid_map
        self.entry_id    = entry_id
        self.cell_type   = cell_type      # "arterial" | "venous"
        self.is_platelet = is_platelet
        self.is_stuck    = False
        self.current_id  = entry_id
        self.progress    = 0.0

        start_pos = vid_map[entry_id]["points"][0].copy()
        self.pos  = start_pos
        self.mesh = pv.Sphere(radius=self.CELL_RADIUS, center=start_pos)
        self.actor = plotter.add_mesh(self.mesh, color=self._base_color(), render=False)

    def _base_color(self):
        if self.is_platelet:
            return "white"
        return "crimson" if self.cell_type == "arterial" else "royalblue"

    def _move_mesh(self, new_pos):
        centroid = np.mean(self.mesh.points, axis=0)
        self.mesh.points = self.mesh.points - centroid + new_pos
        self.pos = new_pos

    def update(self, dt, all_cells):
        # Thrombus lösen auf Signal
        if self.is_stuck and SIM_STATE["clear_clot_signal"]:
            self.is_stuck = False
            self.actor.prop.color = "white"

        if self.is_stuck:
            return

        # MCA okkludiert → Zelle sofort respawnen
        # Ausnahme: Rückstau aktiv → Zellen dürfen langsam durchsickern
        if (self.current_id == LESION["vessel_id"]
                and SIM_STATE["mca_occluded"]
                and not SIM_STATE["rueckstau_active"]):
            self.respawn()
            return

        vessel = self.vid_map[self.current_id]
        dp = vessel["velocity"] * dt / vessel["length"]
        # Rückstau: Zellen in der MCA fließen im Schneckentempo
        if SIM_STATE["rueckstau_active"] and self.current_id == LESION["vessel_id"]:
            dp *= 0.08
        self.progress += dp

        if self.progress >= 1.0:
            valid_next = [
                nid for nid in vessel["next"]
                if nid in self.vid_map
                and self.vid_map[nid]["type"] != "extern"   # A. car. ext. = Gesicht, nicht Gehirn
                and not (SIM_STATE["mca_occluded"]
                         and not SIM_STATE["rueckstau_active"]
                         and nid == LESION["vessel_id"])
            ]
            if valid_next:
                weights = np.array([self.vid_map[nid]["flow"] for nid in valid_next], dtype=float)
                weights /= weights.sum()
                self.current_id = int(np.random.choice(valid_next, p=weights))
                self.progress   = 0.0
            else:
                self.respawn()
                return

        vessel  = self.vid_map[self.current_id]
        new_pos = get_point_at_t(vessel["spline_pts"], self.progress)
        self._move_mesh(new_pos)

        if not self.is_platelet:
            self._update_color(vessel["type"])

        # Thrombose nur wenn Läsion aktiv + Thrombozyt im Läsions-Gefäß
        if self.is_platelet and self.current_id == LESION["vessel_id"] \
                and SIM_STATE["lesion_active"]:
            self._thrombosis_check(new_pos, all_cells)

    def _thrombosis_check(self, pos, all_cells):
        stuck = [c for c in all_cells if c.is_stuck]
        n_stuck = len(stuck)

        if n_stuck == 0:
            # --- Pfad 1: Primäre Adhäsion an freiliegender Läsion ---
            # Mechanismus: GPIb/vWF-Bindung an freiliegendes Kollagen
            # Aspirin-UNABHÄNGIG (COX-1-Hemmung betrifft diesen Pfad nicht)
            # Wird nur geprüft, solange KEIN Thrombus existiert (Läsion noch "offen")
            if np.linalg.norm(pos - LESION["pos_3d"]) < (0.15 + self.CELL_RADIUS):
                if random.random() < ADHESION_PROB:
                    self.is_stuck = True
                    self.actor.prop.color = "yellow"
        else:
            # --- Pfad 2: Aggregation an bestehendem Thrombus ---
            # Läsion gilt ab jetzt als "bedeckt" → nur noch Plättchen→Thrombus-Kontakt
            # Mechanismus: TXA2/ADP → Aspirin hemmt COX-1 → reduziert TXA2
            # prob = AGGREGATION_PROB * (1 − aspirin_level)
            clot_center = np.mean([c.pos for c in stuck], axis=0)
            clot_radius = 0.15 + CLOT_GROWTH_PER_PLATELET * n_stuck
            if np.linalg.norm(pos - clot_center) < clot_radius + self.CELL_RADIUS:
                prob = SIM_STATE["aggregation_prob"] * (1.0 - SIM_STATE["aspirin_level"])
                if random.random() < prob:
                    self.is_stuck = True
                    self.actor.prop.color = "yellow"

    def _update_color(self, vtype):
        # Farbe richtet sich nach Gefäßtyp (nicht cell_type):
        # Arterie = sauerstoffreich (rot), Kapillar = Übergang (Rot→Blau),
        # Vene/Sinus = sauerstoffarm (blau)
        if vtype in ("artery", "brain", "comm", "extern"):
            self.actor.prop.color = "crimson"
        elif vtype == "capillary":
            # O2-Diffusion: Gradient Rot→Blau basierend auf Fortschritt im Kapillarbett
            # progress=0 → arterielles Ende (100% O2), progress=1 → venöses Ende (60% O2)
            t = self.progress
            r = (220 * (1 - t) + 65  * t) / 255   # crimson.r → royalblue.r
            g = (20  * (1 - t) + 105 * t) / 255
            b = (60  * (1 - t) + 225 * t) / 255
            self.actor.prop.color = (r, g, b)
        elif vtype == "sinus":
            self.actor.prop.color = "steelblue"
        else:  # vein
            self.actor.prop.color = "royalblue"

    def respawn(self):
        self.current_id = self.entry_id
        self.progress   = 0.0
        self.is_stuck   = False
        start_pos = self.vid_map[self.entry_id]["points"][0].copy()
        self._move_mesh(start_pos)
        self.actor.prop.color = self._base_color()


# -----------------------------------------------------------------------
# FENTANYL-PARTIKEL
# -----------------------------------------------------------------------

class FentanylParticle:
    """
    Kleine goldene Kugeln, die den Fentanyl-Wirkstoff im Blut repräsentieren.
    Traversieren arterielle Gefäße; an Kapillarbetten diffundieren sie durch
    die BHS (Blut-Hirn-Schranke) ins Gewebe und verblassen dabei.
    """
    RADIUS = 0.08

    def __init__(self, vid_map, plotter, entry_id):
        self.vid_map     = vid_map
        self.entry_id    = entry_id
        self.current_id  = entry_id
        self.progress    = random.random()   # zufällige Startposition im Gefäß
        self.diffusing   = False
        self.diffuse_t   = 0.0
        self.diffuse_pos = None
        self.diffuse_dir = None

        start_pos = get_point_at_t(vid_map[entry_id]["spline_pts"], self.progress)
        self.mesh  = pv.Sphere(radius=self.RADIUS, center=start_pos)
        self.actor = plotter.add_mesh(self.mesh, color="gold", opacity=0.0,
                                      smooth_shading=True, render=False)

    def _move_mesh(self, new_pos):
        centroid = np.mean(self.mesh.points, axis=0)
        self.mesh.points = self.mesh.points - centroid + new_pos

    def update(self, dt):
        fent = SIM_STATE["fentanyl_level"]
        if fent <= 0:
            self.actor.prop.opacity = 0.0
            return

        if self.diffusing:
            self.diffuse_t += dt
            dur  = 1.2   # Diffusions-Dauer: 1.2 s (gut sichtbar)
            frac = self.diffuse_t / dur
            opacity = max(0.0, 0.95 * (1.0 - frac)) * fent
            new_pos = self.diffuse_pos + self.diffuse_dir * frac * 2.5
            self._move_mesh(new_pos)
            self.actor.prop.opacity = opacity
            if self.diffuse_t >= dur:
                self.diffusing = False
                self.respawn()
            return

        vessel = self.vid_map[self.current_id]
        dp = vessel["velocity"] * dt / vessel["length"]
        self.progress += dp

        if self.progress >= 1.0:
            valid_next = [
                nid for nid in vessel["next"]
                if nid in self.vid_map
                and self.vid_map[nid]["type"] != "extern"
            ]
            if valid_next:
                weights = np.array([self.vid_map[nid]["flow"] for nid in valid_next], dtype=float)
                weights /= weights.sum()
                next_id = int(np.random.choice(valid_next, p=weights))
                if self.vid_map[next_id]["type"] == "capillary":
                    # BHS-Diffusion: Partikel wandert vom Arterienende ins Kapillarbett
                    # Richtung = Arterienende → Kapillarbett-Mittelpunkt (= Hirngewebe)
                    entry_pos  = get_point_at_t(vessel["spline_pts"], 1.0).copy()
                    cap_center = self.vid_map[next_id]["points"].mean(axis=0)
                    inward     = cap_center - entry_pos
                    norm_len   = np.linalg.norm(inward)
                    self.diffuse_dir = (inward / norm_len
                                       if norm_len > 0.01 else np.array([0.0, 1.0, 0.0]))
                    self.diffuse_pos = entry_pos.copy()
                    self.diffuse_t   = 0.0
                    self.diffusing   = True
                    self._move_mesh(entry_pos)
                    self.actor.prop.opacity = 0.95 * fent
                    return
                self.current_id = next_id
                self.progress   = 0.0
            else:
                self.respawn()
                return

        vessel  = self.vid_map[self.current_id]
        new_pos = get_point_at_t(vessel["spline_pts"], self.progress)
        self._move_mesh(new_pos)
        self.actor.prop.opacity = 0.75 * fent

    def respawn(self):
        self.current_id = self.entry_id
        self.progress   = random.random() * 0.3
        self.diffusing  = False
        pos = get_point_at_t(self.vid_map[self.entry_id]["spline_pts"], self.progress)
        self._move_mesh(pos)
        self.actor.prop.opacity = 0.0


# -----------------------------------------------------------------------
# VISUALISIERUNG
# -----------------------------------------------------------------------

def build_anatomy_outline():
    head = pv.Sphere(radius=1.0, theta_resolution=14, phi_resolution=14)
    head.points = head.points * np.array([5.5, 5.8, 4.5])
    head.points += np.array([0.0, 3.5, 0.0])
    neck = pv.Cylinder(center=(0.0, -6.0, 0.0), direction=(0, 1, 0),
                       radius=2.5, height=8.0, resolution=14)
    return head, neck


def build_heart_mesh():
    """Anatomisch-stilisiertes Herz aus 3 überlappenden Ellipsoiden.

    Aufbau (Frontansicht von vorne):
      - Ventrikel: schräger Ellipsoid, Apex zeigt nach links-unten
      - Rechter Vorhof: größere Kugel oben-rechts
      - Linker Vorhof: kleinere Kugel oben-links (etwas weiter dorsal)
    Gibt zusammengeführtes pv.PolyData zurück (zum Pulsieren im Loop).
    """
    res = 20   # Auflösung Sphere/Ellipsoid

    # --- Ventrikel-Ellipsoid (Hauptkörper) ---
    # Breite 1.4×, Höhe 2.1×, Tiefe 0.9× → queroval-kegelförmig
    vent = pv.Sphere(radius=1.0, theta_resolution=res, phi_resolution=res)
    pts  = vent.points.copy()
    # Ellipsoid-Skalierung
    pts[:, 0] *= 1.4
    pts[:, 1] *= 2.1
    pts[:, 2] *= 0.9
    # Neigung: Apex nach links-unten (Rotation um Z-Achse, –18°)
    a = np.radians(-18)
    c, s = np.cos(a), np.sin(a)
    px = pts[:, 0]*c - pts[:, 1]*s
    py = pts[:, 0]*s + pts[:, 1]*c
    pts[:, 0], pts[:, 1] = px, py
    vent.points = pts + np.array([-0.15, -14.3, 0.0])   # Zentrum leicht links

    # --- Rechter Vorhof (oben-rechts, größer) ---
    ra = pv.Sphere(radius=0.70, theta_resolution=res, phi_resolution=res)
    ra.points = ra.points * np.array([1.0, 1.0, 0.82])
    ra.points += np.array([0.85, -12.4, 0.10])

    # --- Linker Vorhof (oben-links, kleiner, leicht dorsal) ---
    la = pv.Sphere(radius=0.54, theta_resolution=res, phi_resolution=res)
    la.points = la.points * np.array([1.0, 1.0, 0.75])
    la.points += np.array([-0.45, -12.6, -0.05])

    return vent.merge(ra).merge(la)


def _set_aspirin(value):
    SIM_STATE["aspirin_level"] = value

def _set_aggregation_prob(value):
    SIM_STATE["aggregation_prob"] = value

def _clear_clot(state):
    if state:
        SIM_STATE["clear_clot_signal"] = True

def _set_fentanyl(value):
    SIM_STATE["fentanyl_level"] = value


def setup_plotter(vessels):
    """
    Baut die statische Szene auf.
    Befüllt LESION["pos_3d"] und gibt
    (plotter, clot_mesh, clot_actor, mca_actor, stroke_actor) zurück.
    """
    vid_map = {v["id"]: v for v in vessels}

    # Läsions-Position berechnen und global speichern
    lesion_vessel = vid_map[LESION["vessel_id"]]
    LESION["pos_3d"] = get_point_at_t(lesion_vessel["spline_pts"], LESION["t"])

    head_mesh, neck_mesh = build_anatomy_outline()
    plotter = pv.Plotter()
    plotter.set_background("#0d0d1a")
    plotter.add_mesh(head_mesh, color="#aaaacc", opacity=0.07, style="wireframe")
    plotter.add_mesh(neck_mesh, color="#aaaacc", opacity=0.07, style="wireframe")

    # Herz: anatomisch-stilisierte 3D-Form (Ventrikel + 2 Vorhöfe).
    # Synchroner Puls mit Halsarterien (Basispunkte gespeichert, im Loop skaliert).
    heart_mesh      = build_heart_mesh()
    heart_base_pts  = heart_mesh.points.copy()
    plotter.add_mesh(heart_mesh, color="#8B0000", opacity=0.80, smooth_shading=True)

    # Verbindungslinien Herz ↔ Halsgefäße
    # _hy = y-Höhe der oberen Herzeinbuchtung (t=0 → hy=5 in Formel-Einheiten)
    _hy = HEART_CENTER[1] + HEART_SCALE * 5   # ≈ -13.0
    _art_conn = [                              # (Herzpunkt, Arterienstartpunkt)
        ([HEART_CENTER[0]+0.8, _hy, 0.0],   [1.5, -10, 0.0]),    # Karotis dx
        ([HEART_CENTER[0]-0.8, _hy, 0.0],   [-1.5, -10, 0.0]),   # Karotis sin
        ([HEART_CENTER[0]+0.3, _hy, -0.3],  [0.8,  -10, -1.0]),  # Vertebralis dx
        ([HEART_CENTER[0]-0.3, _hy, -0.3],  [-0.8, -10, -1.0]),  # Vertebralis sin
    ]
    _ven_conn = [                              # (Herzpunkt, Venenendpunkt)
        ([HEART_CENTER[0]+1.5, _hy-0.3, 0.1],  [2.7,  -10, 0.3]),  # Jugularis dx
        ([HEART_CENTER[0]-1.5, _hy-0.3, 0.1],  [-2.7, -10, 0.3]),  # Jugularis sin
    ]
    for h_pt, v_pt in _art_conn:
        _t = pv.Spline(np.array([h_pt, v_pt]), 10).tube(radius=0.06, n_sides=6)
        plotter.add_mesh(_t, color="crimson", opacity=0.50, smooth_shading=True)
    for h_pt, v_pt in _ven_conn:
        _t = pv.Spline(np.array([h_pt, v_pt]), 10).tube(radius=0.06, n_sides=6)
        plotter.add_mesh(_t, color="royalblue", opacity=0.50, smooth_shading=True)
    # Label unter der Herzspitze (hy_min ≈ -17 in Formel → -14 + 0.20×(-17) ≈ -17.4)
    plotter.add_point_labels(
        [HEART_CENTER + np.array([0, HEART_SCALE * (-17) - 0.4, 0])],
        ["Herz"], font_size=9, text_color="white", shape_opacity=0.0, show_points=False
    )

    # MCA-Actor (vessel 13) separat speichern, damit Farbe bei Okklusion geändert werden kann
    # Tube-Meshes der pulsierenden Gefäße speichern, damit Punkte im Loop überschrieben werden
    mca_actor   = None
    pulse_meshes = {}   # vid → PolyData-Mesh
    for v in vessels:
        spline = pv.Spline(v["points"], SPLINE_PTS)
        if v["type"] == "capillary":
            # Kapillarpfade: dünne Tubes als Verbindung sichtbar machen.
            # r["radius"]=0.87 ist Physik-Wert; visuell 0.10 (Kompromiss: sichtbar, aber dünner als Arterien ~0.14–0.20).
            tube = spline.tube(radius=0.10, n_sides=TUBE_SIDES)
            plotter.add_mesh(tube, color="mediumseagreen", opacity=0.25, smooth_shading=True)
            continue
        tube   = spline.tube(radius=v["radius"], n_sides=TUBE_SIDES)
        # "extern" + "comm": sehr transparent – Anatomie-Kontext, kein Hauptpfad
        opacity = 0.10 if v["type"] in ("extern", "comm") else 0.35
        actor  = plotter.add_mesh(tube, color=v["color"], opacity=opacity, smooth_shading=True)
        if v["id"] == LESION["vessel_id"]:
            mca_actor = actor
            pulse_meshes[v["id"]] = tube   # MCA: dynamischer Druck-Puls bei Thrombose
        if v["id"] in PULSE_VESSELS:
            pulse_meshes[v["id"]] = tube

    # Kapillarbetten als halbtransparente grüne Kugeln
    # Repräsentieren das Hirngewebe, das von diesem Kapillarnetz versorgt wird.
    # Radius 0.8 = ungefähre räumliche Ausdehnung des Versorgungsgebiets.
    # cap24_actor (MCA dx.) wird bei Okklusion ebenfalls grau – Infarktzone.
    cap24_actor = None
    for v in vessels:
        if v["type"] != "capillary":
            continue
        center = v["points"].mean(axis=0)
        sphere = pv.Sphere(radius=0.8, center=center,
                           theta_resolution=14, phi_resolution=14)
        actor = plotter.add_mesh(sphere, color="mediumseagreen",
                                 opacity=0.15, smooth_shading=True)
        if v["id"] == 24:
            cap24_actor = actor

    # Läsions-Marker (schwarzer Drahtrahmen) – startet unsichtbar, wird per Button aktiviert
    lesion_marker_actor = plotter.add_mesh(
        pv.Sphere(radius=0.15, center=LESION["pos_3d"]),
        color="black", style="wireframe", line_width=2
    )
    lesion_marker_actor.VisibilityOff()

    # Thrombus-Mesh (startet unsichtbar, wächst dynamisch)
    _CLOT_RES  = BloodCell._CLOT_RES
    clot_mesh  = pv.Sphere(radius=0.001, center=LESION["pos_3d"],
                           theta_resolution=_CLOT_RES, phi_resolution=_CLOT_RES)
    clot_actor = plotter.add_mesh(clot_mesh, color="#8B0000", opacity=0.0, smooth_shading=True)

    # Stroke-Warnung – links oben, über dem Fortschrittsbalken
    stroke_actor = plotter.add_text(
        "ISCHÄMIE: A. cer. media verschlossen!",
        position=(0, 0), font_size=10, color="red"
    )
    stroke_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    stroke_actor.GetPositionCoordinate().SetValue(0.01, 0.82)
    stroke_actor.GetTextProperty().SetJustificationToLeft()
    stroke_actor.VisibilityOff()

    # UI – Slider (Widgets speichern für programmatische Updates durch Szenario-Buttons)
    aspirin_slider = plotter.add_slider_widget(
        callback=_set_aspirin, rng=[0, 1], value=0,
        title="Aspirin Dosis",
        pointa=(0.02, 0.32), pointb=(0.30, 0.32), style="modern"
    )
    agg_slider = plotter.add_slider_widget(
        callback=_set_aggregation_prob, rng=[0, 1], value=0.5,
        title="Thromboseneigung",
        pointa=(0.02, 0.44), pointb=(0.30, 0.44), style="modern"
    )
    plotter.add_slider_widget(
        callback=_set_fentanyl, rng=[0, 1], value=0,
        title="Fentanyl Dosis",
        pointa=(0.02, 0.56), pointb=(0.30, 0.56), style="modern"
    )

    # 3s-Highlight bei Läsions-Aktivierung: große gelbe Gitterkugel + Text
    lesion_highlight_actor = plotter.add_mesh(
        pv.Sphere(radius=0.55, center=LESION["pos_3d"],
                  theta_resolution=14, phi_resolution=14),
        color="yellow", style="wireframe", line_width=4
    )
    lesion_highlight_actor.VisibilityOff()

    lesion_hint_actor = plotter.add_text(
        ">>> LÄSION: A. cer. media dx. <<<",
        position="upper_right", font_size=13, color="yellow"
    )
    lesion_hint_actor.VisibilityOff()

    # Fortschrittsanzeige: einfacher Text-Actor (kein Widget → kein VTK-Crash).
    # Wird im Loop via actor.SetInput() aktualisiert; nur sichtbar wenn Läsion aktiv.
    progress_actor = plotter.add_text(
        "", position=(0, 0), font_size=9, color="orange"
    )
    # Links oben, unter ISCHÄMIE-Warnung
    progress_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    progress_actor.GetPositionCoordinate().SetValue(0.01, 0.77)
    progress_actor.GetTextProperty().SetJustificationToLeft()
    progress_actor.VisibilityOff()

    # UI – Läsion setzen (Toggle: an = Läsion aktiv + Marker sichtbar)
    def _toggle_lesion(state):
        SIM_STATE["lesion_active"] = state
        if state:
            lesion_marker_actor.VisibilityOn()
            lesion_highlight_actor.VisibilityOn()
            lesion_hint_actor.VisibilityOn()
            progress_actor.SetInput("Thrombose: [                    ] 0%")
            progress_actor.VisibilityOn()
            SIM_STATE["lesion_activate_time"] = time.time()
        else:
            lesion_marker_actor.VisibilityOff()
            lesion_highlight_actor.VisibilityOff()
            lesion_hint_actor.VisibilityOff()
            progress_actor.VisibilityOff()
            SIM_STATE["clear_clot_signal"] = True
            SIM_STATE["lesion_activate_time"] = None

    plotter.add_checkbox_button_widget(
        _toggle_lesion, value=False, position=(10, 200), size=30,
        border_size=2, color_on="red", color_off="grey", background_color="white"
    )
    plotter.add_text("Läsion setzen", position=(50, 208), font_size=9, color="white")

    # UI – Thrombus lösen
    plotter.add_checkbox_button_widget(
        _clear_clot, value=False, position=(10, 10), size=30,
        border_size=2, color_on="green", color_off="grey", background_color="white"
    )
    plotter.add_text("Thrombus lösen", position=(50, 18), font_size=9, color="white")

    # UI – Szenario-Buttons
    # Jeder Button setzt aggregation_prob + aspirin_level und aktualisiert Slider-Position.
    # _make_scenario erzeugt Closure über die Widget-Objekte.
    _DEFAULT_AGG = AGGREGATION_PROB   # 0.50
    _DEFAULT_ASP = 0.0

    def _make_scenario(agg, asp):
        def cb(state):
            if state:
                SIM_STATE["aggregation_prob"] = agg
                SIM_STATE["aspirin_level"]    = asp
                agg_slider.GetRepresentation().SetValue(agg)
                aspirin_slider.GetRepresentation().SetValue(asp)
            else:
                SIM_STATE["aggregation_prob"] = _DEFAULT_AGG
                SIM_STATE["aspirin_level"]    = _DEFAULT_ASP
                agg_slider.GetRepresentation().SetValue(_DEFAULT_AGG)
                aspirin_slider.GetRepresentation().SetValue(_DEFAULT_ASP)
        return cb

    plotter.add_text("Szenarien:", position=(10, 52), font_size=8, color="lightyellow")
    y = 68
    for label, agg, asp in SCENARIOS:
        plotter.add_checkbox_button_widget(
            _make_scenario(agg, asp),
            value=False, position=(10, y), size=22,
            border_size=1, color_on="orange", color_off="grey", background_color="white"
        )
        plotter.add_text(label, position=(37, y + 5), font_size=8, color="white")
        y += 30

    # 3D-Beschriftungen: zwei Gruppen – Minimal (6 Schlüsselgefäße) + Erweitert (alle)
    # Taste 'L' schaltet zwischen 3 Modi: keine → minimal → alle
    _MINIMAL_IDS = {0, 1, 8, 13, 19, 22}  # Karotis dx/sin, Basilaris, MCA dx, Sinus, Jugularis dx

    # t-Parameter auf Spline (0=Anfang, 1=Ende): Punkt landet exakt auf dem Rohr,
    # gewählt an der Stelle, wo das Gefäß am wenigsten mit Nachbarn überlappt.
    _LABEL_T = {
        0:  0.40,  1:  0.40,   # Karotis dx/sin: Hals-Mitte
        2:  0.50,  3:  0.50,   # Vertebralis dx/sin
        4:  0.40,  5:  0.40,   # ICA dx/sin
        8:  0.60,              # Basilaris: oberes Drittel, weg vom Vertebralis-Zusammenfluss
        9:  0.50,  10: 0.50,   # ACA prox dx/sin
        13: 0.45,  14: 0.45,   # MCA dx/sin: freies Hemisphären-Segment
        15: 0.55,  16: 0.55,   # ACA dist dx/sin
        17: 0.50,  18: 0.50,   # PCA dx/sin
        19: 0.30,              # Sinus sagittalis: obere Hälfte, weg von Kapillar-Mündungen
        20: 0.50,  21: 0.50,   # Sinus transv dx/sin
        22: 0.45,  23: 0.45,   # Jugularis dx/sin: Hals-Mitte
        24: 0.40,  25: 0.40,   # Kap MCA dx/sin
        26: 0.30,  27: 0.30,   # Kap ACA dx/sin
        28: 0.40,  29: 0.40,   # Kap PCA dx/sin
    }

    min_pts, min_names = [], []
    ext_pts, ext_names = [], []
    for v in vessels:
        if v["id"] not in VESSEL_LABELS:
            continue
        t_val = _LABEL_T.get(v["id"], 0.5)
        pos   = get_point_at_t(v["spline_pts"], t_val).tolist()
        name  = VESSEL_LABELS[v["id"]]
        if v["id"] in _MINIMAL_IDS:
            min_pts.append(pos)
            min_names.append(name)
        else:
            ext_pts.append(pos)
            ext_names.append(name)

    labels_min = plotter.add_point_labels(
        np.array(min_pts, dtype=float), min_names,
        font_size=12, text_color="white",
        shape_opacity=0.0, show_points=False,
        always_visible=True,
    )
    labels_ext = plotter.add_point_labels(
        np.array(ext_pts, dtype=float), ext_names,
        font_size=9, text_color="#cccccc",
        shape_opacity=0.0, show_points=False,
        always_visible=True,
    )
    # Punkt-Marker separat als eigene Actors (show_points=False oben),
    # damit Visibility synchron mit den Labels gesteuert werden kann
    pts_min_actor = plotter.add_mesh(
        pv.PolyData(np.array(min_pts, dtype=float)),
        color="white", point_size=7, render_points_as_spheres=True,
    )
    pts_ext_actor = plotter.add_mesh(
        pv.PolyData(np.array(ext_pts, dtype=float)),
        color="#cccccc", point_size=5, render_points_as_spheres=True,
    )
    labels_ext.VisibilityOff()    # Start: nur Minimal sichtbar
    pts_ext_actor.VisibilityOff()

    _lmode = [1]   # 0=keine, 1=minimal, 2=alle
    def _cycle_labels():
        _lmode[0] = (_lmode[0] + 1) % 3
        if _lmode[0] == 0:
            labels_min.VisibilityOff()
            labels_ext.VisibilityOff()
            pts_min_actor.VisibilityOff()
            pts_ext_actor.VisibilityOff()
        elif _lmode[0] == 1:
            labels_min.VisibilityOn()
            labels_ext.VisibilityOff()
            pts_min_actor.VisibilityOn()
            pts_ext_actor.VisibilityOff()
        else:
            labels_min.VisibilityOn()
            labels_ext.VisibilityOn()
            pts_min_actor.VisibilityOn()
            pts_ext_actor.VisibilityOn()
    plotter.add_key_event('l', _cycle_labels)

    # Gefäss-Info-Panel (linke Seite, per Button einblendbar)
    _INFO_TEXT = (
        "GEFÄß-ÜBERSICHT\n"
        "-------------------\n"
        "ARTERIEN  (O2-reich, vom Herz)\n"
        "  Karotis dx./sin.     Hauptarterie Hals\n"
        "  Vertebralis dx./sin. Wirbelarterie\n"
        "  ICA dx./sin.         innere Karotis\n"
        "  Basilaris            Hirnstamm\n"
        "  MCA dx./sin.         Grosshirnrinde\n"
        "  ACA dx./sin.         Stirnlappen\n"
        "  PCA dx./sin.         Hinterhauptlappen\n"
        "\n"
        "KAPILLAREN  (gruene Rohre)\n"
        "  verbinden Arterien mit Kapillarbetten\n"
        "\n"
        "KAPILLARBETTEN  (grüne Kugeln)\n"
        "  stehen für das versorgte Hirngewebe\n"
        "\n"
        "WARUM wird das Blut in den Kapillaren blau?\n"
        "  Arterien transportieren O2-reiches Blut\n"
        "  (rot) ins Gehirn. In den Kapillaren gibt\n"
        "  das Blut den Sauerstoff an die Hirnzellen\n"
        "  ab. Deshalb wird es O2-arm (blau) und\n"
        "  fließt als venöses Blut zurück zum Herz.\n"
        "\n"
        "VENEN  (O2-arm, zum Herz hin)\n"
        "  Sinus sagittalis     Längsblutleiter\n"
        "  Sinus transversus    Querblutleiter\n"
        "  Jugularis dx./sin.   Halsvene"
    )
    info_panel_actor = plotter.add_text(
        _INFO_TEXT, position=(0, 0), font_size=9, color="white"
    )
    info_panel_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    info_panel_actor.GetPositionCoordinate().SetValue(0.01, 0.97)
    info_panel_actor.GetTextProperty().SetJustificationToLeft()
    info_panel_actor.GetTextProperty().SetVerticalJustificationToTop()
    info_panel_actor.GetTextProperty().SetBackgroundColor(0.04, 0.04, 0.12)
    info_panel_actor.GetTextProperty().SetBackgroundOpacity(0.88)
    info_panel_actor.GetTextProperty().SetFrame(True)
    info_panel_actor.GetTextProperty().SetFrameColor(0.40, 0.40, 0.70)
    info_panel_actor.GetTextProperty().SetFrameWidth(2)
    info_panel_actor.VisibilityOff()

    def _toggle_info(state):
        if state:
            info_panel_actor.VisibilityOn()
        else:
            info_panel_actor.VisibilityOff()

    plotter.add_checkbox_button_widget(
        _toggle_info, value=False, position=(10, 240), size=22,
        border_size=1, color_on="lightyellow", color_off="grey", background_color="white"
    )
    plotter.add_text("Gefäss-Info", position=(37, 248), font_size=8, color="white")

    # --- Anleitungs-Panel ---
    # Overlay, das erklärt, wie man die Simulation bedient.
    # Togglebar per Button (y=272) und Taste H.
    _GUIDE_TEXT = (
        "ANLEITUNG  –  Zerebrale Blutbahn-Simulation\n"
        "============================================\n"
        "\n"
        "THROMBOSE-SZENARIO (Hauptfeature):\n"
        "  1. Häkchen 'Läsion setzen' aktivieren\n"
        "     Eine Verletzungsstelle erscheint an der MCA dx.\n"
        "  2. 'Thromboseneigung'-Slider erhöhen\n"
        "     Weisse Thrombozyten sammeln sich an der Läsion\n"
        "  3. Fortschrittsbalken rechts beobachten\n"
        "  4. Bei 100 %: Gefäß vollständig blockiert\n"
        "     -> R-Taste: Ruptur (Gefäss platzt, Hirnblutung)\n"
        "     -> B-Taste: Rückstau (minimaler Restfluss)\n"
        "  5. Reset: Häkchen 'Läsion setzen' absetzen\n"
        "     Thrombus löst sich, Fluss kehrt zurueck\n"
        "\n"
        "ASPIRIN (Prophylaxe):\n"
        "  Aspirin-Slider erhöhen während Läsion aktiv\n"
        "  -> hemmt Thrombozytenaggregation\n"
        "  -> erstes Plättchen haftet, weitere fliessen durch\n"
        "  -> Thrombus wächst kaum bis gar nicht\n"
        "\n"
        "SZENARIEN (links):\n"
        "  Gesund / Rauchen / Diabetes / + Aspirin\n"
        "  -> setzt Slider automatisch auf typische Patientenwerte\n"
        "\n"
        "FENTANYL:\n"
        "  Slider erhöhen -> Arterien weiten sich, Herzschlag\n"
        "  wird langsamer, gold-Partikel zeigen Wirkstoff der\n"
        "  durch die Blut-Hirn-Schranke ins Gehirn gelangt\n"
        "\n"
        "BLUTDRUCK (Windkessel-Modell):\n"
        "  Oben links: arterieller Druck in mmHg\n"
        "  Berechnet per C*dP/dt = Q(t) - P/R\n"
        "  (R = Gefaesswiderstand, C = Compliance)\n"
        "  Okklusion -> R steigt -> Blutdruck ~130 mmHg\n"
        "  Fentanyl  -> Vasodilatation -> ~65 mmHg\n"
        "  Hinweis: Druck passt sich in ~3-5 Sek. an\n"
        "\n"
    )
    guide_panel_actor = plotter.add_text(
        _GUIDE_TEXT, position=(0, 0), font_size=9, color="white"
    )
    guide_panel_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    guide_panel_actor.GetPositionCoordinate().SetValue(0.99, 0.99)
    guide_panel_actor.GetTextProperty().SetJustificationToRight()
    guide_panel_actor.GetTextProperty().SetVerticalJustificationToTop()
    guide_panel_actor.GetTextProperty().SetBackgroundColor(0.03, 0.03, 0.10)
    guide_panel_actor.GetTextProperty().SetBackgroundOpacity(0.93)
    guide_panel_actor.GetTextProperty().SetFrame(True)
    guide_panel_actor.GetTextProperty().SetFrameColor(0.50, 0.50, 0.90)
    guide_panel_actor.GetTextProperty().SetFrameWidth(2)
    guide_panel_actor.VisibilityOff()

    def _toggle_guide(state=None):
        # Funktioniert als Checkbox-Callback (state=bool) und als Key-Event (state=None)
        if state is None:
            # Key-Event: togglen
            if guide_panel_actor.GetVisibility():
                guide_panel_actor.VisibilityOff()
            else:
                guide_panel_actor.VisibilityOn()
        elif state:
            guide_panel_actor.VisibilityOn()
        else:
            guide_panel_actor.VisibilityOff()

    plotter.add_checkbox_button_widget(
        _toggle_guide, value=False, position=(10, 272), size=22,
        border_size=1, color_on="cyan", color_off="grey", background_color="white"
    )
    plotter.add_text("Anleitung (H)", position=(37, 280), font_size=8, color="white")
    plotter.add_key_event('h', lambda: _toggle_guide(None))

    # Legende: 2-Spalten-Layout (KAMERA|TASTEN, FARBEN|ZELLEN) – unten, linksbündig
    # Courier-Monospace: linke Spalte = 32 Zeichen breit, rechte Spalte ab Position 33.
    legend_actor = plotter.add_text(
        "KAMERA                          TASTEN                   \n"
        "  1  Gesamtansicht                L  Labels umschalten   \n"
        "  2  Kapillaren (oben)            H  Anleitung           \n"
        "  3  Laesion MCA dx.                                     \n"
        "                                                         \n"
        "FARBEN                          ZELLEN                   \n"
        "  rot        = Arterie            weiss      = Thrombozyt\n"
        "  gruen Rohr = Kapillare          gelb       = klebend   \n"
        "  gruen Kug. = Kapillarbett       gold       = Fentanyl  \n"
        "  blau       = Vene / Sinus       rot>blau   = O2-Abgabe ",
        position=(0, 0), font_size=11, color="white"
    )
    legend_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    legend_actor.GetPositionCoordinate().SetValue(0.99, 0.01)
    legend_actor.GetTextProperty().SetFontFamilyToCourier()
    legend_actor.GetTextProperty().SetJustificationToRight()
    legend_actor.GetTextProperty().SetVerticalJustificationToBottom()
    legend_actor.GetTextProperty().SetBackgroundColor(0.06, 0.06, 0.16)
    legend_actor.GetTextProperty().SetBackgroundOpacity(0.82)
    legend_actor.GetTextProperty().SetFrame(True)
    legend_actor.GetTextProperty().SetFrameColor(0.55, 0.55, 0.80)
    legend_actor.GetTextProperty().SetFrameWidth(2)

    # "H  Anleitung" farblich hervorheben: zweiter Aktor (cyan) überlagert exakt dieselbe Zeile.
    # Selbe Schriftart/Größe/Anker → Zeile 3 von oben (= Zeile 8 von unten bei 10 Zeilen gesamt).
    _h_line = "                                  H  Anleitung           "  # 57 Zeichen, H an Pos 34
    _h_overlay = plotter.add_text(
        "\n\n" + _h_line + "\n\n\n\n\n\n\n",
        position=(0, 0), font_size=11, color="cyan"
    )
    _h_overlay.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    _h_overlay.GetPositionCoordinate().SetValue(0.99, 0.01)
    _h_overlay.GetTextProperty().SetFontFamilyToCourier()
    _h_overlay.GetTextProperty().SetJustificationToRight()
    _h_overlay.GetTextProperty().SetVerticalJustificationToBottom()

    # --- Kamera-Presets (Tasten 1–3) ---
    # Jeder Preset setzt Position UND Fokuspunkt → Scroll-Zoom zeigt in die richtige Richtung.
    # _cam_full: Blickrichtung zuerst auf "von vorne" setzen, dann reset_camera() für optimalen Zoom.
    # Das entspricht exakt dem manuellen Ablauf "Taste 2 → Taste 1" (= Gesamtansicht_optimal).
    def _cam_full():
        plotter.camera_position = [(0, 4, 10), (0, 6.5, -0.3), (0, 1, 0)]
        plotter.reset_camera()

    def _cam_cap():
        # Sinus-Eingang / Kapillarmündung bei [0, 7.5, -0.3]; leicht von vorne-oben
        plotter.camera_position = [(0, 4, 10), (0, 6.5, -0.3), (0, 1, 0)]

    def _cam_lesion():
        p = LESION["pos_3d"]
        if p is not None:
            plotter.camera_position = [(p[0]+1, p[1]-1, p[2]+6), tuple(p), (0, 1, 0)]

    plotter.add_key_event('1', _cam_full)
    plotter.add_key_event('2', _cam_cap)
    plotter.add_key_event('3', _cam_lesion)

    # Aspirin-Overlay: zweite Tube-Schicht über allen Haupt-Gefäßen (cyan, leicht größer).
    # Zeigt visuell, dass der Wirkstoff im Blut zirkuliert – erscheint/verschwindet per show/hide.
    _ASPIRIN_OVERLAY_TYPES = {"artery", "brain", "vein", "sinus"}
    aspirin_overlay_actors = []
    for v in vessels:
        if v["type"] not in _ASPIRIN_OVERLAY_TYPES:
            continue
        _spline = pv.Spline(v["points"], SPLINE_PTS)
        _tube   = _spline.tube(radius=v["radius"] * 1.08, n_sides=TUBE_SIDES)
        _actor  = plotter.add_mesh(_tube, color="deepskyblue", opacity=0.18, smooth_shading=True)
        _actor.VisibilityOff()
        aspirin_overlay_actors.append(_actor)

    # Aspirin-Info-Text (rechts, unterhalb Mitte) – erscheint wenn aspirin_level > 0
    aspirin_info_actor = plotter.add_text(
        "", position=(0, 0), font_size=10, color="deepskyblue"
    )
    aspirin_info_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    aspirin_info_actor.GetPositionCoordinate().SetValue(0.99, 0.75)
    aspirin_info_actor.GetTextProperty().SetJustificationToRight()
    aspirin_info_actor.GetTextProperty().SetBackgroundColor(0.02, 0.06, 0.18)
    aspirin_info_actor.GetTextProperty().SetBackgroundOpacity(0.82)
    aspirin_info_actor.GetTextProperty().SetFrame(True)
    aspirin_info_actor.GetTextProperty().SetFrameColor(0.0, 0.55, 0.90)
    aspirin_info_actor.GetTextProperty().SetFrameWidth(2)
    aspirin_info_actor.VisibilityOff()

    # --- Blut-Hirn-Schranke ---
    # Ellipsoid-Drahtrahmen um den Gehirnbereich:
    #   weiß/opak = intakte Schranke; goldfarben/heller = durch Fentanyl durchlässig.
    bhs_sphere = pv.Sphere(radius=1.0, theta_resolution=14, phi_resolution=14)
    bhs_sphere.points = bhs_sphere.points * np.array([5.2, 4.5, 4.0])
    bhs_sphere.points += np.array([0.0, 4.5, -0.2])
    bhs_actor = plotter.add_mesh(bhs_sphere, color="white", opacity=0.06,
                                 style="wireframe", line_width=1)

    # Fentanyl-Overlay: goldgelbe Tube-Schicht über arteriellen Gefäßen
    _FENTANYL_OVERLAY_TYPES = {"artery", "brain"}
    fentanyl_overlay_actors = []
    for v in vessels:
        if v["type"] not in _FENTANYL_OVERLAY_TYPES:
            continue
        _spline = pv.Spline(v["points"], SPLINE_PTS)
        _tube   = _spline.tube(radius=v["radius"] * 1.10, n_sides=TUBE_SIDES)
        _actor  = plotter.add_mesh(_tube, color="gold", opacity=0.20, smooth_shading=True)
        _actor.VisibilityOff()
        fentanyl_overlay_actors.append(_actor)

    # Fentanyl-Partikel-Pool (werden in run_simulation aktualisiert)
    fentanyl_particles = []
    for _i in range(MAX_FENTANYL_PARTICLES):
        _eid = ARTERIAL_ENTRIES[_i % len(ARTERIAL_ENTRIES)]
        fentanyl_particles.append(FentanylParticle(vid_map, plotter, _eid))

    # Fentanyl-Info-Text (rechts, unterhalb Aspirin-Info)
    fentanyl_info_actor = plotter.add_text(
        "", position=(0, 0), font_size=10, color="gold"
    )
    fentanyl_info_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    fentanyl_info_actor.GetPositionCoordinate().SetValue(0.99, 0.60)
    fentanyl_info_actor.GetTextProperty().SetJustificationToRight()
    fentanyl_info_actor.GetTextProperty().SetBackgroundColor(0.08, 0.06, 0.0)
    fentanyl_info_actor.GetTextProperty().SetBackgroundOpacity(0.82)
    fentanyl_info_actor.GetTextProperty().SetFrame(True)
    fentanyl_info_actor.GetTextProperty().SetFrameColor(0.80, 0.65, 0.0)
    fentanyl_info_actor.GetTextProperty().SetFrameWidth(2)
    fentanyl_info_actor.VisibilityOff()

    # --- Gefäß-Elastizität: Ruptur + Rückstau ---
    # Ruptur-Kugel: roter Drahtrahmen an der Läsion, pulsiert wenn aktiv
    ruptur_sphere_mesh = pv.Sphere(radius=0.30, center=LESION["pos_3d"],
                                   theta_resolution=16, phi_resolution=16)
    ruptur_vis_actor = plotter.add_mesh(
        ruptur_sphere_mesh, color="red", style="wireframe", line_width=3)
    ruptur_vis_actor.VisibilityOff()

    # Hinweis nach Okklusion: Nutzer wählt Ausgang
    outcome_hint_actor = plotter.add_text(
        "Druck steigt!   R = Ruptur   |   B = Rueckstau",
        position=(0, 0), font_size=11, color="orange"
    )
    outcome_hint_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    outcome_hint_actor.GetPositionCoordinate().SetValue(0.50, 0.07)
    outcome_hint_actor.GetTextProperty().SetJustificationToCentered()
    outcome_hint_actor.GetTextProperty().SetBackgroundColor(0.10, 0.05, 0.0)
    outcome_hint_actor.GetTextProperty().SetBackgroundOpacity(0.82)
    outcome_hint_actor.GetTextProperty().SetFrame(True)
    outcome_hint_actor.GetTextProperty().SetFrameColor(0.80, 0.40, 0.0)
    outcome_hint_actor.GetTextProperty().SetFrameWidth(2)
    outcome_hint_actor.VisibilityOff()

    # Ruptur-Text – Mitte bei y=0.14 (über Outcome-Hint bei 0.07, unter freier Zone)
    ruptur_text_actor = plotter.add_text(
        "RUPTUR: Hirnblutung!", position=(0, 0), font_size=13, color="red"
    )
    ruptur_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    ruptur_text_actor.GetPositionCoordinate().SetValue(0.50, 0.85)
    ruptur_text_actor.GetTextProperty().SetJustificationToCentered()
    ruptur_text_actor.GetTextProperty().SetBackgroundColor(0.15, 0.0, 0.0)
    ruptur_text_actor.GetTextProperty().SetBackgroundOpacity(0.85)
    ruptur_text_actor.GetTextProperty().SetFrame(True)
    ruptur_text_actor.GetTextProperty().SetFrameColor(0.80, 0.0, 0.0)
    ruptur_text_actor.GetTextProperty().SetFrameWidth(2)
    ruptur_text_actor.VisibilityOff()

    # Rückstau-Text – ebenfalls Mitte y=0.14 (mutually exclusive mit Ruptur)
    rueckstau_text_actor = plotter.add_text(
        "RUECKSTAU: minimaler Restfluss", position=(0, 0), font_size=11, color="orange"
    )
    rueckstau_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    rueckstau_text_actor.GetPositionCoordinate().SetValue(0.50, 0.85)
    rueckstau_text_actor.GetTextProperty().SetJustificationToCentered()
    rueckstau_text_actor.GetTextProperty().SetBackgroundColor(0.08, 0.04, 0.0)
    rueckstau_text_actor.GetTextProperty().SetBackgroundOpacity(0.85)
    rueckstau_text_actor.GetTextProperty().SetFrame(True)
    rueckstau_text_actor.GetTextProperty().SetFrameColor(0.60, 0.30, 0.0)
    rueckstau_text_actor.GetTextProperty().SetFrameWidth(2)
    rueckstau_text_actor.VisibilityOff()

    # Key-Events: R = Ruptur, B = Rückstau (nur nach Okklusion aktiv)
    def _trigger_ruptur():
        if not SIM_STATE["mca_occluded"] or SIM_STATE["ruptur_active"] or SIM_STATE["rueckstau_active"]:
            return
        SIM_STATE["ruptur_active"] = True
        ruptur_vis_actor.VisibilityOn()
        ruptur_text_actor.VisibilityOn()
        if mca_actor:
            mca_actor.prop.color   = "#FF2222"
            mca_actor.prop.opacity = 0.90

    def _trigger_rueckstau():
        if not SIM_STATE["mca_occluded"] or SIM_STATE["ruptur_active"] or SIM_STATE["rueckstau_active"]:
            return
        SIM_STATE["rueckstau_active"] = True
        rueckstau_text_actor.VisibilityOn()
        if mca_actor:
            mca_actor.prop.color   = "#999999"
            mca_actor.prop.opacity = 0.55

    plotter.add_key_event('r', _trigger_ruptur)
    plotter.add_key_event('b', _trigger_rueckstau)

    outcome_actors = {
        "hint":          outcome_hint_actor,
        "ruptur_mesh":   ruptur_sphere_mesh,
        "ruptur_vis":    ruptur_vis_actor,
        "ruptur_text":   ruptur_text_actor,
        "rueckstau_text": rueckstau_text_actor,
    }

    # Blutdruck-Anzeige (Windkessel-Modell): links oben, immer sichtbar
    pressure_actor = plotter.add_text(
        "Blutdruck: --- / --- mmHg", position=(0, 0), font_size=9, color="lightcyan"
    )
    pressure_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    pressure_actor.GetPositionCoordinate().SetValue(0.01, 0.68)
    pressure_actor.GetTextProperty().SetJustificationToLeft()
    pressure_actor.GetTextProperty().SetBackgroundColor(0.02, 0.06, 0.12)
    pressure_actor.GetTextProperty().SetBackgroundOpacity(0.75)

    return (plotter, clot_mesh, clot_actor, mca_actor, stroke_actor,
            cap24_actor, pulse_meshes, lesion_marker_actor,
            lesion_highlight_actor, lesion_hint_actor, progress_actor,
            heart_mesh, heart_base_pts, aspirin_info_actor, aspirin_overlay_actors,
            outcome_actors, fentanyl_info_actor, fentanyl_overlay_actors,
            bhs_actor, fentanyl_particles, pressure_actor)


def run_simulation(vessels, plotter, clot_mesh, clot_actor, mca_actor, stroke_actor,
                   cap24_actor, pulse_meshes, lesion_marker_actor,
                   lesion_highlight_actor, lesion_hint_actor, progress_actor,
                   heart_mesh, heart_base_pts, aspirin_info_actor, aspirin_overlay_actors,
                   outcome_actors, fentanyl_info_actor, fentanyl_overlay_actors,
                   bhs_actor, fentanyl_particles, pressure_actor):
    """Animationsloop: spawnt Zellen, bewegt sie, aktualisiert Thrombus-Mesh."""
    vid_map = {v["id"]: v for v in vessels}

    art_weights = np.array([vid_map[eid]["flow"] for eid in ARTERIAL_ENTRIES], dtype=float)
    art_weights /= art_weights.sum()

    vessel_13_radius = vid_map[LESION["vessel_id"]]["radius"]  # Okklusions-Schwelle
    # Anzahl Plättchen für vollständige Okklusion (für Fortschrittsbalken)
    import math  # nur lokal nötig, schadet nicht hier
    n_required = math.ceil((vessel_13_radius - 0.15) / CLOT_GROWTH_PER_PLATELET)
    _CLOT_RES = BloodCell._CLOT_RES
    cells = []
    frame = 0
    p_history = []   # Druckverlauf der letzten ~2 s für Systole/Diastole-Anzeige

    plotter.show(interactive_update=True)
    # Gesamtansicht_optimal: Blickrichtung auf "von vorne" setzen, dann reset_camera().
    # plotter.show() setzt eine skurrile Auto-Perspektive; reset_camera() allein behält
    # diese Blickrichtung bei. Erst nach explizitem Frontvektor ergibt reset_camera()
    # die gewünschte Gesamtansicht (identisch mit manuell: Taste 2 → Taste 1).
    plotter.camera_position = [(0, 4, 10), (0, 6.5, -0.3), (0, 1, 0)]
    plotter.reset_camera()

    while True:
        # --- Spawn ---
        if len(cells) < MAX_CELLS and frame % SPAWN_RATE == 0:
            if np.random.random() < ARTERIAL_FRACTION:
                entry_id    = int(np.random.choice(ARTERIAL_ENTRIES, p=art_weights))
                cell_type   = "arterial"
                is_platelet = (np.random.random() < PLATELET_RATIO)
            else:
                entry_id    = VENOUS_ENTRIES[0]
                cell_type   = "venous"
                is_platelet = False
            cells.append(BloodCell(vid_map, plotter, entry_id, cell_type, is_platelet))

        # --- Lokale Thrombozyten-Rekrutierung ---
        # Biologie: Gefäßverletzung setzt Kollagen + vWF frei → Margination von
        # Thrombozyten aus dem vorbeiströmenden Blut (ADP-Rekrutierung, ~2/s).
        # Technisch: Thrombozyt direkt in vessel 13 gespawnt; durchläuft normal MCA → Kapillar → Vene.
        if SIM_STATE["lesion_active"] and not SIM_STATE["mca_occluded"]:
            now = time.time()
            if now - SIM_STATE["last_platelet_inject"] >= 2.0:
                SIM_STATE["last_platelet_inject"] = now
                cells.append(BloodCell(vid_map, plotter, LESION["vessel_id"], "arterial", is_platelet=True))

        # --- Update ---
        for cell in cells:
            cell.update(TIME_STEP, cells)

        # --- Aspirin-Overlay + Info-Text: ein-/ausblenden ---
        if SIM_STATE["aspirin_level"] > 0:
            for _a in aspirin_overlay_actors:
                _a.VisibilityOn()
            aspirin_info_actor.SetInput(
                "Aspirin eingenommen\n"
                "--------------------\n"
                "Sichtbar: Gefäße schimmern\n"
                "hellblau (Wirkstoff im Blut)\n"
                "\n"
                "Ohne Aspirin:\n"
                "  Plättchen haften aneinander\n"
                "  -> Pfropf wächst an Läsion\n"
                "  -> Blockade\n"
                "\n"
                "Mit Aspirin:\n"
                "  Erstes Plättchen haftet\n"
                "  Weitere fliessen vorbei\n"
                "  -> Kein Pfropf"
                "\n"
                "'Thrombus lösen' klicken, um Thrombose zu lösen"
            )
            aspirin_info_actor.VisibilityOn()
        else:
            for _a in aspirin_overlay_actors:
                _a.VisibilityOff()
            aspirin_info_actor.VisibilityOff()

        # --- Fentanyl: Partikel + Overlay + BHS + Info ---
        fent = SIM_STATE["fentanyl_level"]
        for fp in fentanyl_particles:
            fp.update(TIME_STEP)

        if fent > 0:
            for _a in fentanyl_overlay_actors:
                _a.VisibilityOn()
                _a.prop.opacity = 0.10 + 0.12 * fent
            # BHS: weißlich → goldgelb, opacity steigt (Schranke sichtbar durchlässig)
            bhs_actor.prop.color   = (1.0, max(0.25, 0.90 - 0.65 * fent), 0.0)
            bhs_actor.prop.opacity = 0.06 + 0.20 * fent
            fentanyl_info_actor.SetInput(
                "Fentanyl (Opioid-Schmerzmittel)\n"
                "--------------------------------\n"
                "Die Blut-Hirn-Schranke (BHS)\n"
                "schuetzt das Gehirn: fast kein\n"
                "Stoff dringt von aussen durch.\n"
                "\n"
                "Fentanyl ist fettlöslich und\n"
                "ueberwindet die BHS direkt.\n"
                "-> Gold-Partikel: Wirkstoff im Blut\n"
                "-> Partikel wandern in die gruenen\n"
                "   Kapillarbetten (= Hirngewebe)\n"
                "\n"
                "Weitere Effekte:\n"
                "  Gefässe weiten sich\n"
                "  Herzschlag verlangsamt"
            )
            fentanyl_info_actor.VisibilityOn()
        else:
            for _a in fentanyl_overlay_actors:
                _a.VisibilityOff()
            bhs_actor.prop.color   = "white"
            bhs_actor.prop.opacity = 0.06
            fentanyl_info_actor.VisibilityOff()

        # --- Thrombus-Mesh aktualisieren + Okklusions-Check ---
        stuck = [c for c in cells if c.is_stuck]
        if stuck:
            clot_center = np.mean([c.pos for c in stuck], axis=0)
            clot_radius = 0.15 + CLOT_GROWTH_PER_PLATELET * len(stuck)
            new_sphere  = pv.Sphere(radius=clot_radius, center=clot_center,
                                    theta_resolution=_CLOT_RES, phi_resolution=_CLOT_RES)
            clot_mesh.points        = new_sphere.points
            clot_actor.prop.opacity = 0.85

            # Okklusion erkennen
            if not SIM_STATE["mca_occluded"] and clot_radius >= vessel_13_radius:
                SIM_STATE["mca_occluded"] = True
                if mca_actor:
                    mca_actor.prop.color   = "#555555"
                    mca_actor.prop.opacity = 0.6
                if cap24_actor:
                    cap24_actor.prop.color   = "#555555"
                    cap24_actor.prop.opacity = 0.08
                stroke_actor.VisibilityOn()
        else:
            clot_actor.prop.opacity = 0.0

        # --- Fortschrittsanzeige aktualisieren (nur wenn Läsion aktiv) ---
        if SIM_STATE["lesion_active"]:
            n_s = len(stuck)
            filled = min(20, int(n_s / n_required * 20))
            bar = "#" * filled + "-" * (20 - filled)
            pct = min(100, int(n_s / n_required * 100))
            progress_actor.SetInput(
                f"Thrombose: [{bar}] {pct}%  ({n_s}/{n_required} Plättchen)"
            )

        # --- Clear-Signal: Thrombus lösen + Okklusion + Outcome zurücksetzen ---
        if SIM_STATE["clear_clot_signal"]:
            SIM_STATE["clear_clot_signal"] = False
            SIM_STATE["ruptur_active"]     = False
            SIM_STATE["rueckstau_active"]  = False
            outcome_actors["ruptur_vis"].VisibilityOff()
            outcome_actors["ruptur_text"].VisibilityOff()
            outcome_actors["rueckstau_text"].VisibilityOff()
            if SIM_STATE["mca_occluded"]:
                SIM_STATE["mca_occluded"] = False
                if mca_actor:
                    mca_actor.prop.color   = COLOR["brain"]
                    mca_actor.prop.opacity = 0.35
                if cap24_actor:
                    cap24_actor.prop.color   = "mediumseagreen"
                    cap24_actor.prop.opacity = 0.15
                stroke_actor.VisibilityOff()

        # --- Outcome-Hint: nach Okklusion einblenden bis Ausgang gewählt ---
        if (SIM_STATE["mca_occluded"]
                and not SIM_STATE["ruptur_active"]
                and not SIM_STATE["rueckstau_active"]):
            outcome_actors["hint"].VisibilityOn()
        else:
            outcome_actors["hint"].VisibilityOff()

        # --- Puls: große Arterien sinusförmig weiten und verengen ---
        t_sim    = frame * TIME_STEP
        # Bradykardie bei Fentanyl: Herzfrequenz sinkt bis auf 70% des Normalwerts
        eff_freq  = PULSE_FREQ * (1.0 - 0.30 * SIM_STATE["fentanyl_level"])
        pulse_sin = np.sin(2 * np.pi * eff_freq * t_sim)
        for vid, mesh in pulse_meshes.items():
            v = vid_map[vid]
            # Vasodilatation bei Fentanyl: Basisradius weitet sich um bis zu 20%
            r_base = v["radius"] * (1.0 + 0.20 * SIM_STATE["fentanyl_level"])
            if vid == LESION["vessel_id"]:
                # Elastizität: Radius wächst mit Druckstau (auf Basis des dilatierten Radius)
                if SIM_STATE["lesion_active"]:
                    n_s = len(stuck)
                    clot_frac = min(1.0, n_s / max(1, n_required))
                    if SIM_STATE["ruptur_active"]:
                        r_new = r_base * 1.35          # max. Dehnung, statisch
                    elif SIM_STATE["rueckstau_active"]:
                        # Noch leicht elastisch, gedämpfter Puls
                        r_new = r_base * (1.18 + 0.04 * np.sin(
                            2 * np.pi * eff_freq * 0.5 * t_sim))
                    elif SIM_STATE["mca_occluded"]:
                        r_new = r_base * 1.25          # vollständig gestaut, statisch
                    else:
                        extra_amp = clot_frac * 0.15
                        r_new = r_base * (
                            1.0 + clot_frac * 0.08
                            + (PULSE_AMPLITUDE + extra_amp) * pulse_sin)
                else:
                    r_new = r_base * (1 + PULSE_AMPLITUDE * pulse_sin)
            else:
                # Puls-Amplitude skaliert mit dem aktuellen Druck:
                # höherer Blutdruck → sichtbar stärkere Gefäßdehnung
                p_norm = max(0.5, SIM_STATE["windkessel_p"] / 100.0)
                r_new = r_base * (1 + PULSE_AMPLITUDE * p_norm * pulse_sin)
            new_tube    = pv.Spline(v["points"], SPLINE_PTS).tube(radius=r_new, n_sides=TUBE_SIDES)
            mesh.points = new_tube.points

        # Ruptur-Kugel pulsiert (Throb-Effekt)
        if SIM_STATE["ruptur_active"]:
            _r_burst = 0.28 + 0.12 * abs(np.sin(3 * np.pi * PULSE_FREQ * t_sim))
            _burst   = pv.Sphere(radius=_r_burst, center=LESION["pos_3d"],
                                 theta_resolution=16, phi_resolution=16)
            outcome_actors["ruptur_mesh"].points = _burst.points

        # --- Herzpuls: synchron mit Arterien-Puls (gleiche Frequenz + Amplitude) ---
        heart_mesh.points = HEART_CENTER + (heart_base_pts - HEART_CENTER) * (1.0 + PULSE_AMPLITUDE * pulse_sin)

        # --- Windkessel-Modell: C·dP/dt = Q(t) - P/R ---
        # Q(t): positive Halbwelle des Herzsinussignals (Systole = Auswurfphase)
        Q_t   = max(0.0, pulse_sin) * WK_Q0
        # Peripherer Widerstand: steigt bei Okklusion, sinkt bei Vasodilatation
        R_eff = WK_R * (1.0 + 0.50 * float(SIM_STATE["mca_occluded"]))
        R_eff *= (1.0 - 0.40 * SIM_STATE["fentanyl_level"])
        R_eff  = max(R_eff, 0.05)
        # Euler-Schritt
        P  = SIM_STATE["windkessel_p"]
        P += TIME_STEP * (Q_t - P / R_eff) / WK_C
        SIM_STATE["windkessel_p"] = P
        # Systole/Diastole aus rollendem Fenster (~2 s)
        p_history.append(P)
        if len(p_history) > 20:
            p_history.pop(0)
        p_sys = int(round(max(p_history)))
        p_dia = int(round(min(p_history)))
        pressure_actor.SetInput(f"Blutdruck: {p_sys} / {p_dia} mmHg")

        # --- 3s Läsions-Highlight ausblenden ---
        t0 = SIM_STATE["lesion_activate_time"]
        if t0 is not None and (time.time() - t0) >= 1.0:
            lesion_highlight_actor.VisibilityOff()
            lesion_hint_actor.VisibilityOff()
            SIM_STATE["lesion_activate_time"] = None  # nicht wiederholt prüfen

        plotter.update()
        frame += 1
        time.sleep(0.005)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main(render_3d=True):
    vessels = build_vessel_network()
    if render_3d:
        (plotter, clot_mesh, clot_actor, mca_actor, stroke_actor,
         cap24_actor, pulse_meshes, lesion_marker_actor,
         lesion_highlight_actor, lesion_hint_actor, progress_actor,
         heart_mesh, heart_base_pts, aspirin_info_actor,
         aspirin_overlay_actors, outcome_actors,
         fentanyl_info_actor, fentanyl_overlay_actors,
         bhs_actor, fentanyl_particles, pressure_actor) = setup_plotter(vessels)
        run_simulation(vessels, plotter, clot_mesh, clot_actor, mca_actor, stroke_actor,
                       cap24_actor, pulse_meshes, lesion_marker_actor,
                       lesion_highlight_actor, lesion_hint_actor, progress_actor,
                       heart_mesh, heart_base_pts, aspirin_info_actor, aspirin_overlay_actors,
                       outcome_actors, fentanyl_info_actor, fentanyl_overlay_actors,
                       bhs_actor, fentanyl_particles, pressure_actor)


if __name__ == "__main__":
    check_only = "--check-only" in sys.argv
    main(render_3d=not check_only)
