"""
Auswertung der IMU-Daten aus den Sturzversuchen im Vorstieg.

Versuchsaufbau
--------------
* C1 = sichernde Person, 70 kg, IMU am Bauch, etwa im Körperschwerpunkt
* C2 = kletternde Person, 78 kg, IMU am Bauch, etwa im Körperschwerpunkt
* Hüfte beim Sturz auf 9,80 m, letzter Karabiner auf 8,65 m
  -> geometrisch 2 * 1,15 m + 0,20 m Schlappseil = 2,5 m freier Fall
* Dynamisch gesichert, sichernde Person 1–2 m von der Wand entfernt
* Messungen in zeitlicher Reihenfolge: 1–5 Petzl GriGri, 6–10 Wild Country Revo,
  11–15 Petzl Reverso (Tube)

Physik hinter den Kennwerten
----------------------------
Ein Beschleunigungssensor misst die spezifische Kraft (Beschleunigung ohne den
Anteil der Schwerkraft): 1 g in Ruhe, 0 g im freien Fall. Betrag mal Körpermasse
ist die gesamte äußere Kraft auf den Körper ohne Schwerkraft, also Seil plus
Wand- bzw. Bodenkontakt:  F ≈ m * |a_gemessen|

Was belastbar ist und was nicht
-------------------------------
* Kraftspitze der Kletternden: unabhängig vom Filter (2,3–2,6 g von 6 Hz bis
  ungefiltert), also belastbar.
* Beschleunigungsspitze der Sichernden: hängt stark vom Filter ab (2,3 g bei
  6 Hz, 6 g ungefiltert) und enthält Bodenabdruck und Wandkontakt. Sie wird nur
  in der CSV und in 05_filtereinfluss.png gezeigt. Als Kraft der Sichernden
  dient stattdessen der Mittelwert in der Flugphase (FLUGFENSTER nach
  "Seil greift"), in der sie nur am Seil hängt.
* Wege und Geschwindigkeiten: aus der vertikalen Beschleunigung integriert
  (Euler-Winkel ZYX ins Weltsystem, g abgezogen), Drift über Ruhephasen vor und
  nach dem Sturz korrigiert. Die Korrektur beträgt bis zu 0,7 m/s, die Werte
  sind also Schätzungen (± 15 %). Die Drift je Messung steht in der CSV.
* Fallhöhe: Die Zeit vom Loslassen bis zum ersten Seilzug liefert eine harte
  Obergrenze h <= g/2 * T². Sie liegt bei 1,1–2,0 m und damit unter den 2,5 m
  aus der Geometrie (siehe 04_fallhoehe.png).
* Gerätevergleich: n = 5 je Gerät, Geräte in festen Blöcken nacheinander
  getestet. Gerät und Reihenfolge lassen sich nicht trennen
  (siehe 03_reihenfolge.png).

Aufruf:  python auswertung.py   (Ergebnisse landen in ./results)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------
BASIS = Path(__file__).resolve().parent
DATEN = BASIS / "Data"
ERGEBNISSE = BASIS / "results"

G = 9.81  # m/s²
FS = 120.0  # Hz (SampleTimeFine-Abstand 8333 µs)
GRENZFREQUENZ = 15.0  # Hz, Tiefpass für "Körper"-Beschleunigung

PERSONEN = {
    "C1": {"name": "Sichernde Person (C1)", "masse": 70.0},
    "C2": {"name": "Kletternde Person (C2)", "masse": 78.0},
}

GERAETE = ["GriGri", "Revo", "Reverso"]
GERAETE_LANG = {
    "GriGri": "Petzl GriGri",
    "Revo": "Wild Country Revo",
    "Reverso": "Petzl Reverso (Tube)",
}
# Kategoriale Farben (validierte Standardpalette, Slots 1–3)
FARBEN = {"GriGri": "#2a78d6", "Revo": "#eb6834", "Reverso": "#1baf7a"}

STURZHOEHE = 9.80  # m
SICHERUNGSHOEHE = 8.65  # m
SCHLAPPSEIL = 0.20  # m
STURZWEG_THEORIE = 2 * (STURZHOEHE - SICHERUNGSHOEHE) + SCHLAPPSEIL  # 2,5 m

# Flugphase der Sichernden: Zeitfenster nach "Seil greift", in dem sie bereits
# > 0,1 m angehoben ist und nur am Seil hängt
FLUGFENSTER = (0.15, 0.45)  # s


# --------------------------------------------------------------------------
# Einlesen
# --------------------------------------------------------------------------
@dataclass
class Signal:
    t: np.ndarray  # s, gemeinsame Zeitbasis (SampleTimeFine)
    a_betrag_roh: np.ndarray  # g
    a_betrag: np.ndarray  # g, tiefpassgefiltert
    a_vert: np.ndarray  # m/s², vertikal im Weltsystem, ohne g, gefiltert


def tiefpass(x: np.ndarray, fc: float = GRENZFREQUENZ, ordnung: int = 4) -> np.ndarray:
    b, a = butter(ordnung, fc / (FS / 2), btype="low")
    return filtfilt(b, a, x, axis=0)


def lade_signal(pfad: Path) -> Signal:
    df = pd.read_csv(pfad, skipinitialspace=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.iloc[1:].reset_index(drop=True)  # erste Zeile: Acc/Gyr = 0 (Sensorstart)

    zeit_us = df["SampleTimeFine"].to_numpy(dtype=np.int64)
    zeit_us = np.unwrap(zeit_us, period=2**32)  # 32-bit-Überlauf abfangen
    t = zeit_us / 1e6

    acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy()
    euler = df[["Euler_X", "Euler_Y", "Euler_Z"]].to_numpy()  # Roll, Pitch, Yaw
    rot = Rotation.from_euler("ZYX", euler[:, ::-1], degrees=True)
    acc_welt = rot.apply(acc)

    betrag_roh = np.linalg.norm(acc, axis=1) / G
    return Signal(
        t=t,
        a_betrag_roh=betrag_roh,
        a_betrag=tiefpass(betrag_roh),
        a_vert=tiefpass(acc_welt[:, 2] - G),
    )


def finde_messungen() -> pd.DataFrame:
    muster = re.compile(r"(C[12])-\w+?_[0-9A-F]+_(\d{8})_(\d{6})\.csv")
    zeilen = []
    for pfad in sorted(DATEN.glob("*.csv")):
        m = muster.match(pfad.name)
        if m:
            zeilen.append({"person": m.group(1), "zeitstempel": m.group(3), "pfad": pfad})
    df = pd.DataFrame(zeilen).pivot(index="zeitstempel", columns="person", values="pfad")
    df = df.sort_index().reset_index()
    if len(df) != 15:
        raise ValueError(f"Erwartet 15 Messungen, gefunden {len(df)}")
    df["nr"] = np.arange(1, 16)
    df["geraet"] = [GERAETE[i // 5] for i in range(15)]
    df["versuch"] = [i % 5 + 1 for i in range(15)]
    return df


# --------------------------------------------------------------------------
# Ereigniserkennung & Integration
# --------------------------------------------------------------------------
def zusammenhaengende_bereiche(maske: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, maske.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def finde_ruhe_nach(a: np.ndarray, start: int, toleranz=0.12, dauer_s=0.4) -> int:
    """Erster Index nach `start`, ab dem |a| für `dauer_s` innerhalb 1 g ± toleranz bleibt."""
    n = int(dauer_s * FS)
    ruhig = np.abs(a - 1.0) < toleranz
    for i in range(start, len(a) - n):
        if ruhig[i : i + n].all():
            return i
    return len(a) - 1


def integriere(sig: Signal, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Vertikale Geschwindigkeit und Weg (Ruhe bei i0 und i1, Drift linear korrigiert).

    Dritter Rückgabewert: korrigierte Drift in m/s (Restgeschwindigkeit am Ruhepunkt).
    """
    a = sig.a_vert.copy()
    a -= np.median(a[max(0, i0 - int(0.4 * FS)) : i0 + 1])  # Offset aus der Ruhephase
    v = np.zeros_like(a)
    v[i0 + 1 :] = np.cumsum((a[i0:-1] + a[i0 + 1 :]) / 2) / FS
    # Drift: v(i1) muss 0 sein (Ruhe), linear zwischen i0 und i1 verteilt
    drift = np.zeros_like(v)
    drift[i0 : i1 + 1] = np.linspace(0, v[i1], i1 - i0 + 1)
    drift[i1 + 1 :] = v[i1]
    drift_ms = float(v[i1])
    v -= drift
    v[:i0] = 0
    s = np.zeros_like(v)
    s[i0 + 1 :] = np.cumsum((v[i0:-1] + v[i0 + 1 :]) / 2) / FS
    return v, s, drift_ms


def auswerten(c1: Signal, c2: Signal) -> tuple[dict, dict]:
    t, tc = c2.t, c1.t  # gemeinsame Uhr (SampleTimeFine)

    # ---- Kletternde Person: freier Fall = längster Bereich mit |a| < 0,5 g
    verbunden: list[list[int]] = []
    for s, e in zusammenhaengende_bereiche(c2.a_betrag < 0.5):
        if verbunden and (s - verbunden[-1][1]) < 0.15 * FS:  # kurze Lücken überbrücken
            verbunden[-1][1] = e
        else:
            verbunden.append([s, e])
    ff_s, ff_e = max(verbunden, key=lambda b: b[1] - b[0])

    # Loslassen: letzter Zeitpunkt vor dem freien Fall mit |a| >= 0,9 g
    i_los = ff_s
    while i_los > 0 and c2.a_betrag[i_los] < 0.9:
        i_los -= 1
    # "Seil greift": |a| steigt wieder über 1 g -> maximale Fallgeschwindigkeit
    i_vmax = ff_e
    while i_vmax < len(t) - 1 and c2.a_betrag[i_vmax] < 1.0:
        i_vmax += 1

    # ---- Seilzug beginnt: erste Abweichung > 0,25 g bei der Sichernden.
    # Falls sie erst nach "Seil greift" reagiert: letztes Minimum bei der Kletternden.
    j_los = int(np.searchsorted(tc, t[i_los]))
    j_seil = int(np.searchsorted(tc, t[i_vmax]))
    j_react = j_los + int(np.argmax(np.abs(c1.a_betrag[j_los:] - 1.0) > 0.25))
    i_min = i_vmax
    while i_min > i_los and c2.a_betrag[i_min - 1] < c2.a_betrag[i_min]:
        i_min -= 1
    t_zug = tc[j_react] if tc[j_react] < t[i_vmax] else t[i_min]
    i_zug = int(np.searchsorted(t, t_zug))

    # ---- Integration Kletternde
    i_peak2 = i_vmax + int(np.argmax(c2.a_betrag[i_vmax : i_vmax + int(1.0 * FS)]))
    i_ruhe2 = finde_ruhe_nach(c2.a_betrag, i_peak2 + int(0.3 * FS))
    i_start = max(0, i_los - int(0.1 * FS))
    v2, s2, drift2 = integriere(c2, i_start, i_ruhe2)
    i_vmax_int = i_los + int(np.argmin(v2[i_los : i_peak2 + 1]))
    i_tief = i_vmax_int + int(np.argmin(s2[i_vmax_int : i_ruhe2 + 1]))  # tiefster Punkt, v = 0

    def fenster_mittel(zeit, a):
        m = (zeit >= t[i_vmax] + FLUGFENSTER[0]) & (zeit <= t[i_vmax] + FLUGFENSTER[1])
        return float(np.mean(a[m]))

    m2 = PERSONEN["C2"]["masse"]
    t_frei = t_zug - t[i_los]
    k2 = {
        "t_loslassen": t[i_los],
        "t_seil": t[i_vmax],
        "F_peak_kN": m2 * G * c2.a_betrag[i_peak2] / 1000,
        "a_peak_g": c2.a_betrag[i_peak2],
        "a_peak_roh_g": c2.a_betrag_roh[i_vmax : i_vmax + int(1.0 * FS)].max(),
        "F_flugfenster_kN": m2 * G * fenster_mittel(t, c2.a_betrag) / 1000,
        "anstiegszeit_s": t[i_peak2] - t_zug,
        "freifallzeit_s": t_frei,
        "fallhoehe_obergrenze_m": 0.5 * G * t_frei**2,
        "fallhoehe_integriert_m": -s2[i_zug],
        "v_obergrenze_ms": G * t_frei,
        "v_max_ms": -v2[i_vmax_int],
        "bremszeit_s": t[i_tief] - t[i_vmax_int],
        "bremsweg_m": s2[i_vmax_int] - s2[i_tief],
        "gesamtweg_m": -s2[i_tief],
        "drift_ms": drift2,
        "ruhephase_gefunden": bool(i_ruhe2 < len(t) - 1),
    }

    # ---- Sichernde Person
    fenster = slice(j_los, min(len(tc), j_seil + int(1.2 * FS)))
    j_peak1 = fenster.start + int(np.argmax(c1.a_betrag[fenster]))
    j_ruhe1 = finde_ruhe_nach(c1.a_betrag, j_peak1 + int(0.3 * FS))
    j_start = max(0, j_los - int(0.1 * FS))
    v1, s1, drift1 = integriere(c1, j_start, j_ruhe1)

    m1 = PERSONEN["C1"]["masse"]
    k1 = {
        "F_flugfenster_kN": m1 * G * fenster_mittel(tc, c1.a_betrag) / 1000,
        "anhebung_m": s1[j_start : j_ruhe1 + 1].max(),
        "v_hoch_max_ms": v1[j_start : j_ruhe1 + 1].max(),
        "seilzug_vor_seil_greift_s": t[i_vmax] - t_zug,
        "a_peak_15Hz_g": c1.a_betrag[j_peak1],
        "a_peak_roh_g": c1.a_betrag_roh[fenster].max(),
        "drift_ms": drift1,
        "ruhephase_gefunden": bool(j_ruhe1 < len(tc) - 1),
    }
    # Seilreibung an der Umlenkung: Kraft Sichernde / Kletternde in der Flugphase
    k1["kraftverhaeltnis_flug"] = k1["F_flugfenster_kN"] / k2["F_flugfenster_kN"]

    t0 = t[i_vmax]
    verlauf = {
        "c2": {"t": t - t0, "a": c2.a_betrag, "a_roh": c2.a_betrag_roh, "v": v2, "s": s2},
        "c1": {"t": tc - t0, "a": c1.a_betrag, "a_roh": c1.a_betrag_roh, "v": v1, "s": s1},
        "marken": {
            "Loslassen": t[i_los] - t0,
            "Seilzug beginnt": t_zug - t0,
            "Seil greift (v max)": 0.0,
            "Tiefster Punkt": t[i_tief] - t0,
        },
    }
    return {"C1": k1, "C2": k2}, verlauf


# --------------------------------------------------------------------------
# Statistik
# --------------------------------------------------------------------------
KENNWERTE = {
    # Spalte: (Beschriftung, Einheit)
    "C2_F_peak_kN": ("Max. Kraft Kletternde", "kN"),
    "C1_F_flugfenster_kN": ("Kraft Sichernde in der Flugphase", "kN"),
    "C1_kraftverhaeltnis_flug": ("Kraftverhältnis Sichernde / Kletternde", "–"),
    "C2_anstiegszeit_s": ("Seilzug bis Kraftspitze", "s"),
    "C2_freifallzeit_s": ("Loslassen bis Seilzug", "s"),
    "C2_v_max_ms": ("Max. Fallgeschwindigkeit", "m/s"),
    "C2_bremsweg_m": ("Bremsweg", "m"),
    "C2_gesamtweg_m": ("Gesamter Sturzweg", "m"),
    "C1_anhebung_m": ("Anhebung Sichernde", "m"),
}


def holm(p: np.ndarray) -> np.ndarray:
    reihenfolge = np.argsort(p)
    korrigiert = np.empty(len(p))
    laufend = 0.0
    for rang, idx in enumerate(reihenfolge):
        laufend = max(laufend, min(1.0, p[idx] * (len(p) - rang)))
        korrigiert[idx] = laufend
    return korrigiert


def statistik(df: pd.DataFrame) -> pd.DataFrame:
    """Mittelwert/SD je Gerät, Kruskal-Wallis (Holm über alle Kennwerte), Trend über die Messfolge."""
    zeilen = []
    for col, (label, einheit) in KENNWERTE.items():
        gruppen = [df.loc[df.geraet == g, col].to_numpy() for g in GERAETE]
        zeile = {"Kennwert": label, "Einheit": einheit}
        for g, werte in zip(GERAETE, gruppen):
            zeile[f"{g} MW"] = werte.mean()
            zeile[f"{g} SD"] = werte.std(ddof=1)
        zeile["Kruskal-Wallis p"] = stats.kruskal(*gruppen).pvalue
        trend = stats.spearmanr(df.nr, df[col])
        zeile["Trend Messfolge rho"] = trend.statistic
        zeile["Trend p"] = trend.pvalue
        zeilen.append(zeile)
    st = pd.DataFrame(zeilen)
    st.insert(st.columns.get_loc("Kruskal-Wallis p") + 1, "p Holm-korrigiert",
              holm(st["Kruskal-Wallis p"].to_numpy()))
    return st


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GITTER = "#e4e3df"
FLAECHE = "#fcfcfb"


def stil():
    plt.rcParams.update(
        {
            "figure.facecolor": FLAECHE,
            "axes.facecolor": FLAECHE,
            "axes.edgecolor": GITTER,
            "axes.labelcolor": TEXT2,
            "axes.titlecolor": TEXT,
            "axes.grid": True,
            "grid.color": GITTER,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT2,
            "ytick.color": TEXT2,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def titel(fig, text: str, unterzeile: str = ""):
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, color=TEXT)
    if unterzeile:
        fig.text(0.01, 0.955 if fig.get_figheight() > 8 else 0.92, unterzeile, fontsize=9.5,
                 color=TEXT2, va="top")


def plot_zeitverlaeufe(verlaeufe: list[dict], kennwerte: list[dict], meta: pd.DataFrame, ziel: Path):
    """Einzelstürze je Gerät. Bewusst keine Mittelwertkurve: Sie verschmiert die Kraftspitzen,
    weil diese nicht in jedem Sturz zum selben Zeitpunkt kommen."""
    zeilen = [
        ("c2", "a", "Kletternde (C2)\n|a| [g]"),
        ("c1", "a", "Sichernde (C1)\n|a| [g]"),
        ("c2", "v", "Kletternde (C2)\nVertikalgeschw. [m/s]"),
        ("c1", "s", "Sichernde (C1)\nAnhebung [m]"),
    ]
    fig, axs = plt.subplots(len(zeilen), 3, figsize=(15, 13), sharex=True, sharey="row")
    for spalte, g in enumerate(GERAETE):
        idx = meta.index[meta.geraet == g]
        for zeile, (person, groesse, ylabel) in enumerate(zeilen):
            ax = axs[zeile, spalte]
            for i in idx:
                vl = verlaeufe[i][person]
                m = (vl["t"] >= -1.0) & (vl["t"] <= 1.8)
                ax.plot(vl["t"][m], vl[groesse][m], color=FARBEN[g], linewidth=1.2, alpha=0.85)
            ax.axvline(0, color=TEXT2, linewidth=1, linestyle="--")
            if spalte == 0:
                ax.set_ylabel(ylabel)
            if zeile == 0:
                ax.set_title(GERAETE_LANG[g], loc="left")
                spitze = np.mean([kennwerte[i]["C2"]["a_peak_g"] for i in idx])
                ax.axhline(spitze, color=TEXT2, linewidth=0.8, linestyle=":")
                ax.text(1.8, spitze, f"Ø Spitze {spitze:.2f} g ", ha="right", va="bottom",
                        fontsize=9, color=TEXT2)
            if zeile == len(zeilen) - 1:
                ax.set_xlabel("Zeit rel. zu 'Seil greift' [s]")
    axs[0, 0].set_xlim(-1.0, 1.8)
    titel(fig, "Einzelstürze je Sicherungsgerät (je Linie ein Sturz, Tiefpass 15 Hz)",
          "Gestrichelt: 'Seil greift' = Kletternde erreicht wieder 1 g, also maximale Fallgeschwindigkeit. "
          "Achsen je Zeile gleich skaliert.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ziel, dpi=150)
    plt.close(fig)


def punkte_je_geraet(ax, df: pd.DataFrame, col: str, rng):
    """Einzelwerte je Gerät mit Mittelwertstrich, y-Achse ab 0."""
    oben = 0.0
    for k, g in enumerate(GERAETE):
        w = df.loc[df.geraet == g, col].to_numpy()
        x = k + rng.uniform(-0.13, 0.13, len(w))
        ax.scatter(x, w, s=45, color=FARBEN[g], edgecolor=FLAECHE, linewidth=1.5, zorder=3)
        ax.hlines(w.mean(), k - 0.28, k + 0.28, color=TEXT, linewidth=1.5, zorder=4)
        ax.text(k + 0.32, w.mean(), f"Ø {w.mean():.2f}", va="center", fontsize=8.5, color=TEXT2)
        oben = max(oben, w.max())
    ax.set_xticks(range(3), GERAETE)
    ax.set_xlim(-0.5, 2.85)
    ax.set_ylim(0, oben * 1.15)
    ax.grid(axis="x", visible=False)


def plot_kennwerte(df: pd.DataFrame, st: pd.DataFrame, ziel: Path):
    fig, axs = plt.subplots(3, 3, figsize=(13, 11))
    rng = np.random.default_rng(0)
    for ax, (col, (label, einheit)), p in zip(axs.flat, KENNWERTE.items(), st["Kruskal-Wallis p"]):
        punkte_je_geraet(ax, df, col, rng)
        ax.set_title(f"{label} [{einheit}]", loc="left", fontsize=10)
        ax.text(0.99, 0.03, f"Kruskal-Wallis p = {p:.2f}", transform=ax.transAxes, ha="right",
                fontsize=8.5, color=TEXT2)
    titel(fig, "Kennwerte je Sicherungsgerät",
          "Punkt = ein Sturz, Strich = Mittelwert (n = 5). Achsen beginnen bei 0, "
          "damit kleine Unterschiede nicht größer wirken, als sie sind.")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ziel, dpi=150)
    plt.close(fig)


def geraete_legende(ax, **kw):
    griffe = [plt.Line2D([], [], marker="o", linestyle="", color=FARBEN[g], markersize=7,
                         label=GERAETE_LANG[g]) for g in GERAETE]
    return ax.legend(handles=griffe, **kw)


def plot_reihenfolge(df: pd.DataFrame, st: pd.DataFrame, ziel: Path):
    """Kennwerte über der Messnummer: Gerät und zeitliche Reihenfolge sind nicht trennbar."""
    auswahl = ["C2_F_peak_kN", "C2_bremsweg_m", "C2_gesamtweg_m"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, col in zip(axs, auswahl):
        label, einheit = KENNWERTE[col]
        ax.plot(df.nr, df[col], color=GITTER, linewidth=1.2, zorder=1)
        for g in GERAETE:
            d = df[df.geraet == g]
            ax.scatter(d.nr, d[col], s=50, color=FARBEN[g], edgecolor=FLAECHE, linewidth=1.5, zorder=3)
        zeile = st[st.Kennwert == label].iloc[0]
        ax.set_title(f"{label} [{einheit}]", loc="left", fontsize=10)
        ax.text(0.99, 0.04, f"Trend: Spearman ρ = {zeile['Trend Messfolge rho']:+.2f}, "
                f"p = {zeile['Trend p']:.2f}", transform=ax.transAxes, ha="right", fontsize=8.5, color=TEXT2)
        ax.set_ylim(0, df[col].max() * 1.15)
        ax.set_xticks(range(1, 16))
        ax.set_xlabel("Messnummer (zeitliche Reihenfolge)")
        ax.grid(axis="x", visible=False)
    geraete_legende(axs[0], loc="lower left", fontsize=9)
    titel(fig, "Reihenfolge der Messungen",
          "Die Geräte wurden blockweise nacheinander getestet. Ein Unterschied zwischen den Geräten "
          "wäre von einem Trend über die Zeit (Seil, Ermüdung, Gewöhnung) nicht zu unterscheiden.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(ziel, dpi=150)
    plt.close(fig)


def plot_fallhoehe(df: pd.DataFrame, ziel: Path):
    """Freier Fall bis zum ersten Seilzug: Integration, Obergrenze aus der Fallzeit, Geometrie."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for g in GERAETE:
        d = df[df.geraet == g]
        ax.vlines(d.nr, d.C2_fallhoehe_integriert_m, d.C2_fallhoehe_obergrenze_m, color=FARBEN[g],
                  linewidth=2, alpha=0.45, zorder=2)
        ax.scatter(d.nr, d.C2_fallhoehe_integriert_m, s=55, marker="o", color=FARBEN[g],
                   edgecolor=FLAECHE, linewidth=1.5, zorder=3)
        ax.scatter(d.nr, d.C2_fallhoehe_obergrenze_m, s=65, marker="^", color=FARBEN[g],
                   edgecolor=FLAECHE, linewidth=1.5, zorder=3)
    ax.axhline(STURZWEG_THEORIE, color=TEXT, linewidth=1.2, linestyle="--")
    ax.text(15.4, STURZWEG_THEORIE, f"Geometrie: {STURZWEG_THEORIE:.1f} m ", ha="right", va="bottom",
            fontsize=9.5, color=TEXT)
    form = [
        plt.Line2D([], [], marker="^", linestyle="", color=TEXT2, markersize=8,
                   label="Obergrenze aus Fallzeit (g/2 · T², idealer freier Fall)"),
        plt.Line2D([], [], marker="o", linestyle="", color=TEXT2, markersize=7,
                   label="Aus gemessener Beschleunigung integriert"),
    ]
    erste = ax.legend(handles=form, loc="lower left", fontsize=9)
    ax.add_artist(erste)
    geraete_legende(ax, loc="lower right", fontsize=9)
    ax.set_ylim(0, 2.9)
    ax.set_xlim(0.4, 15.6)
    ax.set_xticks(range(1, 16))
    ax.set_xlabel("Messnummer")
    ax.set_ylabel("Fallhöhe bis zum ersten Seilzug [m]")
    ax.grid(axis="x", visible=False)
    titel(fig, "Gemessene Fallhöhe liegt in allen 15 Stürzen unter den 2,5 m aus der Geometrie",
          "Die Zeit vom Loslassen bis zum ersten Seilzug (0,48–0,64 s) reicht selbst im idealen "
          "freien Fall nicht für 2,5 m (dafür wären 0,71 s nötig).")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(ziel, dpi=150)
    plt.close(fig)


def plot_filtereinfluss(meta: pd.DataFrame, signale: list[tuple[Signal, Signal]], ziel: Path):
    """Spitzenbeschleunigung je Grenzfrequenz: robust bei der Kletternden, nicht bei der Sichernden."""
    stufen = [6, 10, 15, 20, 30, None]
    namen = ["6 Hz", "10 Hz", "15 Hz", "20 Hz", "30 Hz", "ungefiltert"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (k, person) in zip(axs, [(1, "C2"), (0, "C1")]):
        for g in GERAETE:
            idx = meta.index[meta.geraet == g]
            mittel = [
                np.mean([(signale[i][k].a_betrag_roh if fc is None
                          else tiefpass(signale[i][k].a_betrag_roh, fc)).max() for i in idx])
                for fc in stufen
            ]
            ax.plot(range(len(stufen)), mittel, color=FARBEN[g], marker="o", markersize=7,
                    markeredgecolor=FLAECHE, markeredgewidth=1.5, label=GERAETE_LANG[g])
        ax.set_xticks(range(len(stufen)), namen)
        ax.set_xlim(-0.3, len(stufen) - 0.4)
        ax.set_title(PERSONEN[person]["name"], loc="left")
        ax.set_xlabel("Grenzfrequenz des Tiefpassfilters")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Mittlere Spitze |a| [g]")
    axs[0].set_ylim(0, None)
    axs[0].legend(loc="lower left", fontsize=9)
    titel(fig, "Einfluss des Filters auf die Spitzenbeschleunigung",
          "Bei der Kletternden ist die Spitze vom Filter unabhängig. Bei der Sichernden bestimmt die "
          "Filterwahl das Ergebnis, deshalb wird ihre Spitze nicht als Kraft ausgewertet.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(ziel, dpi=150)
    plt.close(fig)


def plot_einzelmessung(vl: dict, zeile: pd.Series, k: dict, ziel: Path):
    g = zeile.geraet
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, person, name in [(axs[0], "c2", "Kletternde Person (C2)"), (axs[1], "c1", "Sichernde Person (C1)")]:
        ax.plot(vl[person]["t"], vl[person]["a_roh"], color="#a8a7a1", linewidth=1, label="ungefiltert")
        ax.plot(vl[person]["t"], vl[person]["a"], color=FARBEN[g], label="Tiefpass 15 Hz")
        ax.set_ylabel("|a| [g]")
        ax.set_title(name, loc="left")
        ax.legend(loc="upper right")
    axs[1].axvspan(*FLUGFENSTER, color=TEXT2, alpha=0.08, linewidth=0)
    axs[1].text(FLUGFENSTER[0], axs[1].get_ylim()[1] * 0.9, " Flugphase", fontsize=8, color=TEXT2)
    axs[2].plot(vl["c2"]["t"], vl["c2"]["s"], color=FARBEN[g], label="Kletternde (C2)")
    axs[2].plot(vl["c1"]["t"], vl["c1"]["s"], color=TEXT2, label="Sichernde (C1)")
    axs[2].set_ylabel("Vertikaler Weg [m]")
    hinweis = "" if k["C2"]["ruhephase_gefunden"] else " – Aufnahme endet vor der Ruhephase, Wege unsicherer"
    axs[2].set_title("Vertikalbewegung (doppelt integriert, driftkorrigiert)" + hinweis, loc="left")
    axs[2].legend(loc="center right")
    for ax in axs:
        for tm in vl["marken"].values():
            ax.axvline(tm, color=TEXT2, linewidth=0.8, linestyle=":")
    for name, tm in vl["marken"].items():
        axs[0].text(tm, axs[0].get_ylim()[1], f" {name}", rotation=90, va="top", ha="right",
                    fontsize=7, color=TEXT2)
    axs[2].set_xlim(-1.5, 2.5)
    axs[2].set_xlabel("Zeit relativ zu 'Seil greift' [s]")
    fig.suptitle(
        f"Messung {zeile.nr:02d} – {GERAETE_LANG[g]}, Versuch {zeile.versuch}  |  "
        f"F Kletternde = {k['C2']['F_peak_kN']:.2f} kN, Sturzweg = {k['C2']['gesamtweg_m']:.2f} m, "
        f"Anhebung = {k['C1']['anhebung_m']:.2f} m",
        x=0.01, ha="left", fontsize=11, color=TEXT,
    )
    fig.tight_layout()
    fig.savefig(ziel, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------
def main():
    stil()
    ERGEBNISSE.mkdir(exist_ok=True)
    (ERGEBNISSE / "einzelmessungen").mkdir(exist_ok=True)

    meta = finde_messungen()
    kennzeilen, kennwerte, verlaeufe, signale = [], [], [], []
    for _, zeile in meta.iterrows():
        c1, c2 = lade_signal(zeile.C1), lade_signal(zeile.C2)
        k, vl = auswerten(c1, c2)
        signale.append((c1, c2))
        kennwerte.append(k)
        verlaeufe.append(vl)
        eintrag = {"nr": zeile.nr, "geraet": zeile.geraet, "versuch": zeile.versuch,
                   "datei_zeit": zeile.zeitstempel}
        for person, werte in k.items():
            for name, wert in werte.items():
                if not name.startswith("t_"):
                    eintrag[f"{person}_{name}"] = wert
        kennzeilen.append(eintrag)
        plot_einzelmessung(vl, zeile, k, ERGEBNISSE / "einzelmessungen" /
                           f"messung_{zeile.nr:02d}_{zeile.geraet}.png")

    df = pd.DataFrame(kennzeilen)
    df.to_csv(ERGEBNISSE / "kennwerte_einzelmessungen.csv", index=False, sep=";", decimal=",",
              float_format="%.4f")
    st = statistik(df)
    st.to_csv(ERGEBNISSE / "statistik_geraetevergleich.csv", index=False, sep=";", decimal=",",
              float_format="%.4f")

    plot_zeitverlaeufe(verlaeufe, kennwerte, meta, ERGEBNISSE / "01_zeitverlaeufe.png")
    plot_kennwerte(df, st, ERGEBNISSE / "02_kennwerte.png")
    plot_reihenfolge(df, st, ERGEBNISSE / "03_reihenfolge.png")
    plot_fallhoehe(df, ERGEBNISSE / "04_fallhoehe.png")
    plot_filtereinfluss(meta, signale, ERGEBNISSE / "05_filtereinfluss.png")

    # Konsolenausgabe
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = lambda x: f"{x:.2f}"  # noqa: E731
    print("Gerätevergleich (Mittelwert und Standardabweichung, n = 5 je Gerät)\n")
    print(st.to_string(index=False, float_format=fmt))

    print("\nFallhöhe bis zum ersten Seilzug")
    print(f"  Geometrie (2 x {STURZHOEHE - SICHERUNGSHOEHE:.2f} m + {SCHLAPPSEIL:.2f} m Schlappseil): "
          f"{STURZWEG_THEORIE:.2f} m, dafür nötig {np.sqrt(2 * STURZWEG_THEORIE / G):.2f} s freier Fall")
    print(f"  Gemessene Zeit Loslassen bis Seilzug: {df.C2_freifallzeit_s.min():.2f}–"
          f"{df.C2_freifallzeit_s.max():.2f} s (Mittel {df.C2_freifallzeit_s.mean():.2f} s)")
    print(f"  Obergrenze daraus (idealer freier Fall): {df.C2_fallhoehe_obergrenze_m.min():.2f}–"
          f"{df.C2_fallhoehe_obergrenze_m.max():.2f} m (Mittel {df.C2_fallhoehe_obergrenze_m.mean():.2f} m)")
    print(f"  Aus der Beschleunigung integriert: {df.C2_fallhoehe_integriert_m.min():.2f}–"
          f"{df.C2_fallhoehe_integriert_m.max():.2f} m (Mittel {df.C2_fallhoehe_integriert_m.mean():.2f} m)")

    print("\nQualität der Integration")
    print(f"  Driftkorrektur Kletternde: bis {df.C2_drift_ms.abs().max():.2f} m/s, "
          f"Sichernde: bis {df.C1_drift_ms.abs().max():.2f} m/s")
    ohne = df.loc[~(df.C2_ruhephase_gefunden & df.C1_ruhephase_gefunden), "nr"].tolist()
    if ohne:
        print(f"  Aufnahme endet vor der Ruhephase bei Messung: {ohne} (Wege dort unsicherer)")
    print(f"\nErgebnisse gespeichert in: {ERGEBNISSE}")


if __name__ == "__main__":
    main()
