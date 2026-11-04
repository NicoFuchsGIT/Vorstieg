"""
Auswertung der IMU-Daten aus den Sturzversuchen im Vorstieg.

Versuchsaufbau
--------------
* C1 = sichernde Person, 70 kg, IMU am Gurt/an der Hüfte (Dateien "C1-Rleg_...")
* C2 = kletternde Person, 78 kg, IMU am Gurt/an der Hüfte (Dateien "C2-Lski_...")
* Sturz aus 9,80 m, letzte Zwischensicherung auf 8,65 m
  -> 1,15 m über der Sicherung, also etwa 2 * 1,15 m + 0,20 m Schlappseil ≈ 2,5 m
     Sturzweg ohne Seildehnung
* Dynamisch gesichert, sichernde Person 1–2 m von der Wand entfernt
* Messungen in zeitlicher Reihenfolge: 1–5 Petzl GriGri, 6–10 Wild Country Revo,
  11–15 Petzl Reverso (Tube)

Physik hinter den Kennwerten
----------------------------
Ein Beschleunigungssensor misst die spezifische Kraft (Beschleunigung ohne den
Anteil der Schwerkraft). Im Ruhezustand zeigt er also 1 g, im freien Fall 0 g.
Multipliziert man den Betrag mit der Körpermasse, erhält man die gesamte äußere
Kraft auf den Körper, die nicht aus der Schwerkraft stammt, also Seil plus
Wand- bzw. Bodenkontakt:

    F ≈ m * |a_gemessen|

Dreht man die Sensorbeschleunigung mit den Euler-Winkeln (Xsens, ZYX) ins
Weltsystem und zieht g ab, bleibt die vertikale Beschleunigung. Aus ihr ergeben
sich durch Integration Geschwindigkeit und Weg: Fallweg, Bremsweg und das
Anheben der sichernden Person. Die Drift wird über eine Ruhephase vor und nach
dem Sturz korrigiert. Die Wege sind deshalb Schätzwerte (Größenordnung ± 10–20 %).

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


def integriere(sig: Signal, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertikale Geschwindigkeit und Weg (Ruhe bei i0 und i1, Drift linear korrigiert)."""
    a = sig.a_vert.copy()
    a -= np.median(a[max(0, i0 - int(0.4 * FS)) : i0 + 1])  # Offset aus der Ruhephase
    v = np.zeros_like(a)
    v[i0 + 1 :] = np.cumsum((a[i0:-1] + a[i0 + 1 :]) / 2) / FS
    # Drift: v(i1) muss 0 sein (Ruhe), linear zwischen i0 und i1 verteilt
    drift = np.zeros_like(v)
    drift[i0 : i1 + 1] = np.linspace(0, v[i1], i1 - i0 + 1)
    drift[i1 + 1 :] = v[i1]
    v -= drift
    v[:i0] = 0
    s = np.zeros_like(v)
    s[i0 + 1 :] = np.cumsum((v[i0:-1] + v[i0 + 1 :]) / 2) / FS
    return v, s


def auswerten(c1: Signal, c2: Signal) -> tuple[dict, dict]:
    t = c2.t
    # ---- Kletternde Person: freier Fall = längster Bereich mit |a| < 0,5 g
    bereiche = zusammenhaengende_bereiche(c2.a_betrag < 0.5)
    # Lücken < 0,15 s überbrücken (kurzes Anschlagen im freien Fall)
    verbunden: list[list[int]] = []
    for s, e in bereiche:
        if verbunden and (s - verbunden[-1][1]) < 0.15 * FS:
            verbunden[-1][1] = e
        else:
            verbunden.append([s, e])
    ff_s, ff_e = max(verbunden, key=lambda b: b[1] - b[0])

    # Loslassen: letzter Zeitpunkt vor dem freien Fall mit |a| >= 0,9 g
    i_los = ff_s
    while i_los > 0 and c2.a_betrag[i_los] < 0.9:
        i_los -= 1
    # Seil beginnt zu greifen: |a| steigt wieder über 1 g -> maximale Fallgeschwindigkeit
    i_vmax = ff_e
    while i_vmax < len(t) - 1 and c2.a_betrag[i_vmax] < 1.0:
        i_vmax += 1

    i_peak2 = i_vmax + int(np.argmax(c2.a_betrag[i_vmax : i_vmax + int(1.0 * FS)]))
    i_ruhe2 = finde_ruhe_nach(c2.a_betrag, i_peak2 + int(0.3 * FS))
    i_start = max(0, i_los - int(0.1 * FS))
    v2, s2 = integriere(c2, i_start, i_ruhe2)
    i_vmax_int = i_los + int(np.argmin(v2[i_los : i_peak2 + 1]))
    i_tief = i_vmax_int + int(np.argmin(s2[i_vmax_int : i_ruhe2 + 1]))  # tiefster Punkt
    i_stop = i_tief  # tiefster Punkt = Vertikalgeschwindigkeit 0

    m2 = PERSONEN["C2"]["masse"]
    ruck2 = np.gradient(c2.a_betrag, 1 / FS)
    k2 = {
        "t_loslassen": t[i_los],
        "t_seil": t[i_vmax],
        "t_peak": t[i_peak2],
        "fallzeit_s": t[i_vmax] - t[i_los],
        "v_max_ms": -v2[i_vmax_int],
        "fallweg_m": -s2[i_vmax_int],
        "bremszeit_s": t[i_stop] - t[i_vmax_int],
        "bremsweg_m": s2[i_vmax_int] - s2[i_tief],
        "gesamtweg_m": -s2[i_tief],
        "a_peak_g": c2.a_betrag[i_peak2],
        "a_peak_roh_g": c2.a_betrag_roh[i_vmax : i_vmax + int(1.0 * FS)].max(),
        "F_peak_kN": m2 * G * c2.a_betrag[i_peak2] / 1000,
        "anstiegszeit_s": t[i_peak2] - t[i_vmax],
        "ruck_max_g_s": ruck2[i_vmax:i_peak2 + 1].max(),
        "a_mittel_brems_g": np.mean(c2.a_betrag[i_vmax_int:i_stop + 1]) if i_stop > i_vmax_int else np.nan,
    }

    # ---- Sichernde Person: Zeitbasis an C2 anpassen (gemeinsame Uhr)
    tc = c1.t
    j_los = int(np.searchsorted(tc, t[i_los]))
    j_seil = int(np.searchsorted(tc, t[i_vmax]))
    # Reaktionsbeginn: erste Abweichung > 0,25 g von 1 g nach dem Loslassen
    abw = np.abs(c1.a_betrag - 1.0) > 0.25
    j_react = j_los + int(np.argmax(abw[j_los:]))
    fenster = slice(j_los, min(len(tc), j_seil + int(1.2 * FS)))
    j_peak1 = fenster.start + int(np.argmax(c1.a_betrag[fenster]))
    j_peak1_roh = fenster.start + int(np.argmax(c1.a_betrag_roh[fenster]))
    j_ruhe1 = finde_ruhe_nach(c1.a_betrag, j_peak1 + int(0.3 * FS))
    j_start = max(0, j_los - int(0.1 * FS))
    v1, s1 = integriere(c1, j_start, j_ruhe1)
    j_hoch = j_start + int(np.argmax(s1[j_start : j_ruhe1 + 1]))

    m1 = PERSONEN["C1"]["masse"]
    ruck1 = np.gradient(c1.a_betrag, 1 / FS)
    k1 = {
        "t_reaktion": tc[j_react],
        "t_peak": tc[j_peak1],
        "reaktion_nach_loslassen_s": tc[j_react] - t[i_los],
        "reaktion_vs_seil_s": tc[j_react] - t[i_vmax],
        "a_peak_g": c1.a_betrag[j_peak1],
        "a_peak_roh_g": c1.a_betrag_roh[j_peak1_roh],
        "F_peak_kN": m1 * G * c1.a_betrag[j_peak1] / 1000,
        "ruck_max_g_s": ruck1[fenster].max(),
        "anhebung_m": s1[j_hoch],
        "v_hoch_max_ms": v1[j_start : j_ruhe1 + 1].max(),
        "peak_versatz_zu_C2_s": tc[j_peak1] - t[i_peak2],
        "dauer_belastung_s": tc[j_ruhe1] - tc[j_react],
    }
    # Kraftverhältnis Sichernde/Kletternde (Seilreibung an der Zwischensicherung)
    k1["kraftverhaeltnis_C1_C2"] = k1["F_peak_kN"] / k2["F_peak_kN"]

    verlauf = {
        "c2": {"t": t - t[i_vmax], "a": c2.a_betrag, "a_roh": c2.a_betrag_roh, "v": v2, "s": s2},
        "c1": {"t": tc - t[i_vmax], "a": c1.a_betrag, "a_roh": c1.a_betrag_roh, "v": v1, "s": s1},
        "marken": {
            "Loslassen": t[i_los] - t[i_vmax],
            "Seil greift (v max)": 0.0,
            "Tiefster Punkt": t[i_tief] - t[i_vmax],
            "Reaktion C1": tc[j_react] - t[i_vmax],
        },
    }
    return {"C1": k1, "C2": k2}, verlauf


# --------------------------------------------------------------------------
# Statistik
# --------------------------------------------------------------------------
KENNWERTE = {
    # (person, spalte): (Beschriftung, Einheit)
    ("C2", "F_peak_kN"): ("Max. Kraft Kletternde", "kN"),
    ("C1", "F_peak_kN"): ("Max. Kraft Sichernde", "kN"),
    ("C2", "a_peak_g"): ("Max. Beschl. Kletternde", "g"),
    ("C1", "a_peak_g"): ("Max. Beschl. Sichernde", "g"),
    ("C1", "a_peak_roh_g"): ("Max. Beschl. Sichernde (ungefiltert)", "g"),
    ("C2", "ruck_max_g_s"): ("Max. Ruck Kletternde", "g/s"),
    ("C2", "fallzeit_s"): ("Fallzeit (Loslassen bis Seil greift)", "s"),
    ("C2", "v_max_ms"): ("Max. Fallgeschwindigkeit", "m/s"),
    ("C2", "fallweg_m"): ("Fallweg bis Seil greift", "m"),
    ("C2", "bremszeit_s"): ("Bremszeit (v max bis v = 0)", "s"),
    ("C2", "bremsweg_m"): ("Bremsweg", "m"),
    ("C2", "gesamtweg_m"): ("Gesamter Sturzweg", "m"),
    ("C2", "anstiegszeit_s"): ("Anstiegszeit bis Kraftspitze", "s"),
    ("C1", "anhebung_m"): ("Anhebung Sichernde", "m"),
    ("C1", "reaktion_vs_seil_s"): ("Zug bei Sichernder rel. zu 'Seil greift'", "s"),
    ("C1", "kraftverhaeltnis_C1_C2"): ("Kraftverhältnis Sichernde / Kletternde", "–"),
}


def statistik(df: pd.DataFrame) -> pd.DataFrame:
    zeilen = []
    for (person, spalte), (label, einheit) in KENNWERTE.items():
        col = f"{person}_{spalte}"
        gruppen = [df.loc[df.geraet == g, col].dropna().to_numpy() for g in GERAETE]
        zeile = {"Kennwert": label, "Einheit": einheit}
        for g, werte in zip(GERAETE, gruppen):
            zeile[f"{g} MW"] = werte.mean()
            zeile[f"{g} SD"] = werte.std(ddof=1)
        zeile["ANOVA p"] = stats.f_oneway(*gruppen).pvalue
        zeile["Kruskal-Wallis p"] = stats.kruskal(*gruppen).pvalue
        # paarweise Mann-Whitney-U-Tests mit Holm-Korrektur
        paare = [(0, 1), (0, 2), (1, 2)]
        p_roh = [stats.mannwhitneyu(gruppen[a], gruppen[b]).pvalue for a, b in paare]
        reihenfolge = np.argsort(p_roh)
        p_holm = np.empty(3)
        laufend = 0.0
        for rang, idx in enumerate(reihenfolge):
            laufend = max(laufend, min(1.0, p_roh[idx] * (3 - rang)))
            p_holm[idx] = laufend
        for (a, b), p in zip(paare, p_holm):
            zeile[f"p {GERAETE[a]}–{GERAETE[b]} (Holm)"] = p
        zeilen.append(zeile)
    return pd.DataFrame(zeilen)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GITTER = "#e4e3df"


def stil():
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
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


def gemeinsame_zeitachse(t_lokal: np.ndarray, y: np.ndarray, t_gitter: np.ndarray) -> np.ndarray:
    return np.interp(t_gitter, t_lokal, y, left=np.nan, right=np.nan)


def plot_zeitverlaeufe(verlaeufe: list[dict], meta: pd.DataFrame, ziel: Path):
    """Mittelwertkurven ± SD je Gerät: Beschleunigung C2/C1, Geschwindigkeit C2, Weg C1."""
    tg = np.arange(-1.0, 1.8, 1 / FS)
    zeilen = [
        ("c2", "a", "Beschleunigung |a| [g]", "Kletternde Person (C2) – Beschleunigung"),
        ("c1", "a", "Beschleunigung |a| [g]", "Sichernde Person (C1) – Beschleunigung"),
        ("c2", "v", "Vertikalgeschw. [m/s]", "Kletternde Person (C2) – Vertikalgeschwindigkeit"),
        ("c1", "s", "Vertikaler Weg [m]", "Sichernde Person (C1) – Anhebung"),
    ]
    fig, axs = plt.subplots(len(zeilen), 1, figsize=(10, 13), sharex=True)
    for ax, (person, groesse, ylabel, titel) in zip(axs, zeilen):
        for g in GERAETE:
            kurven = np.array(
                [
                    gemeinsame_zeitachse(verlaeufe[i][person]["t"], verlaeufe[i][person][groesse], tg)
                    for i in meta.index[meta.geraet == g]
                ]
            )
            mw, sd = np.nanmean(kurven, axis=0), np.nanstd(kurven, axis=0, ddof=1)
            ax.fill_between(tg, mw - sd, mw + sd, color=FARBEN[g], alpha=0.15, linewidth=0)
            ax.plot(tg, mw, color=FARBEN[g], label=GERAETE_LANG[g])
        ax.axvline(0, color=TEXT2, linewidth=1, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.set_title(titel, loc="left")
    axs[0].text(0.01, axs[0].get_ylim()[1] * 0.92, "Seil greift", color=TEXT2, fontsize=9)
    axs[0].legend(loc="upper right")
    axs[-1].set_xlabel("Zeit relativ zu 'Seil greift' (max. Fallgeschwindigkeit) [s]")
    fig.suptitle("Mittelwert ± Standardabweichung je Sicherungsgerät (n = 5)", x=0.01, ha="left",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(ziel, dpi=150)
    plt.close(fig)


def plot_ueberlagerung(verlaeufe: list[dict], meta: pd.DataFrame, ziel: Path):
    """Alle 5 Versuche je Gerät überlagert (gefiltert), getrennt nach Person."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True, sharey="row")
    for spalte, g in enumerate(GERAETE):
        for zeile, person in enumerate(["c2", "c1"]):
            ax = axs[zeile, spalte]
            for i in meta.index[meta.geraet == g]:
                vl = verlaeufe[i][person]
                ax.plot(vl["t"], vl["a"], color=FARBEN[g], linewidth=1.2, alpha=0.8)
            ax.axvline(0, color=TEXT2, linewidth=1, linestyle="--")
            ax.set_xlim(-1.0, 1.8)
            if spalte == 0:
                ax.set_ylabel(f"{PERSONEN[person.upper()]['name']}\n|a| [g]")
            if zeile == 0:
                ax.set_title(GERAETE_LANG[g], loc="left", color=TEXT)
            else:
                ax.set_xlabel("Zeit rel. zu 'Seil greift' [s]")
    fig.suptitle("Einzelversuche (Tiefpass 15 Hz) – je Linie ein Sturz", x=0.01, ha="left",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(ziel, dpi=150)
    plt.close(fig)


def plot_kennwerte(df: pd.DataFrame, ziel: Path):
    auswahl = [
        ("C2", "F_peak_kN"), ("C1", "F_peak_kN"), ("C1", "kraftverhaeltnis_C1_C2"),
        ("C2", "ruck_max_g_s"), ("C2", "fallzeit_s"), ("C2", "v_max_ms"),
        ("C2", "bremszeit_s"), ("C2", "bremsweg_m"), ("C2", "gesamtweg_m"),
        ("C1", "anhebung_m"), ("C1", "a_peak_roh_g"), ("C1", "reaktion_vs_seil_s"),
    ]
    fig, axs = plt.subplots(4, 3, figsize=(13, 14))
    rng = np.random.default_rng(0)
    for ax, (person, spalte) in zip(axs.flat, auswahl):
        label, einheit = KENNWERTE[(person, spalte)]
        col = f"{person}_{spalte}"
        werte = [df.loc[df.geraet == g, col].to_numpy() for g in GERAETE]
        bp = ax.boxplot(werte, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": TEXT, "linewidth": 1.5},
                        whiskerprops={"color": TEXT2}, capprops={"color": TEXT2},
                        boxprops={"edgecolor": TEXT2})
        for patch, g in zip(bp["boxes"], GERAETE):
            patch.set_facecolor(FARBEN[g] + "33")
        for k, (g, w) in enumerate(zip(GERAETE, werte)):
            x = k + 1 + rng.uniform(-0.12, 0.12, len(w))
            ax.scatter(x, w, s=40, color=FARBEN[g], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
            ax.text(k + 1.3, np.mean(w), f"Ø {np.mean(w):.2f}", va="center", fontsize=8, color=TEXT2)
        ax.set_xticks([1, 2, 3], GERAETE)
        ax.set_title(f"{label} [{einheit}]", loc="left", fontsize=10)
        ax.set_xlim(0.5, 3.8)
    fig.suptitle("Kennwerte je Sicherungsgerät (Punkte = Einzelstürze)", x=0.01, ha="left",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(ziel, dpi=150)
    plt.close(fig)


def plot_einzelmessung(vl: dict, zeile: pd.Series, k: dict, ziel: Path):
    g = zeile.geraet
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, person, titel in [(axs[0], "c2", "Kletternde Person (C2)"), (axs[1], "c1", "Sichernde Person (C1)")]:
        ax.plot(vl[person]["t"], vl[person]["a_roh"], color="#a8a7a1", linewidth=1, label="ungefiltert")
        ax.plot(vl[person]["t"], vl[person]["a"], color=FARBEN[g], label="Tiefpass 15 Hz")
        ax.set_ylabel("|a| [g]")
        ax.set_title(titel, loc="left")
        ax.legend(loc="upper right")
    axs[2].plot(vl["c2"]["t"], vl["c2"]["s"], color=FARBEN[g], label="Kletternde (C2)")
    axs[2].plot(vl["c1"]["t"], vl["c1"]["s"], color=TEXT2, label="Sichernde (C1)")
    axs[2].set_ylabel("Vertikaler Weg [m]")
    axs[2].set_title("Vertikalbewegung (doppelt integriert, driftkorrigiert)", loc="left")
    axs[2].legend(loc="center right")
    for ax in axs:
        for name, tm in vl["marken"].items():
            ax.axvline(tm, color=TEXT2, linewidth=0.8, linestyle=":")
    for name, tm in vl["marken"].items():
        axs[0].text(tm, axs[0].get_ylim()[1], f" {name}", rotation=90, va="top", fontsize=7, color=TEXT2)
    axs[2].set_xlim(-1.5, 2.5)
    axs[2].set_xlabel("Zeit relativ zu 'Seil greift' [s]")
    fig.suptitle(
        f"Messung {zeile.nr:02d} – {GERAETE_LANG[g]}, Versuch {zeile.versuch}  |  "
        f"F C2 = {k['C2']['F_peak_kN']:.2f} kN, F C1 = {k['C1']['F_peak_kN']:.2f} kN, "
        f"Sturzweg = {k['C2']['gesamtweg_m']:.2f} m",
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
    kennzeilen, verlaeufe = [], []
    for _, zeile in meta.iterrows():
        c1, c2 = lade_signal(zeile.C1), lade_signal(zeile.C2)
        k, vl = auswerten(c1, c2)
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

    plot_zeitverlaeufe(verlaeufe, meta, ERGEBNISSE / "01_mittelwertkurven.png")
    plot_ueberlagerung(verlaeufe, meta, ERGEBNISSE / "02_einzelversuche_ueberlagert.png")
    plot_kennwerte(df, ERGEBNISSE / "03_kennwerte_boxplots.png")

    # Konsolenausgabe
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Theoretischer Sturzweg ohne Seildehnung: {STURZWEG_THEORIE:.2f} m "
          f"-> v_theo = {np.sqrt(2 * G * STURZWEG_THEORIE):.2f} m/s, "
          f"t_theo = {np.sqrt(2 * STURZWEG_THEORIE / G):.2f} s\n")
    anzeige = st[["Kennwert", "Einheit"] + [f"{g} {x}" for g in GERAETE for x in ("MW", "SD")]
                 + ["Kruskal-Wallis p"]]
    print(anzeige.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nErgebnisse gespeichert in: {ERGEBNISSE}")


if __name__ == "__main__":
    main()
