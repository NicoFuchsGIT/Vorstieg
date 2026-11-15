"""
Erzeugt das Poster zum Vorstiegssturz-Versuch aus der FH-Vorlage:
vier Abbildungen, eine Tabelle und der ausgefüllte Text (IMRAD).

Aufruf:  python poster.py
Ergebnis: poster/Poster_Vorstieg.docx, Abbildungen in poster/bilder/
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from matplotlib.ticker import FuncFormatter
from scipy import stats

import auswertung as A

POSTER = A.BASIS / "poster"
BILDER = POSTER / "bilder"
VORLAGE = POSTER / "Poster-Template_STPAA.docx"
ZIEL = POSTER / "Poster_Vorstieg.docx"

TITEL = "Fangstoß beim Vorstiegssturz"
UNTERTITEL = "IMU-Messung an kletternder und sichernder Person mit drei Sicherungsgeräten"
TEAM = "Kilian Chalmers, Nico Fuchs, Vincent Pflug, Luisa Przibylla"
KURS = "Sportgerätepraxis: Theorie, Anwendung, Analyse – BHF 5"

SPALTE_CM = 12.2  # Spaltenbreite der Vorlage (A3, zweispaltig)
BEISPIEL_NR = 4  # Beispielsturz für Abb. 2
KURZ = {"GriGri": "GriGri", "Revo": "Revo", "Reverso": "Reverso"}


def de(x: float, n: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def komma_achse(ax, n_x: int | None = None, n_y: int | None = None):
    if n_x is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, n_x)))
    if n_y is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, n_y)))


def poster_stil():
    A.stil()
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.dpi": 400, "lines.linewidth": 1.6})


def cm(x):  # cm -> Zoll
    return x / 2.54


# --------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------
def lade_alles():
    meta = A.finde_messungen()
    kenn, verl, zeilen = [], [], []
    for _, z in meta.iterrows():
        k, vl = A.auswerten(A.lade_signal(z.C1), A.lade_signal(z.C2))
        kenn.append(k)
        verl.append(vl)
        e = {"nr": z.nr, "geraet": z.geraet}
        for person, werte in k.items():
            for name, wert in werte.items():
                e[f"{person}_{name}"] = wert
        zeilen.append(e)
    return meta, pd.DataFrame(zeilen), kenn, verl


TABELLE = [
    # Spalte, Beschriftung, Nachkommastellen
    ("C2_F_peak_kN", "Max. Kraft Kletternde F_max [kN]", 2),
    ("C1_F_flugfenster_kN", "Kraft Sichernde (Flug) F_S [kN]", 2),
    ("C2_anstiegszeit_s", "Seilzug bis Kraftspitze [s]", 2),
    ("C2_bremsweg_m", "Bremsweg [m]", 2),
    ("C2_gesamtweg_m", "Gesamter Sturzweg [m]", 2),
    ("C1_anhebung_m", "Anhebung Sichernde [m]", 2),
]


def gruppen(df, col):
    return [df.loc[df.geraet == g, col].to_numpy() for g in A.GERAETE]


def kw_p(df, col):
    return stats.kruskal(*gruppen(df, col)).pvalue


def groesste_differenz(df, col):
    """Größter Mittelwertunterschied zwischen zwei Geräten mit 95-%-Konfidenzintervall (Welch)."""
    g = gruppen(df, col)
    best = None
    for a in range(3):
        for b in range(a + 1, 3):
            d = g[a].mean() - g[b].mean()
            if best is None or abs(d) > abs(best[0]):
                va, vb = g[a].var(ddof=1) / 5, g[b].var(ddof=1) / 5
                se = np.sqrt(va + vb)
                fg = (va + vb) ** 2 / (va**2 / 4 + vb**2 / 4)
                h = stats.t.ppf(0.975, fg) * se
                best = (d, d - h, d + h, A.GERAETE[a], A.GERAETE[b])
    return best


# --------------------------------------------------------------------------
# Abbildungen
# --------------------------------------------------------------------------
def abb1_aufbau(ziel: Path):
    bilder = [("video_a_vor_sturz.png", "(a) vor dem Sturz"),
              ("video_b_freier_fall.png", "(b) freier Fall"),
              ("video_c_fangen.png", "(c) Fangen")]
    fig, axs = plt.subplots(1, 3, figsize=(cm(SPALTE_CM), cm(5.3)))
    for ax, (datei, text) in zip(axs, bilder):
        ax.imshow(mpimg.imread(BILDER / datei)[110:760])  # oben/unten beschnitten
        ax.set_title(text, fontsize=8.5, loc="left", pad=3)
        ax.axis("off")
    kw = dict(fontsize=7.5, color="white", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.25", fc="#0b0b0b", ec="none", alpha=0.75))
    pfeil = dict(arrowstyle="-", color="white", lw=1)
    axs[0].annotate("C2 kletternd\n78 kg", xy=(285, 15), xytext=(30, 210), arrowprops=pfeil, **kw)
    axs[0].annotate("C1 sichernd\n70 kg", xy=(440, 610), xytext=(110, 480), arrowprops=pfeil, **kw)
    axs[2].annotate("C1 wird\nangehoben", xy=(350, 550), xytext=(30, 610), arrowprops=pfeil, **kw)
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0, wspace=0.03)
    fig.savefig(ziel)
    plt.close(fig)


def abb2_beispiel(verl, kenn, meta, ziel: Path):
    i = int(meta.index[meta.nr == BEISPIEL_NR][0])
    vl, k = verl[i], kenn[i]
    m = vl["marken"]
    farbe_c2, farbe_c1 = A.FARBEN["GriGri"], "#52514e"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(cm(SPALTE_CM), cm(6.6)), sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1]})
    for ax in (ax1, ax2):
        ax.axvspan(m["Loslassen"], m["Seilzug beginnt"], color="#2a78d6", alpha=0.08, lw=0)
        ax.axvspan(m["Seilzug beginnt"], m["Tiefster Punkt"], color="#eb6834", alpha=0.08, lw=0)
    ax1.plot(vl["c1"]["t"], vl["c1"]["a"], color=farbe_c1, lw=1.2, label="Sichernde (C1)")
    ax1.plot(vl["c2"]["t"], vl["c2"]["a"], color=farbe_c2, label="Kletternde (C2)")
    ax1.axhline(1, color=A.GITTER, lw=0.8, zorder=0)
    spitze = k["C2"]["a_peak_g"]
    tp = vl["c2"]["t"][np.argmax(np.where((vl["c2"]["t"] > 0) & (vl["c2"]["t"] < 1), vl["c2"]["a"], 0))]
    ax1.annotate(rf"$F_\mathrm{{max}}$ = {de(k['C2']['F_peak_kN'])} kN", xy=(tp, spitze), xytext=(1.05, 2.75),
                 fontsize=8, color=A.TEXT, arrowprops=dict(arrowstyle="-", color=A.TEXT2, lw=0.8))
    oben = 3.4
    ax1.set_ylim(0, oben)
    ax1.text((m["Loslassen"] + m["Seilzug beginnt"]) / 2, oben * 0.97, "freier Fall", ha="center",
             va="top", fontsize=8, color=A.TEXT2)
    ax1.text((m["Seilzug beginnt"] + m["Tiefster Punkt"]) / 2, oben * 0.97, "Bremsphase", ha="center",
             va="top", fontsize=8, color=A.TEXT2)
    ax1.set_ylabel("|a| [g]")
    ax1.set_yticks([0, 1, 2, 3])
    ax1.legend(loc="center right", handlelength=1.4, bbox_to_anchor=(1.0, 0.52))
    ax2.plot(vl["c1"]["t"], vl["c1"]["s"], color=farbe_c1, lw=1.2)
    ax2.plot(vl["c2"]["t"], vl["c2"]["s"], color=farbe_c2)
    ax2.axhline(0, color=A.GITTER, lw=0.8, zorder=0)
    ax2.text(2.0, k["C1"]["anhebung_m"] + 0.25, f"C1: +{de(k['C1']['anhebung_m'])} m", ha="right",
             fontsize=8, color=A.TEXT2)
    ax2.text(2.0, -k["C2"]["gesamtweg_m"] + 0.3, f"C2: −{de(k['C2']['gesamtweg_m'])} m", ha="right",
             fontsize=8, color=A.TEXT2)
    ax2.set_ylabel("Vertikaler Weg z [m]")
    ax2.set_xlabel("Zeit relativ zu „Seil greift“ [s]")
    ax2.set_xlim(-1.2, 2.0)
    ax2.set_ylim(-4.6, 1.8)
    komma_achse(ax1, n_y=0)
    komma_achse(ax2, n_x=1, n_y=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(ziel)
    plt.close(fig)


def abb3_verlaeufe(verl, df, meta, ziel: Path):
    fig, axs = plt.subplots(1, 3, figsize=(cm(SPALTE_CM), cm(4.3)), sharey=True)
    for ax, g in zip(axs, A.GERAETE):
        idx = meta.index[meta.geraet == g]
        for i in idx:
            v = verl[i]["c2"]
            mm = (v["t"] >= -0.9) & (v["t"] <= 1.5)
            ax.plot(v["t"][mm], v["a"][mm], color=A.FARBEN[g], lw=0.9, alpha=0.85)
        mw = df.loc[df.geraet == g, "C2_a_peak_g"].mean()
        ax.axhline(mw, color=A.TEXT2, lw=0.7, ls=":")
        ax.text(1.45, 0.08, f"Ø Spitze {de(mw)} g", ha="right", va="bottom", fontsize=7.5, color=A.TEXT2)
        ax.axvline(0, color=A.TEXT2, lw=0.7, ls="--")
        ax.set_title(A.GERAETE_LANG[g].replace(" (Tube)", ""), loc="left", fontsize=8.5)
        ax.set_xlim(-0.9, 1.5)
        ax.set_xticks([-0.5, 0, 0.5, 1.0])
        komma_achse(ax, n_x=1, n_y=0)
    axs[0].set_ylim(0, 3.0)
    axs[0].set_yticks([0, 1, 2, 3])
    axs[0].set_ylabel("|a| Kletternde [g]")
    axs[1].set_xlabel("Zeit relativ zu „Seil greift“ [s]")
    fig.tight_layout(pad=0.4, w_pad=0.6)
    fig.savefig(ziel)
    plt.close(fig)


def abb4_kennwerte(df, ziel: Path):
    auswahl = [("C2_F_peak_kN", r"Kraft $F_\mathrm{max}$ [kN]", 1),
               ("C2_gesamtweg_m", "Sturzweg [m]", 0),
               ("C1_anhebung_m", "Anhebung C1 [m]", 1)]
    fig, axs = plt.subplots(1, 3, figsize=(cm(SPALTE_CM), cm(4.0)))
    rng = np.random.default_rng(1)
    for ax, (col, label, n) in zip(axs, auswahl):
        oben = 0
        for kx, g in enumerate(A.GERAETE):
            w = df.loc[df.geraet == g, col].to_numpy()
            ax.scatter(kx - 0.16 + rng.uniform(-0.07, 0.07, 5), w, s=16, color=A.FARBEN[g],
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.errorbar(kx + 0.2, w.mean(), yerr=w.std(ddof=1), fmt="o", color=A.TEXT, ms=3.5,
                        capsize=2.5, lw=1, zorder=4)
            oben = max(oben, w.max())
        ax.set_xticks(range(3), [KURZ[g] for g in A.GERAETE], fontsize=7.5)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(0, oben * 1.12)
        ax.set_title(label, loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
        komma_achse(ax, n_y=n)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(ziel)
    plt.close(fig)


# --------------------------------------------------------------------------
# Word-Dokument
# --------------------------------------------------------------------------
class Poster:
    def __init__(self):
        self.doc = Document(str(VORLAGE))
        body = self.doc.element.body
        kinder = list(body)
        # Titelabschnitt (bis zum ersten Abschnittswechsel) behalten, Rest leeren
        ende_titel = next(i for i, el in enumerate(kinder) if el.findall(".//" + qn("w:sectPr")))
        self.titel_abs = [el for el in kinder[: ende_titel + 1] if el.tag == qn("w:p")]
        for el in kinder[ende_titel + 1 :]:
            if el.tag != qn("w:sectPr"):
                body.remove(el)
        self.n_abb = 0
        # Vorlage ist für wenig Text ausgelegt (12 pt); für IMRAD mit 4 Abbildungen etwas kleiner
        normal = self.doc.styles["Normal"]
        normal.font.size = Pt(9.5)
        normal.paragraph_format.line_spacing = 1.12
        for name, pt in (("Caption", 8.5), ("References", 8), ("Heading 2", 15), ("Heading 3", 13)):
            self.doc.styles[name].font.size = Pt(pt)
            self.doc.styles[name].paragraph_format.space_before = Pt(4)
            self.doc.styles[name].paragraph_format.space_after = Pt(3)

    def kopf(self):
        from docx.text.paragraph import Paragraph
        ps = [Paragraph(el, self.doc._body) for el in self.titel_abs]

        def setze(p, text):
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            p.runs[0].text = text
        setze(ps[0], TITEL)
        setze(ps[1], TEAM)
        setze(ps[2], KURS)
        for leer in ps[3:]:
            leer.paragraph_format.line_spacing = Pt(8)
            leer.paragraph_format.space_after = Pt(0)
        # Untertitel zwischen Titel und Team
        neu = copy.deepcopy(ps[1]._p)
        ps[0]._p.addnext(neu)
        unter = Paragraph(neu, self.doc._body)
        setze(unter, UNTERTITEL)
        unter.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in unter.runs:
            r.font.size = Pt(17)
        try:
            unter.style = self.doc.styles["Überschrift Team"]
        except KeyError:
            pass

    def spaltenumbruch(self):
        par = self.doc.add_paragraph()
        par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.line_spacing = Pt(1)
        par.add_run().add_break(WD_BREAK.COLUMN)

    def h(self, text, stil="Heading 2"):
        return self.doc.add_paragraph(text, style=stil)

    def p(self, *teile, stil="Normal", nach=4):
        """Absatz; Teile sind str oder (str, 'b'|'i'|'sub')."""
        par = self.doc.add_paragraph(style=stil)
        for t in teile:
            if isinstance(t, str):
                par.add_run(t)
            else:
                r = par.add_run(t[0])
                r.bold = "b" in t[1]
                r.italic = "i" in t[1]
                if "u" in t[1]:
                    r.font.subscript = True
        par.paragraph_format.space_after = Pt(nach)
        return par

    def punkt(self, *teile):
        par = self.p(*teile, nach=1)
        par.paragraph_format.left_indent = Cm(0.45)
        par.paragraph_format.first_line_indent = Cm(-0.35)
        par.runs[0].text = "• " + par.runs[0].text
        par.alignment = WD_ALIGN_PARAGRAPH.LEFT
        return par

    def bild(self, datei: Path, *unterschrift):
        self.n_abb += 1
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(2)
        par.paragraph_format.keep_with_next = True
        par.add_run().add_picture(str(datei), width=Cm(SPALTE_CM))
        cap = self.doc.add_paragraph(style="Caption")
        cap.add_run(f"Abb. {self.n_abb}: ").bold = True
        for t in unterschrift:
            if isinstance(t, str):
                cap.add_run(t)
            else:
                r = cap.add_run(t[0]); r.font.subscript = True

    def tabelle(self, kopf, zeilen, breiten_cm, *unterschrift):
        cap = self.doc.add_paragraph(style="Caption")
        cap.add_run("Tab. 1: ").bold = True
        cap.add_run("".join(unterschrift))
        cap.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1 + len(zeilen), cols=len(kopf))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        tblPr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tblPr.append(layout)
        w = tblPr.find(qn("w:tblW"))
        if w is None:
            w = OxmlElement("w:tblW"); tblPr.append(w)
        w.set(qn("w:type"), "dxa"); w.set(qn("w:w"), str(int(sum(breiten_cm) / 2.54 * 1440)))
        for gc, b in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), breiten_cm):
            gc.set(qn("w:w"), str(int(b / 2.54 * 1440)))
        for r, werte in enumerate([kopf] + zeilen):
            for c, text in enumerate(werte):
                zelle = t.cell(r, c)
                zelle.width = Cm(breiten_cm[c])
                par = zelle.paragraphs[0]
                par.paragraph_format.space_after = Pt(0)
                par.paragraph_format.space_before = Pt(0)
                if c > 0:
                    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                teile = text.split("_") if c == 0 and r > 0 else [text]
                run = par.add_run(teile[0])
                run.font.size = Pt(8)
                run.bold = r == 0
                if len(teile) > 1:  # Index tiefgestellt: "F_max [kN]"
                    index, _, rest = teile[1].partition(" ")
                    ri = par.add_run(index); ri.font.size = Pt(8); ri.font.subscript = True
                    rr = par.add_run(" " + rest); rr.font.size = Pt(8)
                self._rand(zelle, oben=(r == 0), unten=(r in (0, len(zeilen))))
        abstand = self.doc.add_paragraph()
        abstand.paragraph_format.space_after = Pt(0)
        abstand.paragraph_format.line_spacing = Pt(5)

    @staticmethod
    def _rand(zelle, oben, unten):
        tcPr = zelle._tc.get_or_add_tcPr()
        raender = OxmlElement("w:tcBorders")
        for seite, an in (("top", oben), ("bottom", unten)):
            el = OxmlElement(f"w:{seite}")
            el.set(qn("w:val"), "single" if an else "nil")
            el.set(qn("w:sz"), "8")
            el.set(qn("w:color"), "000000")
            raender.append(el)
        tcPr.append(raender)


def main():
    poster_stil()
    BILDER.mkdir(parents=True, exist_ok=True)
    meta, df, kenn, verl = lade_alles()

    abb1_aufbau(BILDER / "abb1_aufbau.png")
    abb2_beispiel(verl, kenn, meta, BILDER / "abb2_beispielsturz.png")
    abb3_verlaeufe(verl, df, meta, BILDER / "abb3_zeitverlaeufe.png")
    abb4_kennwerte(df, BILDER / "abb4_kennwerte.png")

    # ---- Zahlen für den Text
    F = df.C2_F_peak_kN
    T = df.C2_freifallzeit_s
    h_int, h_max = df.C2_fallhoehe_integriert_m, df.C2_fallhoehe_obergrenze_m
    p_F = kw_p(df, "C2_F_peak_kN")
    d, lo, hi, ga, gb = groesste_differenz(df, "C2_F_peak_kN")
    verh = df.C1_kraftverhaeltnis_flug
    rho = stats.spearmanr(df.nr, df.C2_bremsweg_m)
    p_min = min(kw_p(df, c) for c, _, _ in TABELLE)
    a_g = df.C2_a_peak_g
    h_geo = A.STURZWEG_THEORIE
    t_geo = np.sqrt(2 * h_geo / A.G)

    zeilen = []
    for col, label, n in TABELLE:
        z = [label] + [f"{de(w.mean(), n)} ± {de(w.std(ddof=1), n)}" for w in gruppen(df, col)]
        zeilen.append(z + [de(kw_p(df, col))])
    pd.DataFrame(zeilen, columns=["Kennwert", "GriGri", "Revo", "Reverso", "p"]).to_csv(
        POSTER / "tabelle1.csv", sep=";", index=False)

    # ---- Dokument
    P = Poster()
    P.kopf()

    P.h("Einleitung")
    P.p("Beim Vorstiegssturz nimmt das Sicherungssystem die Fallenergie auf. Die Höhe des Fangstoßes "
        "hängt von Seil, Sturzhöhe, Reibung in den Zwischensicherungen und dem Verhalten der sichernden "
        "Person ab (Pavier, 1998). Hersteller berichten für halbautomatische Geräte höhere Kräfte als "
        "für Tubes, betonen aber die große Streuung (Petzl, o. J.). Inertialsensoren (IMU) erlauben "
        "solche Messungen direkt am Körper (Cudejko et al., 2022).")
    P.p(("Fragestellung: ", "b"), "Unterscheidet sich die maximale Fangstoßkraft auf die kletternde Person "
        "zwischen GriGri, Revo und Tube bei dynamischer Sicherung?", nach=6)

    P.h("Methoden")
    P.p("Eine kletternde Person (C2, 78 kg) stürzte in der Kletterhalle Marswiese (Wien) an einer "
        "senkrechten Wand 15-mal angekündigt ins Seil. Eine erfahrene Person (C1, 70 kg) sicherte "
        "dynamisch aus 1–2 m Wandabstand (Abb. 1). Je fünf Stürze in fester Reihenfolge mit Petzl "
        "GriGri, Wild Country Revo und Petzl Reverso (Tube). Massen ohne Ausrüstung.")
    P.punkt(("Material: ", "b"), "Einfachseil 9 mm (ca. 6 Jahre, Hersteller unbekannt), sechs "
            "Expressschlingen AustriAlpin Micro 11 cm.")
    P.punkt(("Geometrie (geschätzt): ", "b"), f"Hüfte ca. {de(A.STURZHOEHE, 1)} m, letzter Karabiner ca. "
            f"{de(A.SICHERUNGSHOEHE)} m (Wandplatten abgezählt), ca. 0,2 m Schlappseil.")
    P.punkt(("Sensorik: ", "b"), "je ein Xsens DOT mit Gurt am Bauch nahe dem Körperschwerpunkt, "
            "120 Hz, zeitsynchron (Beschleunigung, Euler-Winkel).")
    P.punkt(("Auswertung (Python): ", "b"), "Betrag der Beschleunigung |a|, Butterworth-Tiefpass "
            "4. Ordnung, 15 Hz, phasenfrei. Der Sensor zeigt 1 g in Ruhe und 0 g im freien Fall, "
            "die Kraft von Seil und Wand auf die Person ist daher")
    gl = P.p("F = m · |a|        (1)", nach=2)
    gl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    P.punkt("Vertikalbeschleunigung (über Euler-Winkel ins Weltsystem, g abgezogen) zweifach "
            "integriert, Drift über Ruhephasen korrigiert → vertikaler Weg z (Abb. 2). „Seil greift“: "
            "C2 erreicht wieder 1 g.")
    P.punkt(("Statistik: ", "b"), "Mittelwert ± Standardabweichung (MW ± SD, n = 5 je Gerät), "
            "Kruskal-Wallis-Test (p-Wert, α = 0,05), 95-%-Konfidenzintervall (KI) des größten "
            "Unterschieds, Spearman-Rangkorrelation ρ für den Trend über die Messfolge.")
    P.bild(BILDER / "abb1_aufbau.png",
           "Versuchsaufbau in drei Videobildern eines Sturzes: (a) Ausgangsposition, (b) freier Fall, "
           "(c) Fangen, die sichernde Person wird angehoben. (Eigene Aufnahme)")
    P.bild(BILDER / "abb2_beispielsturz.png",
           f"Beispielsturz (Messung {BEISPIEL_NR}, GriGri). Oben: Betrag der Beschleunigung |a| mit "
           "freiem Fall (blau) und Bremsphase (orange). Unten: vertikaler Weg z, positiv nach oben.")

    P.spaltenumbruch()
    P.h("Ergebnisse")
    P.p(f"Vom Loslassen bis zum ersten Seilzug vergingen {de(T.min())}–{de(T.max())} s. Danach stieg |a| "
        f"von C2 auf {de(a_g.min(), 1)}–{de(a_g.max(), 1)} g (Abb. 3), also "
        f"F", ("max", "u"), f" = {de(F.min())}–{de(F.max())} kN. F", ("max", "u"), " unterschied sich "
        f"zwischen den Geräten nicht signifikant (p = {de(p_F)}; Tab. 1, Abb. 4). Der größte Unterschied "
        f"({ga} − {gb}) betrug {de(d)} kN (95-%-KI {de(lo)} bis {de(hi)} kN). Auch die übrigen Kennwerte "
        f"unterschieden sich nicht signifikant (alle p ≥ {de(np.floor(p_min * 100) / 100)}).")
    P.bild(BILDER / "abb3_zeitverlaeufe.png",
           "Betrag der Beschleunigung |a| der kletternden Person, je Linie ein Sturz (n = 5 je Gerät). "
           "Gestrichelt: „Seil greift“.")
    P.tabelle(["Kennwert", "GriGri", "Revo", "Reverso", "p"], zeilen, [4.3, 2.2, 2.2, 2.2, 1.2],
              "Kennwerte je Gerät (MW ± SD, n = 5) und p-Wert des Kruskal-Wallis-Tests. "
              "Wege als Beträge.")
    P.bild(BILDER / "abb4_kennwerte.png",
           "Einzelstürze (farbig) und MW ± SD (schwarz) je Gerät: maximale Kraft auf die kletternde "
           "Person, gesamter Sturzweg von C2 und Anhebung von C1.")

    P.h("Diskussion")
    P.p(f"Mit rund {de(F.mean(), 1)} kN lag der Fangstoß bei allen Geräten im Bereich eines weichen "
        "Sturzes. Ein Geräteunterschied war nicht nachweisbar, das KI schließt Unterschiede über etwa "
        f"{de(max(abs(lo), abs(hi)), 1)} kN aus. Die Sichernde wurde jedes Mal rund "
        f"{de(df.C1_anhebung_m.mean(), 1)} m angehoben. Diese dynamische Sicherung dürfte den Fangstoß "
        "stärker bestimmen als das Gerät, passend zur beschriebenen Streuung (Petzl, o. J.). Auf die "
        f"Sichernde wirkte im Flug das {de(verh.mean())}-Fache der Kraft auf die Kletternde (F", ("S", "u"),
        ", Tab. 1), erklärbar durch Seilreibung am obersten "
        "Karabiner (Pavier, 1998).")
    P.p(("Limitationen: ", "b"), f"Die geschätzte Geometrie ergibt {de(h_geo, 1)} m freien Fall "
        f"({de(t_geo)} s). Gemessen wurden höchstens {de(T.max())} s, also im Mittel {de(h_int.mean(), 1)} m "
        f"(integriert) bis maximal {de(h_max.mean(), 1)} m (aus der Fallzeit). Die Sturzhöhe war somit "
        "kleiner als geschätzt und nicht konstant. Zudem: ein Team, n = 5, Geräte blockweise getestet "
        f"(Bremsweg nahm über die Messfolge tendenziell zu, ρ = {de(rho.statistic)}, "
        f"p = {de(rho.pvalue)}), Wege nur auf etwa ± 15 % genau.")
    P.p(("Fazit: ", "b"), "Bei dynamischer Sicherung und kurzer Sturzhöhe zeigte sich kein Einfluss des "
        "Sicherungsgeräts auf den Fangstoß. Randomisierte Reihenfolge, gemessene Sturzhöhe und mehr "
        "Stürze sollten dies absichern.", nach=6)

    P.h("Literaturverzeichnis", stil="Heading 3")
    for ref in [
        "Cudejko, T., Button, K., & Al-Amri, M. (2022). Validity and reliability of accelerations and "
        "orientations measured using wearable sensors during functional activities. Scientific Reports, "
        "12, 14619. https://doi.org/10.1038/s41598-022-18845-x",
        "Pavier, M. (1998). Experimental and theoretical simulations of climbing falls. Sports "
        "Engineering, 1(2), 79–91.",
        "Petzl. (o. J.). Influence of the belay device. Tech Tips. Abgerufen am 15. November 2026, von "
        "https://www.petzl.com/US/en/Sport/Influence-of-the-belay-device",
    ]:
        P.doc.add_paragraph(ref, style="References")

    P.doc.save(str(ZIEL))
    print("gespeichert:", ZIEL)
    print(pd.DataFrame(zeilen).to_string(index=False, header=False))
    print(f"groesste Differenz F_max: {ga}-{gb} {d:.2f} [{lo:.2f}, {hi:.2f}] kN | p_min {p_min:.3f}")


if __name__ == "__main__":
    main()
