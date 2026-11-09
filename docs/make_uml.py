"""Generate AIDE_UML_Diagrams.pdf: component, use case, class (two views)
and sequence diagrams for the AIDE project, drawn with reportlab.

    python docs/make_uml.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent / "AIDE_UML_Diagrams.pdf"
W, H = landscape(A3)

INK = colors.HexColor("#0f172a")
SLATE = colors.HexColor("#475569")
MUTED = colors.HexColor("#94a3b8")
LINE = colors.HexColor("#cbd5e1")
INDIGO = colors.HexColor("#4f46e5")
HEAD = colors.HexColor("#e0e7ff")
FILL = colors.white
MOD_HEAD = colors.HexColor("#fef3c7")
RES_HEAD = colors.HexColor("#dcfce7")
PKG = colors.HexColor("#f8fafc")

FS = 8.6          # member font size
LH = 11.2         # member line height


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
class Box:
    """A UML classifier box placed by its top-left corner."""

    def __init__(self, x, top, w, title, attrs=(), methods=(), stereo=None, head=HEAD,
                 compartments=True):
        self.x, self.top, self.w = x, top, w
        self.title, self.attrs, self.methods, self.stereo = title, attrs, methods, stereo
        self.head, self.compartments = head, compartments
        self.head_h = 30 if stereo else 20
        self.h = self.head_h
        if compartments:
            self.h += max(len(attrs), 1) * LH + 6
            if methods is not None:
                self.h += max(len(methods), 1) * LH + 6

    @property
    def y(self):
        return self.top - self.h

    def anchor(self, side, f=0.5):
        if side == "l":
            return self.x, self.top - self.h * f
        if side == "r":
            return self.x + self.w, self.top - self.h * f
        if side == "t":
            return self.x + self.w * f, self.top
        return self.x + self.w * f, self.y

    def draw(self, c):
        c.setStrokeColor(INK)
        c.setLineWidth(0.9)
        c.setFillColor(FILL)
        c.rect(self.x, self.y, self.w, self.h, stroke=1, fill=1)
        c.setFillColor(self.head)
        c.rect(self.x, self.top - self.head_h, self.w, self.head_h, stroke=1, fill=1)
        c.setFillColor(INK)
        ty = self.top - 14
        if self.stereo:
            c.setFont("Helvetica-Oblique", 7.8)
            c.drawCentredString(self.x + self.w / 2, ty, f"«{self.stereo}»")
            ty -= 11
        c.setFont("Helvetica-Bold", 9.6)
        c.drawCentredString(self.x + self.w / 2, ty, self.title)
        if not self.compartments:
            return
        y = self.top - self.head_h - LH + 1
        c.setFont("Helvetica", FS)
        for a in self.attrs:
            c.drawString(self.x + 6, y, a)
            y -= LH
        if not self.attrs:
            y -= LH
        y -= 6 - LH + LH
        if self.methods is not None:
            c.setStrokeColor(INK)
            c.line(self.x, y + LH - 2, self.x + self.w, y + LH - 2)
            y -= 3
            for m in self.methods:
                c.drawString(self.x + 6, y, m)
                y -= LH


def _route(pa, aside, pb, bside, via=None):
    if via is not None:
        return [pa, *via, pb]
    horiz = {"l", "r"}
    if aside in horiz and bside in horiz:
        mx = (pa[0] + pb[0]) / 2
        return [pa, (mx, pa[1]), (mx, pb[1]), pb]
    if aside not in horiz and bside not in horiz:
        my = (pa[1] + pb[1]) / 2
        return [pa, (pa[0], my), (pb[0], my), pb]
    if aside in horiz:
        return [pa, (pb[0], pa[1]), pb]
    return [pa, (pa[0], pb[1]), pb]


def _unit(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1
    return dx / n, dy / n


def _arrow_open(c, tip, frm, size=8):
    ux, uy = _unit(frm, tip)
    bx, by = tip[0] - ux * size, tip[1] - uy * size
    px, py = -uy * size * 0.5, ux * size * 0.5
    c.line(tip[0], tip[1], bx + px, by + py)
    c.line(tip[0], tip[1], bx - px, by - py)


def _triangle(c, tip, frm, size=11, fill=FILL):
    ux, uy = _unit(frm, tip)
    bx, by = tip[0] - ux * size, tip[1] - uy * size
    px, py = -uy * size * 0.55, ux * size * 0.55
    p = c.beginPath()
    p.moveTo(*tip)
    p.lineTo(bx + px, by + py)
    p.lineTo(bx - px, by - py)
    p.close()
    c.setFillColor(fill)
    c.drawPath(p, stroke=1, fill=1)


def _diamond(c, tip, toward, filled, size=12):
    ux, uy = _unit(tip, toward)
    mx, my = tip[0] + ux * size / 2, tip[1] + uy * size / 2
    ex, ey = tip[0] + ux * size, tip[1] + uy * size
    px, py = -uy * size * 0.32, ux * size * 0.32
    p = c.beginPath()
    p.moveTo(*tip)
    p.lineTo(mx + px, my + py)
    p.lineTo(ex, ey)
    p.lineTo(mx - px, my - py)
    p.close()
    c.setFillColor(INK if filled else FILL)
    c.drawPath(p, stroke=1, fill=1)
    return ex, ey


def link(c, a, aside, b, bside, kind="assoc", label=None, ma=None, mb=None, fa=0.5, fb=0.5,
         via=None, label_at=None):
    """kind: assoc, nav (open arrow at b), dep (dashed, open arrow at b),
    comp/agg (diamond at a, open arrow at b), inherit (hollow triangle at b)."""
    pa, pb = a.anchor(aside, fa), b.anchor(bside, fb)
    pts = _route(pa, aside, pb, bside, via)
    c.setStrokeColor(INK)
    c.setLineWidth(0.8)
    c.setDash(4, 3) if kind == "dep" else c.setDash()
    start = pts[0]
    if kind in ("comp", "agg"):
        start = _diamond(c, pts[0], pts[1], kind == "comp")
        c.setStrokeColor(INK)
    path = c.beginPath()
    path.moveTo(*start)
    for p in pts[1:]:
        path.lineTo(*p)
    c.drawPath(path, stroke=1, fill=0)
    c.setDash()
    if kind in ("nav", "dep", "comp", "agg"):
        _arrow_open(c, pts[-1], pts[-2])
    elif kind == "inherit":
        _triangle(c, pts[-1], pts[-2])
    c.setFont("Helvetica", 7.6)
    c.setFillColor(SLATE)
    if label:
        lx, ly = label_at or ((pts[len(pts) // 2 - 1][0] + pts[len(pts) // 2][0]) / 2,
                              (pts[len(pts) // 2 - 1][1] + pts[len(pts) // 2][1]) / 2)
        c.drawCentredString(lx, ly + 3, label)
    for m, p, q in ((ma, pts[0], pts[1]), (mb, pts[-1], pts[-2])):
        if m:
            ux, uy = _unit(p, q)
            c.drawString(p[0] + ux * 16 + (4 if abs(uy) > 0.5 else 0),
                         p[1] + uy * 16 + (3 if abs(ux) > 0.5 else 0), m)


def page_header(c, title, subtitle, n):
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(INDIGO)
    c.rect(0, H - 5, W, 5, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(30, H - 38, title)
    c.setFont("Helvetica", 10.5)
    c.setFillColor(SLATE)
    c.drawString(30, H - 56, subtitle)
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED)
    c.drawString(30, 16, "AIDE - Adaptive Intelligent Dynamic Embedding  |  UML diagrams  |  v0.2")
    c.drawRightString(W - 30, 16, f"{n} / 5")


def legend(c, x, y, items):
    c.setFont("Helvetica", 8)
    for i, (sw, text) in enumerate(items):
        c.setFillColor(sw)
        c.setStrokeColor(INK)
        c.rect(x, y - i * 14, 14, 9, stroke=1, fill=1)
        c.setFillColor(SLATE)
        c.drawString(x + 20, y - i * 14 + 1, text)


def package(c, x, y, w, h, name):
    c.setStrokeColor(SLATE)
    c.setLineWidth(0.9)
    c.setFillColor(PKG)
    tw = c.stringWidth(name, "Helvetica-Bold", 9.5) + 18
    c.rect(x, y + h, tw, 16, stroke=1, fill=1)
    c.rect(x, y, w, h, stroke=1, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(x + 9, y + h + 4.5, name)


def component(c, x, y, w, h, name, sub=None):
    c.setStrokeColor(INK)
    c.setLineWidth(0.9)
    c.setFillColor(FILL)
    c.rect(x, y, w, h, stroke=1, fill=1)
    # UML component icon
    ix, iy = x + w - 20, y + h - 17
    c.rect(ix, iy, 13, 11, stroke=1, fill=1)
    c.rect(ix - 3, iy + 6.5, 6, 3, stroke=1, fill=1)
    c.rect(ix - 3, iy + 1.5, 6, 3, stroke=1, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 9.2)
    c.drawString(x + 8, y + h - 16, name)
    if sub:
        c.setFont("Helvetica", 7.8)
        c.setFillColor(SLATE)
        for i, line in enumerate(sub.split("\n")):
            c.drawString(x + 8, y + h - 29 - i * 10, line)
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h),
            "b": (x + w / 2, y), "x": x, "y": y, "w": w, "h": h}


def dep_line(c, pts, label=None, dashed=True):
    c.setStrokeColor(INK)
    c.setLineWidth(0.8)
    c.setDash(4, 3) if dashed else c.setDash()
    p = c.beginPath()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    c.drawPath(p, stroke=1, fill=0)
    c.setDash()
    _arrow_open(c, pts[-1], pts[-2])
    if label:
        c.setFont("Helvetica-Oblique", 7.4)
        c.setFillColor(SLATE)
        mx = (pts[-2][0] + pts[-1][0]) / 2
        my = (pts[-2][1] + pts[-1][1]) / 2
        c.drawString(mx + 4, my, label)


# --------------------------------------------------------------------------
# 1. Component / package diagram
# --------------------------------------------------------------------------
def component_diagram(c):
    page_header(c, "1. Component diagram",
                "Packages and components of the AIDE system and how they depend on each other "
                "(dashed arrow = «use» dependency).", 1)

    # Interface layer
    package(c, 30, 640, 1131, 112, "Interfaces")
    web = component(c, 60, 655, 300, 74, "Web console", "webapp/app.py (Flask, gunicorn)\n"
                    "Operator | Analyst | Model Insights tabs\nvisualize.py overlays and heatmaps")
    api = component(c, 440, 662, 300, 62, "REST API", "/api/v1/* + OpenAPI spec, /healthz\n"
                    "API-key auth, request IDs, JSON errors")
    cli = component(c, 820, 662, 310, 62, "Command line (aide)", "aide/cli.py: embed, extract, "
                    "capacity, attack,\nevaluate, detect, benchmark, train, profiles")

    # Application services
    package(c, 30, 480, 1131, 130, "Application services")
    pipe = component(c, 50, 500, 200, 72, "Sidecar pipeline", "pipeline.py\nembed() / recover()\n"
                     "capacity_report()")
    bl = component(c, 270, 500, 200, 72, "Blind pipeline", "blind.py\nkeyed layout,\n"
                   "self-describing header")
    bench = component(c, 490, 500, 200, 72, "Benchmark", "benchmark.py\nparallel robustness +\n"
                      "inference speed")
    tr = component(c, 710, 500, 200, 72, "Engine trainer", "train.py\nmeasured method/bpp labels\n"
                   "writes data/models/")
    det = component(c, 930, 500, 210, 72, "Detection service", "cli.detect_image()\nclassical + "
                    "quantitative + ML\none verdict per image")

    # Core
    package(c, 30, 205, 700, 235, "Core: hide, protect, verify")
    eng = component(c, 50, 345, 210, 72, "Decision engine", "features.py (LBP+GLCM, entropy,\n"
                    "variance, edge) | scoring.py S(b)\nmodel.py AIDEModel (tree / MLP)")
    component(c, 280, 345, 200, 72, "Embedding methods", "embedding/lsb.py\n"
              "embedding/dct.py (pair QIM)\nembedding/dwt.py (Haar cA2)")
    component(c, 500, 345, 210, 72, "Robustness profiles", "profiles.py\nquality | "
              "balanced |\nrobust | fortress")
    component(c, 50, 225, 210, 90, "Cryptography", "crypto.py\nAES-256-GCM\nscrypt / "
              "SHA-256 KDF\nimage-bound key")
    component(c, 280, 225, 200, 90, "Error correction", "ecc.py\nReed-Solomon, "
              "shortened\ncodewords, interleaving,\nGMD soft decoding")
    component(c, 500, 225, 210, 90, "Tamper detection", "tamper.py\nper-block SHA-256 "
              "+\nhistogram chi-square\n-> erasure mask")

    # Analysis
    package(c, 760, 205, 401, 235, "Analysis and evaluation")
    component(c, 780, 345, 170, 72, "Classical steganalysis", "steganalysis.py\n"
              "chi-square PoV, LSB signals\nsuspicion heatmap")
    component(c, 970, 345, 170, 72, "Quantitative", "quantitative.py\nSPA + RS analysis\n"
              "payload-size estimate")
    ml = component(c, 780, 225, 170, 90, "ML steganalyser", "mlsteg.py\nSPAM + DCT pair-gap\n"
                   "gradient-boosted trees\nOOD guard")
    component(c, 970, 280, 170, 35, "Attacks", "attacks.py: 8 simulations")
    component(c, 970, 225, 170, 45, "Metrics", "metrics.py\nPSNR, SSIM, BER, timing")

    # Data / external
    package(c, 30, 50, 560, 120, "Model artifacts (data/models)")
    art = component(c, 50, 70, 250, 72, "Decision engine model", "weights.json (alpha..delta, "
                    "thresholds)\naide_model.joblib (classifier + bpp)\n13 KB")
    art2 = component(c, 320, 70, 250, 72, "Steganalyser model", "steganalyzer.joblib\n"
                     "steganalyzer_report.json\n(AUC, per-variant detection)")
    package(c, 620, 50, 541, 120, "External libraries")
    component(c, 640, 70, 500, 72, "Third-party", "NumPy, SciPy, OpenCV, scikit-image, "
              "PyWavelets, Pillow\nscikit-learn, joblib  |  pycryptodome  |  reedsolo\n"
              "Flask, gunicorn  |  psutil")

    # Dependencies
    dep_line(c, [web["b"], (web["b"][0], 600), (pipe["t"][0] + 30, 600),
                 (pipe["t"][0] + 30, pipe["t"][1])])
    dep_line(c, [(web["b"][0] + 60, web["b"][1]), (web["b"][0] + 60, 605), (bl["t"][0], 605),
                 bl["t"]])
    dep_line(c, [api["l"], web["r"]], "served by")
    dep_line(c, [(cli["b"][0] - 120, cli["b"][1]), (cli["b"][0] - 120, 600), (bench["t"][0], 600),
                 bench["t"]])
    dep_line(c, [(cli["b"][0] - 60, cli["b"][1]), (cli["b"][0] - 60, 595), (tr["t"][0], 595),
                 tr["t"]])
    dep_line(c, [(cli["b"][0] + 40, cli["b"][1]), (cli["b"][0] + 40, det["t"][1])])
    dep_line(c, [(220, 500), (220, 440)], "\u00abuse\u00bb core")
    dep_line(c, [(380, 500), (380, 440)], "\u00abuse\u00bb core")
    dep_line(c, [(640, 500), (640, 470), (930, 470), (930, 440)], "\u00abuse\u00bb attacks, metrics")
    dep_line(c, [(1060, 500), (1060, 440)], "\u00abuse\u00bb")
    dep_line(c, [(810, 500), (810, 460), (745, 460), (745, 160), (230, 160), (230, 142)],
             "writes")
    dep_line(c, [eng["l"], (40, eng["l"][1]), (40, art["t"][1] + 10),
                 (art["t"][0] - 40, art["t"][1] + 10), (art["t"][0] - 40, art["t"][1])], "loads")
    dep_line(c, [ml["l"], (765, ml["l"][1]), (765, 185), (art2["t"][0], 185), art2["t"]], "loads")

    legend(c, 1000, 820, [(PKG, "package"), (FILL, "component")])


# --------------------------------------------------------------------------
# 2. Use case diagram
# --------------------------------------------------------------------------
def actor(c, x, y, name):
    c.setStrokeColor(INK)
    c.setLineWidth(1.1)
    c.setFillColor(FILL)
    c.circle(x, y + 34, 8, stroke=1, fill=1)
    c.line(x, y + 26, x, y + 8)
    c.line(x - 12, y + 20, x + 12, y + 20)
    c.line(x, y + 8, x - 9, y - 6)
    c.line(x, y + 8, x + 9, y - 6)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 9.5)
    for i, line in enumerate(name.split("\n")):
        c.drawCentredString(x, y - 18 - i * 11, line)
    return x, y + 18


def usecase(c, x, y, text, w=170, h=34):
    c.setStrokeColor(INK)
    c.setLineWidth(0.9)
    c.setFillColor(HEAD)
    c.ellipse(x - w / 2, y - h / 2, x + w / 2, y + h / 2, stroke=1, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica", 8.6)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        c.drawCentredString(x, y - 3 + (len(lines) - 1) * 5 - i * 10, line)
    return {"c": (x, y), "l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2),
            "b": (x, y - h / 2)}


def plain(c, p, q):
    c.setStrokeColor(INK)
    c.setLineWidth(0.8)
    c.setDash()
    c.line(p[0], p[1], q[0], q[1])


def usecase_diagram(c):
    page_header(c, "2. Use case diagram",
                "Who interacts with AIDE and what they can do. «include» = always "
                "performed as part of; «extend» = optional extra step.", 2)
    c.setStrokeColor(INK)
    c.setLineWidth(1.2)
    c.setFillColor(colors.HexColor("#fbfcfe"))
    c.rect(210, 50, 770, 715, stroke=1, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(225, 745, "AIDE system")

    op = actor(c, 100, 610, "Operator\n(sender)")
    rcv = actor(c, 100, 400, "Receiver")
    an = actor(c, 1080, 590, "Intelligence /\nforensic analyst")
    rs = actor(c, 1080, 230, "Researcher /\nstudent")

    u_embed = usecase(c, 380, 680, "Embed secret message")
    u_prof = usecase(c, 380, 610, "Choose profile and mode\n(sidecar / blind)")
    u_cap = usecase(c, 380, 545, "Check capacity")
    u_enc = usecase(c, 600, 700, "Encrypt (AES-256-GCM)\n+ add RS parity", w=180)
    u_sel = usecase(c, 600, 640, "AI selects blocks,\nmethod and bpp", w=180)
    u_att = usecase(c, 380, 470, "Simulate channel attack")
    u_rec = usecase(c, 380, 380, "Recover message")
    u_tam = usecase(c, 380, 305, "Inspect tamper /\nerasure map")
    u_dec = usecase(c, 600, 380, "Repair with ECC and\nverify GCM tag", w=180)

    u_det = usecase(c, 640, 585, "Detect hidden data\nin an image")
    u_batch = usecase(c, 850, 650, "Batch-screen a folder\n(CSV / ranked table)")
    u_est = usecase(c, 850, 560, "Estimate payload size\n(SPA / RS)")
    u_heat = usecase(c, 850, 480, "View suspicion heatmap")
    u_ml = usecase(c, 640, 500, "Score with ML\nsteganalyser", w=170)

    u_bench = usecase(c, 820, 300, "Run robustness benchmark")
    u_train = usecase(c, 820, 235, "Train decision engine")
    u_tdet = usecase(c, 820, 170, "Train steganalyser")
    u_ins = usecase(c, 820, 105, "View Model Insights")
    u_vis = usecase(c, 380, 225, "View embedding map /\ndifference heatmap", w=180)

    for u in (u_embed, u_prof, u_cap, u_att, u_tam, u_vis):
        plain(c, (op[0] + 14, op[1]), u["l"])
    plain(c, (rcv[0] + 14, rcv[1]), u_rec["l"])
    plain(c, (rcv[0] + 14, rcv[1]), u_tam["l"])
    for u in (u_det, u_batch, u_est, u_heat):
        plain(c, (an[0] - 14, an[1]), u["r"])
    for u in (u_bench, u_train, u_tdet, u_ins):
        plain(c, (rs[0] - 14, rs[1]), u["r"])

    def rel(a, b, text):
        dep_line(c, [a, b])
        c.setFont("Helvetica-Oblique", 7.4)
        c.setFillColor(SLATE)
        c.drawCentredString((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 4, text)

    rel(u_embed["r"], u_enc["l"], "«include»")
    rel((u_embed["r"][0] - 10, u_embed["r"][1] - 10), u_sel["l"], "«include»")
    rel(u_rec["r"], u_dec["l"], "«include»")
    rel(u_tam["t"], u_rec["b"], "«extend»")
    rel(u_prof["t"], u_embed["b"], "«extend»")
    rel(u_det["b"], u_ml["t"], "\u00abinclude\u00bb")
    rel(u_batch["l"], (u_det["r"][0] - 8, u_det["r"][1] + 8), "\u00abinclude\u00bb")
    rel(u_est["l"], u_det["r"], "\u00abextend\u00bb")
    rel(u_heat["l"], (u_det["r"][0] - 10, u_det["r"][1] - 9), "\u00abextend\u00bb")


# --------------------------------------------------------------------------
# 3. Class diagram: embedding / recovery domain
# --------------------------------------------------------------------------
def class_diagram_operator(c):
    page_header(c, "3. Class diagram: hiding and recovering a message",
                "Main data types of the sidecar and blind pipelines (Python dataclasses). "
                "Module-level functions are shown as «module» classifiers.", 3)

    # column 1: entry-point modules and blind results
    pipe = Box(30, 760, 240, "pipeline", ["DEFAULT_FRAGILE_METHOD_BUDGETS"], [
        "+embed(cover, message, secret, model,",
        "      profile): EmbedResult",
        "+recover(stego, sidecar, secret): RecoverResult",
        "+capacity_report(cover, model): dict",
        "+max_message_bytes(bits, profile): int",
        "+load_image(path): LoadedImage",
        "+save_sidecar() / load_sidecar()",
    ], stereo="module", head=MOD_HEAD)
    blind = Box(30, 585, 240, "blind", ["HEADER_BITS = 960", "MAGIC, VERSION"], [
        "+embed(cover, message, secret, model,",
        "      profile): BlindEmbedResult",
        "+recover(stego, secret): BlindRecoverResult",
        "+capacity_bits(h, w): int",
        "+max_message_bytes(h, w, profile): int",
    ], stereo="module", head=MOD_HEAD)
    loaded = Box(30, 440, 240, "LoadedImage", [
        "+y: ndarray  (luma)", "+mode: str", "+chroma: (ndarray, ndarray) | None",
        "+raw_bytes: bytes"], [], stereo="dataclass")
    bres = Box(30, 318, 240, "BlindEmbedResult", [
        "+stego: ndarray", "+profile: RobustnessProfile", "+blocks_used: int",
        "+header_bits / payload_bits: int", "+capacity_bits: int"], ["+compose(): Image"],
        stereo="dataclass", head=RES_HEAD)
    brres = Box(30, 180, 240, "BlindRecoverResult", [
        "+success, plaintext, error, profile", "+corrected_bytes, gmd_codewords: int"], None,
        stereo="dataclass", head=RES_HEAD)

    # column 2: sidecar-mode results
    eres = Box(320, 760, 220, "EmbedResult", [
        "+stego: ndarray", "+sidecar: Sidecar", "+mode: str",
        "+chroma: (ndarray, ndarray) | None"], ["+compose(): Image"], stereo="dataclass",
        head=RES_HEAD)
    side = Box(320, 630, 220, "Sidecar", [
        "+width, height: int", "+block_size: int", "+nonce: bytes", "+image_hash: bytes",
        "+ciphertext_len, tag_len: int", "+embedding_map: list", "+references: list",
        "+profile: RobustnessProfile", "+ecc_params: ECCParams", "+kdf: str"],
        ["+to_json(): str", "+from_json(text): Sidecar"], stereo="dataclass")
    rres = Box(320, 420, 220, "RecoverResult", [
        "+success: bool", "+plaintext: bytes | None", "+error: str | None",
        "+tamper_report: TamperReport | None", "+erasure_count: int", "+total_bytes: int",
        "+corrected_bytes: int", "+gmd_codewords: int"], [], stereo="dataclass",
        head=RES_HEAD)

    # column 3: what the sidecar is made of
    emap = Box(610, 760, 200, "EmbeddingMapEntry", [
        "+block_index: int", "+method: str  (lsb|dct|dwt)", "+bpp: int",
        "+bit_start, bit_end: int"], [], stereo="dataclass")
    bref = Box(610, 650, 200, "BlockReference", [
        "+block_index: int", "+sha256: str", "+histogram: list[float]"], [],
        stereo="dataclass")
    prof = Box(610, 550, 200, "RobustnessProfile", [
        "+name: str", "+description: str", "+dct_delta: float", "+dwt_delta: float",
        "+nsym: int  (RS parity)", "+fragile_budgets: dict", "+min_codewords: int",
        "+max_bpp: int"], ["+delta_for(method): float", "+to_dict() / from_dict()"],
        stereo="dataclass (frozen)")
    eccp = Box(610, 380, 200, "ECCParams", ["+nsym: int", "+data_len: int"], [
        "+codeword_len: int", "+t: int  (correctable)", "+to_dict() / from_dict()"],
        stereo="dataclass (frozen)")
    dstats = Box(610, 265, 200, "DecodeStats", [
        "+codewords: int", "+corrected_bytes: int", "+gmd_codewords: int",
        "+failed_codewords: int"], [], stereo="dataclass")

    # column 4: protect / verify modules
    trep = Box(870, 760, 200, "TamperReport", [
        "+hash_mismatch: ndarray[bool]", "+chi_square: ndarray", "+confidence: ndarray",
        "+erasure_mask: ndarray[bool]"], [], stereo="dataclass")
    tamper = Box(870, 640, 200, "tamper", [], [
        "+build_references(gray, blocks)", "+analyze(gray, blocks, refs): TamperReport",
        "+block_hash(block): str"], stereo="module", head=MOD_HEAD)
    ecc = Box(870, 500, 200, "ecc", ["NSYM, CODEWORD_LEN = 255"], [
        "+plan(len, nsym): ECCParams", "+encode(data, params): bytes",
        "+interleave() / deinterleave()", "+decode_with_reliability(enc,",
        "     reliability, erasures)"], stereo="module", head=MOD_HEAD)
    crypto = Box(870, 330, 200, "crypto", ["KDF_SCRYPT, KDF_SHA256"], [
        "+derive_key(hash, secret, kdf)", "+encrypt(pt, hash, secret)",
        "+decrypt(payload, ...)"], stereo="module", head=MOD_HEAD)

    # column 5
    emb = Box(1090, 760, 80, "embedding", ["lsb", "dct", "dwt"], [
        "embed()", "extract()", "extract_soft()"], stereo="package", head=MOD_HEAD)
    exc = Box(1090, 600, 80, "Errors", ["AideCapacity-", " Error", "AideECCError",
                                        "AideAuthen-", " ticationError", "AideBlind-",
                                        " DecodeError"], None, stereo="exception")
    enc = Box(1090, 330, 80, "EncPayload", ["+ciphertext", "+nonce", "+tag"], [],
              stereo="dataclass")

    for b in (pipe, blind, loaded, bres, brres, eres, side, rres, emap, bref, prof, eccp,
              dstats, trep, tamper, ecc, crypto, emb, exc, enc):
        b.draw(c)

    def mid(b, f):
        return b.top - b.h * f

    # pipeline / blind create their results
    link(c, pipe, "r", eres, "l", "dep", "«create»", fa=0.3, fb=0.5)
    link(c, pipe, "b", rres, "l", "dep", None, fa=0.75, fb=0.12,
         via=[(210, 600), (296, 600), (296, mid(rres, 0.12))])
    c.setFont("Helvetica-Oblique", 7.4)
    c.setFillColor(SLATE)
    c.drawString(300, mid(rres, 0.12) + 4, "«create»")
    link(c, pipe, "l", loaded, "l", "dep", None, fa=0.6, fb=0.3,
         via=[(18, mid(pipe, 0.6)), (18, mid(loaded, 0.3))])
    link(c, blind, "r", bres, "r", "dep", None, fa=0.55, fb=0.3,
         via=[(284, mid(blind, 0.55)), (284, mid(bres, 0.3))])
    c.drawString(288, mid(bres, 0.3) + 40, "«create»")
    link(c, bres, "b", brres, "t", "dep")
    link(c, pipe, "t", emb, "t", "dep", "«use» embedding methods (lsb / dct / dwt)",
         via=[(150, 771), (1130, 771)], label_at=(640, 770))

    # Sidecar structure
    link(c, eres, "b", side, "t", "comp", None, mb="1")
    link(c, side, "r", emap, "l", "comp", None, fa=0.05, fb=0.5, mb="1..*",
         via=[(565, mid(side, 0.05)), (565, mid(emap, 0.5))])
    link(c, side, "r", bref, "l", "comp", None, fa=0.16, fb=0.5, mb="1..*",
         via=[(575, mid(side, 0.16)), (575, mid(bref, 0.5))])
    link(c, side, "r", prof, "l", "agg", None, fa=0.45, fb=0.35, mb="1",
         via=[(585, mid(side, 0.45)), (585, mid(prof, 0.35))])
    link(c, side, "r", eccp, "l", "comp", None, fa=0.9, fb=0.4, mb="1",
         via=[(595, mid(side, 0.9)), (595, mid(eccp, 0.4))])
    link(c, rres, "r", trep, "b", "agg", None, fa=0.92, fb=0.15, mb="0..1",
         via=[(600, mid(rres, 0.92)), (600, 150), (840, 150), (840, trep.y - 14),
              (trep.x + trep.w * 0.15, trep.y - 14)])

    # protect / verify modules
    link(c, tamper, "t", trep, "b", "dep", "«create»", fa=0.6, fb=0.6)
    link(c, tamper, "l", bref, "r", "dep", "«create»", fa=0.4, fb=0.5,
         via=[(850, mid(tamper, 0.4)), (850, mid(bref, 0.5))])
    link(c, ecc, "l", eccp, "r", "dep", "«create»", fa=0.3, fb=0.5,
         via=[(855, mid(ecc, 0.3)), (855, mid(eccp, 0.5))])
    link(c, ecc, "l", dstats, "r", "dep", None, fa=0.85, fb=0.4,
         via=[(862, mid(ecc, 0.85)), (862, mid(dstats, 0.4))])
    link(c, crypto, "r", enc, "l", "dep", None, fa=0.6, fb=0.6)

    legend(c, 640, 120, [(MOD_HEAD, "«module» functions"), (HEAD, "data class"),
                         (RES_HEAD, "result returned to caller")])
    c.setFont("Helvetica", 7.8)
    c.setFillColor(SLATE)
    c.drawString(800, 120, "◆ composition    ◇ aggregation    - - >  dependency")
    c.drawString(800, 106, "Multiplicity shown at the target end (1, 0..1, 1..*).")


# --------------------------------------------------------------------------
# 4. Class diagram: decision engine + analyst side
# --------------------------------------------------------------------------
def class_diagram_engine(c):
    page_header(c, "4. Class diagram: decision engine and steganalysis",
                "The trained models: AIDEModel picks blocks/method/bpp; Steganalyzer and the "
                "statistical modules analyse suspect images.", 4)

    model = Box(40, 760, 270, "AIDEModel", [
        "+weights: ScoringWeights", "+thresholds: AIDEThresholds",
        "+method_clf: DecisionTree | MLP", "+bpp_reg: regressor", "+meta: dict"], [
        "+score(features): ndarray", "+decide(features): list[BlockDecision]",
        "+save(directory)", "+load(directory): AIDEModel  «class»",
        "+default(): AIDEModel  «class»"])
    weights = Box(40, 520, 200, "ScoringWeights", [
        "+alpha: float  (texture) = 0.55", "+beta: float  (entropy) = 0.006",
        "+gamma: float  (variance) = 0.0003", "+delta: float  (edge) = 0.44"],
        ["+as_array(): ndarray", "+as_dict(): dict"], stereo="dataclass (frozen)")
    thr = Box(260, 520, 170, "AIDEThresholds", [
        "+tau_min: float  (keep)", "+t_low: float  (-> LSB)", "+e_high: float  (-> DWT)"], [],
        stereo="dataclass (frozen)")
    dec = Box(40, 330, 200, "BlockDecision", [
        "+block_index: int", "+score: float", "+keep: bool", "+method: str | None",
        "+bpp: int | None", "+rank: int | None"], [], stereo="dataclass", head=RES_HEAD)
    block = Box(260, 330, 170, "Block", [
        "+index, row, col: int", "+size: int", "+y0, x0, y1, x1: int"], ["+slice()"],
        stereo="dataclass (frozen)")
    feats = Box(40, 150, 390, "features / scoring", [], [
        "+partition_blocks(h, w, size): list[Block]",
        "+extract_image_features(gray): (blocks, ndarray[n x 4])",
        "+compute_scores(features, weights): ndarray   S(b) = aT + bE + gV + dD",
        "+select_method_fallback(texture, edge, thresholds)"], stereo="module",
        head=MOD_HEAD)
    trn = Box(480, 760, 230, "train", ["DEFAULT_TARGET_PSNR = 42 dB"], [
        "+build_training_set(images, psnr)", "+train(image_dir, out_dir,",
        "      classifier='tree'|'mlp'): AIDEModel"], stereo="module", head=MOD_HEAD)

    stz = Box(780, 760, 250, "Steganalyzer", [
        "+clf: GradientBoostingClassifier", "+threshold: float", "+meta: dict",
        "+ood: outlier model", "+ood_threshold: float | None"], [
        "+predict_proba(gray): float", "+analyze(gray): DetectorReport", "+save(directory)",
        "+load(directory): Steganalyzer  «class»"])
    drep = Box(1060, 760, 110, "DetectorReport", [
        "+probability", "+verdict", "+in_distribution", "+note"], [], stereo="dataclass",
        head=RES_HEAD)
    mls = Box(780, 545, 250, "mlsteg", ["126 feature dims"], [
        "+spam_features(gray): 98 dims", "+dct_gap_features(gray): 28 dims",
        "+extract_features(gray): ndarray",
        "+train_detector(images, out_dir): Steganalyzer"], stereo="module", head=MOD_HEAD)
    sa = Box(780, 375, 250, "steganalysis", [], [
        "+chi_square_pov(gray): dict", "+lsb_correlation() / lsb_bit_balance()",
        "+spatial_suspicion_grid(gray): ndarray",
        "+analyze(gray): SteganalysisReport"], stereo="module", head=MOD_HEAD)
    srep = Box(780, 225, 250, "SteganalysisReport", [
        "+chi_square_p: float", "+lsb_correlation, lsb_bit_balance: float",
        "+spatial_grid: list[list[float]]", "+suspicion_score: float", "+verdict: str",
        "+spa_rate, rs_rate: float", "+estimated_payload_bytes: int"], [],
        stereo="dataclass", head=RES_HEAD)
    qa = Box(1060, 375, 110, "quantitative", [], ["+spa(gray)", "+rs_analysis(gray)"],
             stereo="module", head=MOD_HEAD)
    det = Box(480, 375, 230, "cli.detect_image", [], [
        "+detect_image(path, detector): dict", "combines classical + SPA/RS",
        "+ ML probability into one verdict"], stereo="function", head=MOD_HEAD)
    pipe = Box(480, 560, 230, "pipeline / blind", [], [
        "+embed(..., model: AIDEModel, ...)", "uses decide() to choose where,",
        "how much and which method"], stereo="module", head=MOD_HEAD)

    for b in (model, weights, thr, dec, block, feats, trn, stz, drep, mls, sa, srep, qa, det,
              pipe):
        b.draw(c)

    link(c, model, "b", weights, "t", "comp", None, fa=0.3, fb=0.5, mb="1")
    link(c, model, "b", thr, "t", "comp", None, fa=0.8, fb=0.5, mb="1")
    link(c, weights, "b", dec, "t", "dep", None, fa=0.5, fb=0.5)
    c.setFont("Helvetica-Oblique", 7.4)
    c.setFillColor(SLATE)
    c.drawString(146, 345, "decide() «create»")
    link(c, dec, "r", block, "l", "nav", None, fa=0.25, fb=0.3, mb="1")
    link(c, feats, "t", block, "b", "dep", "«create»", fa=0.75)
    link(c, trn, "l", model, "r", "dep", "«create»", fa=0.5, fb=0.25)
    link(c, pipe, "l", model, "r", "dep", "«use»", fa=0.3, fb=0.75)
    link(c, stz, "r", drep, "l", "dep", None, fa=0.25, fb=0.35)
    link(c, mls, "t", stz, "b", "dep", "«create»")
    link(c, sa, "b", srep, "t", "dep", "«create»")
    link(c, sa, "r", qa, "l", "dep", "«use»", fa=0.5, fb=0.5)
    link(c, det, "r", stz, "l", "dep", "«use»", fa=0.25, fb=0.7,
         via=[(745, det.top - det.h * 0.25), (745, stz.top - stz.h * 0.7)])
    link(c, det, "r", sa, "l", "dep", None, fa=0.7, fb=0.5)
    link(c, trn, "b", feats, "r", "dep", "«use»", fa=0.3, fb=0.5,
         via=[(549, 650), (455, 650), (455, feats.top - feats.h * 0.5)])

    legend(c, 480, 150, [(MOD_HEAD, "«module» / function"), (HEAD, "class / data class"),
                         (RES_HEAD, "prediction / report returned")])
    c.setFont("Helvetica", 8.4)
    c.setFillColor(SLATE)
    for i, line in enumerate([
        "What each model predicts:",
        "AIDEModel.decide(): per block -> keep?, method (LSB/DCT/DWT), safe bpp.",
        "Steganalyzer.predict_proba(): per image -> P(image hides an AIDE payload).",
        "quantitative.spa()/rs_analysis(): per image -> estimated embedding rate.",
    ]):
        c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 8.4)
        c.drawString(480, 95 - i * 12, line)


# --------------------------------------------------------------------------
# 5. Sequence diagram
# --------------------------------------------------------------------------
def sequence_diagram(c):
    page_header(c, "5. Sequence diagram: embed, attack and recover (sidecar mode, web console)",
                "Solid arrow = call, dashed arrow = return. Loops run once per selected image "
                "block.", 5)
    names = ["Operator", "Web console", "pipeline", "AIDEModel", "crypto", "ecc",
             "embedding.dct", "attacks", "tamper"]
    xs = [70 + i * 132 for i in range(len(names))]
    top, bottom = 760, 50
    for i, (n, x) in enumerate(zip(names, xs)):
        if i == 0:
            actor(c, x, top - 30, "")
            c.setFont("Helvetica-Bold", 9.5)
            c.setFillColor(INK)
            c.drawCentredString(x, top - 52, n)
            ly = top - 58
        else:
            c.setStrokeColor(INK)
            c.setFillColor(HEAD)
            c.setLineWidth(0.9)
            c.rect(x - 55, top - 28, 110, 26, stroke=1, fill=1)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 9.2)
            c.drawCentredString(x, top - 19, n)
            ly = top - 28
        c.setStrokeColor(MUTED)
        c.setDash(3, 3)
        c.line(x, ly, x, bottom)
        c.setDash()

    def act(i, y0, y1):
        c.setFillColor(colors.HexColor("#eef2ff"))
        c.setStrokeColor(INK)
        c.rect(xs[i] - 4, y1, 8, y0 - y1, stroke=1, fill=1)

    msgs = []
    y = [top - 80]

    def msg(a, b, text, ret=False, step=20):
        msgs.append((a, b, text, ret, y[0]))
        y[0] -= step

    def note_y():
        return y[0]

    # activations drawn first, messages on top
    seq = [
        (0, 1, "1: upload cover + message"),
        (1, 2, "2: embed(cover, msg, secret)"),
        (2, 3, "3: decide(features)"),
        (3, 2, "list[BlockDecision]", True),
        (2, 4, "4: encrypt(msg, image_hash, secret)"),
        (4, 2, "EncryptedPayload (ciphertext, nonce, tag)", True),
        (2, 5, "5: plan() + encode() + interleave()"),
        (5, 2, "RS-protected bit stream", True),
        ("loop1",),
        (2, 6, "6: embed(block, bpp, bits, delta)"),
        (6, 2, "modified block", True),
        ("endloop",),
        (2, 8, "7: build_references(gray, blocks)"),
        (8, 2, "list[BlockReference]", True),
        (2, 1, "EmbedResult(stego, sidecar)", True),
        (1, 0, "stego image + sidecar JSON", True),
        ("gap",),
        (0, 1, "8: simulate attack (JPEG)"),
        (1, 7, "9: compression(gray, quality)"),
        (7, 1, "attacked image", True),
        ("gap",),
        (0, 1, "10: recover with secret"),
        (1, 2, "11: recover(stego, sidecar)"),
        (2, 8, "12: analyze(gray, blocks, refs)"),
        (8, 2, "TamperReport (erasure_mask)", True),
        ("loop2",),
        (2, 6, "13: extract_soft(block, bpp)"),
        (6, 2, "signed margins (soft bits)", True),
        ("endloop",),
        (2, 5, "14: decode_with_reliability(bits, margins, erasures)"),
        (5, 2, "ciphertext + DecodeStats", True),
        (2, 4, "15: decrypt(payload, image_hash, secret)"),
        (4, 2, "plaintext (GCM tag verified)", True),
        (2, 1, "RecoverResult(success, plaintext)", True),
        (1, 0, "message + repair stats", True),
    ]
    loops = []
    for s in seq:
        if s[0] in ("loop1", "loop2"):
            loops.append([y[0] + 8, None, "loop [each selected block]"])
            y[0] -= 14
        elif s[0] == "endloop":
            loops[-1][1] = y[0] + 8
            y[0] -= 6
        elif s[0] == "gap":
            y[0] -= 14
        else:
            msg(s[0], s[1], s[2], len(s) > 3)

    # activation bars
    def span(i, first, last):
        act(i, msgs[first][4] + 4, msgs[last][4] - 4)

    span(1, 0, 13)
    span(2, 1, 12)
    span(1, 14, 16)
    span(1, 17, 28)
    span(2, 18, 27)
    for a, b, _, ret, yy in msgs:
        if not ret and a == 2:
            target = b
            # short activation on callee until its return
            nxt = next((m for m in msgs if m[3] and m[0] == target and m[4] < yy), None)
            if nxt:
                act(target, yy + 3, nxt[4] - 3)
        if not ret and a == 1 and b == 7:
            act(7, yy + 3, yy - 17)

    for top_y, bot_y, label in loops:
        c.setStrokeColor(SLATE)
        c.setLineWidth(0.8)
        c.rect(xs[2] - 30, bot_y - 4, xs[6] - xs[2] + 90, top_y - bot_y + 8, stroke=1, fill=0)
        c.setFillColor(colors.white)
        c.rect(xs[2] - 30, top_y - 10, 128, 14, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 7.6)
        c.drawString(xs[2] - 26, top_y - 6.5, label)

    for a, b, text, ret, yy in msgs:
        xa, xb = xs[a], xs[b]
        off = 4 if xb > xa else -4
        c.setStrokeColor(INK)
        c.setLineWidth(0.8)
        c.setDash(4, 3) if ret else c.setDash()
        c.line(xa + off, yy, xb - off, yy)
        c.setDash()
        if ret:
            _arrow_open(c, (xb - off, yy), (xa, yy), 7)
        else:
            _triangle(c, (xb - off, yy), (xa, yy), 7, fill=INK)
        c.setFont("Helvetica-Oblique" if ret else "Helvetica", 7.8)
        c.setFillColor(SLATE if ret else INK)
        mid = (xa + xb) / 2
        c.drawCentredString(mid, yy + 3, text)

    c.setFont("Helvetica", 8)
    c.setFillColor(SLATE)
    c.drawString(xs[4] - 50, msgs[28][4] - 24,
                 "Without the right secret step 15 fails: the GCM tag check rejects it, "
                 "so recovery never returns a corrupted message.")


def main():
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("AIDE UML Diagrams")
    c.setAuthor("Yogishwar")
    for draw in (component_diagram, usecase_diagram, class_diagram_operator,
                 class_diagram_engine, sequence_diagram):
        draw(c)
        c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
