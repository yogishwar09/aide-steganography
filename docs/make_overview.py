"""Generate the AIDE project overview handout (A4) and presentation deck
(16:9). Both answer the same five questions: what was built, which
technologies it uses, what it is for, what it predicts, and future scope.

Run docs/assets figures first (cover/stego/embedmap/diff PNGs), then:
    python docs/make_overview.py
"""
from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DOCS = Path(__file__).resolve().parent
ASSETS = DOCS / "assets"
OUT_REPORT = DOCS / "AIDE_Project_Overview.pdf"
OUT_DECK = DOCS / "AIDE_Presentation.pdf"

INK = colors.HexColor("#0f172a")
SLATE = colors.HexColor("#475569")
MUTED = colors.HexColor("#94a3b8")
INDIGO = colors.HexColor("#4f46e5")
INDIGO_SOFT = colors.HexColor("#eef2ff")
LINE = colors.HexColor("#e2e8f0")
GREEN = colors.HexColor("#059669")
AMBER = colors.HexColor("#d97706")
PROFILE_COLORS = [colors.HexColor(c) for c in ("#a5b4fc", "#6366f1", "#4338ca", "#1e1b4b")]

# --------------------------------------------------------------------------
# Shared content (measured numbers from README, benchmark and data/models)
# --------------------------------------------------------------------------
TECH = [
    ("Language / runtime", "Python 3.10-3.12, CPU-only (no GPU or CUDA needed)"),
    ("Image processing", "OpenCV, NumPy, SciPy, Pillow, scikit-image, PyWavelets (DWT)"),
    ("Machine learning", "scikit-learn (decision tree / tiny MLP, gradient-boosted trees), joblib"),
    ("Cryptography", "pycryptodome: AES-256-GCM authenticated encryption, scrypt key derivation"),
    ("Error correction", "reedsolo: Reed-Solomon codes, interleaving, GMD reliability decoding"),
    ("Web + API", "Flask console, versioned REST API with OpenAPI spec, gunicorn, API-key auth"),
    ("Engineering", "pytest (146 tests), ruff lint, GitHub Actions CI, Docker, Makefile"),
    ("Dataset", "BOSSbase 1.01 real photographs (334 images, 342k blocks for training)"),
]

ROBUST_ROWS = [  # scenario, quality, balanced, robust, fortress (% bit-exact recovery, 30 covers)
    ("JPEG Q50", 100, 100, 100, 100),
    ("JPEG Q30", 20, 100, 100, 100),
    ("Noise s=10", 0, 100, 100, 100),
    ("Noise s=20", 0, 0, 100, 100),
    ("Salt&pepper", 0, 0, 100, 100),
    ("Median 3x3", 52, 100, 100, 100),
    ("Rescale 0.5x", 40, 92, 96, 100),
    ("Crop 10%", 84, 88, 100, 100),
]

PROFILES = [
    ("Profile", "PSNR", "SSIM", "Built for"),
    ("quality", "43.4 dB", "0.984", "Lossless channels, JPEG >= Q50"),
    ("balanced (default)", "40.1 dB", "0.967", "JPEG >= Q30, mild noise, filtering, rescaling"),
    ("robust", "35.0 dB", "0.912", "Noise s=20, salt & pepper, 10% crop (objective O4)"),
    ("fortress", "30.5 dB", "0.803", "Hostile channels, short messages"),
]

OBJECTIVES = [
    ("Objective", "Target", "Status"),
    ("O1 Lightweight model", "< 5 ms/block on CPU", "Met: 0.16 ms/block, 13 KB model"),
    ("O2 Adaptive payload", "Per-block map, unique per image", "Met"),
    ("O3 Encryption", "AES-256-GCM, image-bound key", "Met, hardened with scrypt"),
    ("O4 Robustness", "> 98% at JPEG Q50 and noise s=20", "Met by 'robust' (100%)"),
    ("O5 Tamper localization", "Per-block hash map + statistics", "Met"),
    ("O6 Recovery", "> 98% under moderate attacks", "Met by 'robust' (15/16 at 100%)"),
    ("O7 vs CNN baselines", ">= 4x faster, < 10 MB", "Partial: fast + tiny, not yet compared"),
    ("Image quality", "PSNR > 42 dB, SSIM > 0.98", "Met by 'quality' (43.4 dB / 0.984)"),
]

PREDICTIONS = [
    ("Model", "Input", "What it predicts", "Measured result"),
    ("Decision engine\n(decision tree or MLP)",
     "4 features per image block:\ntexture, entropy, variance, edge",
     "Which embedding method suits the block (LSB / DCT / DWT), how many bits it can "
     "safely carry (bpp), or SKIP",
     "87.7% validation accuracy;\nbpp error 0.14 bpp (MAE)"),
    ("ML steganalyser\n(gradient-boosted trees)",
     "126 features per image:\nSPAM pixel-adjacency +\nDCT pair-gap histograms",
     "Probability that an image hides an AIDE payload (clean vs stego)",
     "AUC 0.999 on 100 held-out\ncovers, 1% false positives"),
    ("SPA / RS analysis\n(statistical estimators)",
     "Pixel-pair statistics\nof one image",
     "Estimated embedding rate, i.e. roughly how much data is hidden (LSB family)",
     "Validated against ground truth"),
    ("Chi-square PoV\n(statistical test)",
     "Pixel value histogram",
     "Yes / no: sequential LSB embedding present",
     "Classical baseline"),
]

FUTURE = [
    ("Beat targeted steganalysis",
     "Side-informed embedding with syndrome-trellis codes (STC) or keyed, dithered pair "
     "selection, so the DCT pair-gap fingerprint the ML detector finds disappears."),
    ("Survive geometric attacks",
     "Rotation, real cropping and aspect changes break the block grid. Add a registration "
     "template or embed in an invariant domain (e.g. log-polar / Fourier-Mellin)."),
    ("Scale the training data",
     "Retrain on ALASKA2 (80k+ images) and several camera sources so both models generalise "
     "beyond BOSSbase."),
    ("Colour-aware pipeline",
     "Embed in and model chroma channels, including JPEG chroma subsampling, instead of "
     "luma only."),
    ("Benchmark against deep models",
     "Run the SteganoGAN / HiDDeN comparison (objective O7) and measure on a Raspberry Pi "
     "for edge deployment."),
    ("Wider media and deployment",
     "Video frames and audio carriers, a mobile client, and a hardened multi-user "
     "server with audit logging."),
]

# --------------------------------------------------------------------------
# Shared drawing: robustness bar chart
# --------------------------------------------------------------------------
def robustness_chart(width, height, font_size=7):
    d = Drawing(width, height)
    chart = VerticalBarChart()
    chart.x, chart.y = 30, 34
    chart.width, chart.height = width - 40, height - 56
    chart.data = [[r[i] for r in ROBUST_ROWS] for i in range(1, 5)]
    chart.categoryAxis.categoryNames = [r[0] for r in ROBUST_ROWS]
    chart.categoryAxis.labels.fontSize = font_size
    chart.categoryAxis.labels.fillColor = SLATE
    chart.categoryAxis.strokeColor = LINE
    chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = 0, 100, 25
    chart.valueAxis.labels.fontSize = font_size
    chart.valueAxis.labels.fillColor = SLATE
    chart.valueAxis.strokeColor = LINE
    chart.valueAxis.gridStrokeColor = LINE
    chart.valueAxis.visibleGrid = True
    chart.barSpacing, chart.groupSpacing = 1, 6
    for i, c in enumerate(PROFILE_COLORS):
        chart.bars[i].fillColor = c
        chart.bars[i].strokeColor = None
    d.add(chart)
    legend = Legend()
    legend.x, legend.y = 34, height - 6
    legend.alignment = "right"
    legend.columnMaximum = 1
    legend.fontSize = font_size + 0.5
    legend.fillColor = SLATE
    legend.deltax = 70
    legend.colorNamePairs = list(zip(PROFILE_COLORS, ["quality", "balanced", "robust", "fortress"]))
    legend.dxTextSpace = 4
    legend.boxAnchor = "nw"
    d.add(legend)
    d.add(String(2, 34 + chart.height / 2, "%", fontSize=font_size, fillColor=SLATE))
    return d


# --------------------------------------------------------------------------
# 1. A4 overview report
# --------------------------------------------------------------------------
def build_report():
    ss = getSampleStyleSheet()
    S = {
        "title": ParagraphStyle("t", ss["Title"], fontName="Helvetica-Bold", fontSize=26,
                                leading=30, textColor=INK, alignment=TA_LEFT, spaceAfter=4),
        "sub": ParagraphStyle("s", ss["Normal"], fontSize=11.5, leading=16, textColor=SLATE),
        "h1": ParagraphStyle("h1", ss["Heading1"], fontName="Helvetica-Bold", fontSize=16,
                             leading=20, textColor=INK, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
                             leading=15, textColor=INDIGO, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("b", ss["Normal"], fontSize=9.8, leading=14.2, textColor=INK,
                               spaceAfter=6),
        "bullet": ParagraphStyle("bl", ss["Normal"], fontSize=9.8, leading=14, textColor=INK,
                                 leftIndent=12, bulletIndent=2, spaceAfter=3),
        "cell": ParagraphStyle("c", ss["Normal"], fontSize=8.6, leading=11.2, textColor=INK),
        "cellb": ParagraphStyle("cb", ss["Normal"], fontName="Helvetica-Bold", fontSize=8.6,
                                leading=11.2, textColor=INK),
        "head": ParagraphStyle("hd", ss["Normal"], fontName="Helvetica-Bold", fontSize=8.6,
                               leading=11.2, textColor=colors.white),
        "cap": ParagraphStyle("cap", ss["Normal"], fontSize=8, leading=10.5, textColor=SLATE),
    }

    def P(text, style="body"):
        return Paragraph(text, S[style])

    def bullets(items):
        return [Paragraph(t, S["bullet"], bulletText="•") for t in items]

    def table(rows, widths, bold_first_col=True):
        data = [[Paragraph(c.replace("\n", "<br/>"), S["head"]) for c in rows[0]]]
        for r in rows[1:]:
            data.append([Paragraph(str(c).replace("\n", "<br/>"),
                                   S["cellb"] if (j == 0 and bold_first_col) else S["cell"])
                         for j, c in enumerate(r)])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    def callout(text):
        t = Table([[Paragraph(text, S["body"])]], colWidths=[170 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), INDIGO_SOFT),
            ("LINEBEFORE", (0, 0), (0, -1), 3, INDIGO),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    def figure_row():
        names = [("cover.png", "Original cover"), ("stego.png", "Stego (message hidden)"),
                 ("embedmap.png", "Blocks AIDE chose"), ("diff.png", "Change map (x10)")]
        w = 40 * mm
        imgs = [Image(str(ASSETS / n), width=w, height=w) for n, _ in names]
        caps = [Paragraph(c, S["cap"]) for _, c in names]
        t = Table([imgs, caps], colWidths=[42.5 * mm] * 4)
        t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 1),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 1)]))
        return t

    story = [
        Spacer(1, 6 * mm),
        P("AIDE", "title"),
        P("Adaptive Intelligent Dynamic Embedding &mdash; project overview", "sub"),
        Spacer(1, 2 * mm),
        P("CD753PC Project, MGIT Hyderabad &middot; Version 0.2 &middot; October 2026", "cap"),
        Spacer(1, 6 * mm),
        callout("<b>In one sentence:</b> AIDE hides an encrypted message inside an ordinary "
                "photograph so that it looks unchanged, survives compression, noise and "
                "cropping on the way, can be proven untampered on arrival &mdash; and it also "
                "ships the tools an analyst needs to detect hidden data in intercepted images."),
        Spacer(1, 4 * mm),
    ]

    # 1. What we built
    story += [P("1. What we built", "h1"),
              P("AIDE is a complete, working steganography system &mdash; not a notebook "
                "experiment. It covers both sides of the defence and intelligence use case "
                "from the project report:"),
              *bullets([
                  "<b>Operator side (hide):</b> encrypt a message with AES-256-GCM, protect it "
                  "with Reed-Solomon error correction, and embed it into the image blocks a "
                  "trained AI model picks as safest. The receiver recovers the exact message "
                  "or gets a clear error &mdash; never a silently corrupted one.",
                  "<b>Analyst side (detect):</b> screen suspicious images with no key, using "
                  "chi-square, Sample Pairs and RS analysis plus a trained machine-learning "
                  "steganalyser, with a spatial suspicion heatmap and batch CSV screening.",
                  "<b>Tamper evidence:</b> per-block SHA-256 hashes and histogram tests mark "
                  "exactly which regions were altered, and those regions are fed to the "
                  "error-correction decoder as erasures.",
                  "<b>Four robustness profiles</b> (quality, balanced, robust, fortress) and "
                  "two modes: <i>sidecar</i> (small JSON map travels with the image) and "
                  "<i>blind</i> (only the image and the shared secret are needed).",
                  "<b>Three ways to use it:</b> the <font face='Courier'>aide</font> "
                  "command-line tool, a web console (Operator / Analyst / Model Insights), "
                  "and a versioned REST API with an OpenAPI contract.",
              ]),
              P("How a message travels through AIDE", "h2"),
              table([
                  ("Step", "What happens"),
                  ("1. Analyse", "Image split into blocks; texture, entropy, variance and edge "
                                 "strength measured for each"),
                  ("2. Decide", "Trained model scores each block S(b) and picks method "
                                "(DCT/DWT/LSB), bits per block, or skip"),
                  ("3. Encrypt", "AES-256-GCM with a key derived by scrypt from the shared "
                                 "secret (and the image)"),
                  ("4. Protect", "Reed-Solomon parity, shortened codewords and interleaving "
                                 "spread damage across codewords"),
                  ("5. Embed", "Bits written as DCT coefficient-pair relations (QIM) in the "
                               "chosen blocks"),
                  ("6. Transmit", "Image may be JPEG-compressed, resized, noised, filtered or "
                                  "cropped on the way"),
                  ("7. Recover", "Tamper map finds damaged blocks; GMD decoding repairs bytes; "
                                 "GCM tag proves the message is exact"),
              ], [28 * mm, 142 * mm]),
              Spacer(1, 4 * mm),
              KeepTogether([figure_row(),
                            P("Real output from this repository: a 76-byte message hidden in a "
                              "BOSSbase photo with the balanced profile (PSNR 42.0 dB) and "
                              "recovered bit-exact.", "cap")])]

    # 2. Technologies
    story += [P("2. Technologies", "h1"),
              table([("Area", "Technology")] + TECH, [42 * mm, 128 * mm]),
              P("Core techniques", "h2"),
              *bullets([
                  "<b>Transform-domain embedding:</b> DCT mid-frequency coefficient-pair "
                  "quantisation index modulation (QIM), 2-level Haar DWT, and LSB as a "
                  "fallback for smooth blocks.",
                  "<b>Learned block scoring:</b> S(b) = &alpha;&middot;Texture + "
                  "&beta;&middot;Entropy + &gamma;&middot;Variance + &delta;&middot;Edge. "
                  "The weights are learned, not hand-set: &alpha; = 0.55, &delta; = 0.44, "
                  "entropy and variance near zero.",
                  "<b>Security:</b> scrypt (memory-hard) key derivation, authenticated "
                  "encryption, keyed pseudo-random block layout in blind mode.",
                  "<b>Steganalysis features:</b> SPAM (subtractive pixel adjacency matrix) "
                  "and DCT pair-gap histograms for the ML detector.",
              ]),
              P("Architecture at a glance", "h2"),
              table([
                  ("Module", "Responsibility"),
                  ("features.py / scoring.py / model.py", "Block features, S(b) scoring, trained "
                                                          "decision engine"),
                  ("crypto.py / ecc.py", "AES-256-GCM + scrypt; Reed-Solomon, interleaving, GMD"),
                  ("embedding/ (lsb, dct, dwt)", "Embed and extract (hard and soft decisions)"),
                  ("pipeline.py / blind.py", "Sidecar and blind end-to-end embed / recover"),
                  ("tamper.py / attacks.py / metrics.py", "Tamper map; 8 attack simulations; "
                                                          "PSNR, SSIM, BER"),
                  ("steganalysis.py / quantitative.py / mlsteg.py", "Classical, SPA/RS and ML "
                                                                    "detection"),
                  ("webapp/app.py, cli.py", "Web console + REST API; aide command line"),
              ], [62 * mm, 108 * mm]),
              PageBreak()]

    # 3. What it is for
    story += [P("3. What we are supposed to do with this project", "h1"),
              P("AIDE has three audiences. Each one uses it differently:"),
              table([
                  ("Who", "Goal", "How they use AIDE"),
                  ("Field operator /\nsecure communicator",
                   "Send a confidential message through channels that recompress or damage "
                   "images, without drawing attention",
                   "Pick a profile, check capacity, embed with a shared secret, send the "
                   "image; receiver extracts with the same secret"),
                  ("Intelligence /\nforensic analyst",
                   "Find out whether intercepted media carries hidden data and roughly how "
                   "much",
                   "Analyst tab or aide detect: one verdict per image, payload estimate, "
                   "heatmap, batch CSV for whole folders"),
                  ("Student / researcher",
                   "Learn and evaluate modern steganography and steganalysis with honest "
                   "measurements",
                   "Run the benchmark, compare profiles, retrain models, read Model Insights, "
                   "extend modules"),
              ], [34 * mm, 58 * mm, 78 * mm]),
              P("Typical workflow (demo)", "h2"),
              *bullets([
                  "Start the console: <font face='Courier'>python webapp/app.py</font> and "
                  "open http://127.0.0.1:5050.",
                  "<b>Operator tab:</b> upload a cover, type a message and secret, embed. "
                  "Then simulate an attack (JPEG, noise, crop...), run the tamper check and "
                  "recover the message.",
                  "<b>Analyst tab:</b> upload the stego image and see what each detector "
                  "family says about it.",
                  "<b>Model Insights:</b> show the learned weights, the robustness matrix and "
                  "the detector evaluation.",
              ]),
              P("Robustness profiles (measured on 30 BOSSbase covers)", "h2"),
              table(PROFILES, [36 * mm, 20 * mm, 16 * mm, 98 * mm]),
              Spacer(1, 3 * mm),
              robustness_chart(170 * mm, 62 * mm),
              P("Bit-exact message recovery after each attack, by profile. Stronger profiles "
                "trade image quality for survival.", "cap"),
              PageBreak()]

    # 4/5. Predictions
    story += [P("4. What we are predicting", "h1"),
              P("AIDE never tries to predict or guess the hidden message &mdash; that is "
                "protected by AES-256-GCM. The machine-learning parts predict properties of "
                "<b>images and image blocks</b>:"),
              table(PREDICTIONS, [33 * mm, 38 * mm, 58 * mm, 41 * mm]),
              Spacer(1, 3 * mm),
              P("How the decision engine is trained", "h2"),
              *bullets([
                  "<b>Method labels</b> come from the report's rule (low texture &rarr; LSB, "
                  "strong edges &rarr; DWT, otherwise DCT) with thresholds at the 33rd/67th "
                  "percentiles, so classes are balanced. The model learns to reproduce and "
                  "smooth this rule from all four features.",
                  "<b>Payload labels are measured:</b> random bits are really embedded at "
                  "4/3/2/1 bpp and block PSNR is measured; the label is the highest safe bpp, "
                  "or SKIP. Target block quality: 42 dB.",
                  "Trained on 334 BOSSbase images (342,016 blocks). Inference costs about "
                  "0.16 ms per block on one CPU core; the model file is 13 KB.",
              ]),
              P("Honest finding on detectability", "h2"),
              callout("AIDE images pass all classical blind detectors (chi-square, SPA, RS: "
                      "0/15 flagged). But the targeted ML steganalyser, which knows the "
                      "embedding family, detects every AIDE variant at <b>100% with AUC "
                      "0.998-1.000</b>. The reason is structural: pair QIM forces "
                      "|c<sub>i</sub> - c<sub>j</sub>| &ge; &delta;, leaving a gap natural "
                      "photos do not have. This is the main item in future scope."),
              P("Project objectives: status", "h2"),
              table(OBJECTIVES, [40 * mm, 58 * mm, 72 * mm]),
              PageBreak()]

    story += [P("5. Future scope", "h1"),
              table([("Direction", "What it means")] + FUTURE, [45 * mm, 125 * mm]),
              P("Current limitations to keep in mind", "h2"),
              *bullets([
                  "<b>quality</b> profile does not survive noise (0% at &sigma; &ge; 10) by "
                  "design; use balanced or stronger for noisy channels.",
                  "Messages close to a cover's capacity are less robust than the 156-byte "
                  "benchmark runs; <font face='Courier'>aide capacity</font> reports both limits.",
                  "The ML detector is trained on one camera source; its out-of-distribution "
                  "guard only softens verdicts on unfamiliar images.",
              ]),
              P("Quality and verification", "h2"),
              P("Every number in this document is measured by code in the repository: 146 "
                "automated tests (unit, ground-truth, end-to-end CLI/HTTP, attack recovery), "
                "lint, and CI on Python 3.10-3.12 with a Docker build and health check. The "
                "full benchmark is in <font face='Courier'>reports/benchmark/</font>."),
              Spacer(1, 4 * mm),
              callout("<b>Summary.</b> AIDE delivers adaptive, encrypted, error-corrected "
                      "image steganography that runs on a plain CPU, meets the report's "
                      "robustness and recovery targets with the robust profile, and is honest "
                      "about its one big open problem: resistance to targeted ML "
                      "steganalysis.")]

    def on_page(c, doc):
        c.saveState()
        c.setFillColor(INDIGO)
        c.rect(0, A4[1] - 4, A4[0], 4, stroke=0, fill=1)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(MUTED)
        c.drawString(20 * mm, 10 * mm, "AIDE - Adaptive Intelligent Dynamic Embedding")
        c.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(str(OUT_REPORT), pagesize=A4, leftMargin=20 * mm,
                            rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                            title="AIDE Project Overview", author="Yogishwar")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


# --------------------------------------------------------------------------
# 2. 16:9 presentation deck
# --------------------------------------------------------------------------
W, H = 960, 540


class Deck:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(W, H))
        self.c.setTitle("AIDE Presentation")
        self.c.setAuthor("Yogishwar")
        self.n = 0

    def _chrome(self, section):
        c = self.c
        self.n += 1
        c.setFillColor(colors.white)
        c.rect(0, 0, W, H, stroke=0, fill=1)
        c.setFillColor(INDIGO)
        c.rect(0, H - 6, W, 6, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(INDIGO)
        c.drawString(48, H - 40, section.upper())
        c.setFont("Helvetica", 9)
        c.setFillColor(MUTED)
        c.drawString(48, 22, "AIDE - Adaptive Intelligent Dynamic Embedding")
        c.drawRightString(W - 48, 22, str(self.n))

    def title(self, text, y=H - 82, size=28):
        self.c.setFont("Helvetica-Bold", size)
        self.c.setFillColor(INK)
        self.c.drawString(48, y, text)

    def para(self, text, x, y, width, size=14, color=INK, font="Helvetica", leading=None):
        """Draw word-wrapped text; returns the y below it."""
        leading = leading or size * 1.35
        c = self.c
        c.setFont(font, size)
        c.setFillColor(color)
        for raw in text.split("\n"):
            line = ""
            for word in raw.split():
                trial = (line + " " + word).strip()
                if c.stringWidth(trial, font, size) > width and line:
                    c.drawString(x, y, line)
                    y -= leading
                    line = word
                else:
                    line = trial
            c.drawString(x, y, line)
            y -= leading
        return y

    def bullets(self, items, x=48, y=H - 140, width=860, size=20, gap=22):
        for item in items:
            head, _, rest = item.partition("|")
            self.c.setFillColor(INDIGO)
            self.c.circle(x + 4, y + size * 0.32, 3, stroke=0, fill=1)
            if rest:
                self.c.setFont("Helvetica-Bold", size)
                self.c.setFillColor(INK)
                self.c.drawString(x + 18, y, head)
                y = self.para(rest.strip(), x + 18, y - size * 1.35, width - 18, size - 2,
                              SLATE)
            else:
                y = self.para(head, x + 18, y, width - 18, size)
            y -= gap
        return y

    def card(self, x, y, w, h, heading, body, accent=INDIGO, size=12):
        c = self.c
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setStrokeColor(LINE)
        c.roundRect(x, y, w, h, 8, stroke=1, fill=1)
        c.setFillColor(accent)
        c.rect(x, y + h - 4, w, 4, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", size + 2)
        c.setFillColor(INK)
        c.drawString(x + 14, y + h - 28, heading)
        self.para(body, x + 14, y + h - 50, w - 28, size, SLATE)

    def table(self, rows, x, y, widths, size=11, row_h=26):
        c = self.c
        total = sum(widths)
        for i, row in enumerate(rows):
            ry = y - (i + 1) * row_h
            if i == 0:
                c.setFillColor(INK)
                c.rect(x, ry, total, row_h, stroke=0, fill=1)
            elif i % 2 == 0:
                c.setFillColor(colors.HexColor("#f8fafc"))
                c.rect(x, ry, total, row_h, stroke=0, fill=1)
            cx = x
            for j, (cell, w) in enumerate(zip(row, widths)):
                font = "Helvetica-Bold" if i == 0 or j == 0 else "Helvetica"
                c.setFont(font, size)
                c.setFillColor(colors.white if i == 0 else INK)
                c.drawString(cx + 8, ry + row_h / 2 - size * 0.35, str(cell))
                cx += w
            c.setStrokeColor(LINE)
            c.line(x, ry, x + total, ry)
        return y - len(rows) * row_h

    def next(self):
        self.c.showPage()

    def save(self):
        self.c.save()


def build_deck():
    d = Deck(OUT_DECK)
    c = d.c

    # 1 Title
    d.n += 1
    c.setFillColor(INK)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(INDIGO)
    c.rect(48, 300, 60, 6, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 64)
    c.setFillColor(colors.white)
    c.drawString(48, 220, "AIDE")
    c.setFont("Helvetica", 22)
    c.setFillColor(colors.HexColor("#c7d2fe"))
    c.drawString(48, 182, "Adaptive Intelligent Dynamic Embedding")
    c.setFont("Helvetica", 14)
    c.setFillColor(MUTED)
    c.drawString(48, 150, "AI-guided, encrypted, attack-resilient image steganography + steganalysis")
    c.drawString(48, 70, "CD753PC Project  |  MGIT Hyderabad  |  October 2026")
    c.drawImage(str(ASSETS / "embedmap.png"), 620, 170, 290, 290)
    c.setFont("Helvetica", 11)
    c.drawString(620, 150, "Blocks the AI chose to carry the message")
    d.next()

    # 2 Problem
    d._chrome("Why this project")
    d.title("The problem")
    d.bullets([
        "Hidden messages break in transit|Social media and messengers recompress, resize and "
        "filter images; classic LSB steganography loses the message after one JPEG save.",
        "Encryption alone draws attention|An encrypted file announces that something secret "
        "is being sent. Steganography hides that a message exists at all.",
        "Defenders need the opposite tool|Analysts must screen intercepted media for hidden "
        "payloads without knowing any key.",
        "Most research is GPU-heavy|Deep models (SteganoGAN, HiDDeN) need GPUs; field "
        "devices need something that runs on a plain CPU.",
    ])
    d.next()

    # 3 What we built
    d._chrome("1. What we built")
    d.title("A complete hide-and-detect system")
    cw, ch = 205, 250
    cards = [
        ("Hide", "Encrypt with AES-256-GCM, add Reed-Solomon error correction, embed into "
                 "blocks chosen by a trained AI model."),
        ("Survive", "Four measured robustness profiles survive JPEG, noise, filtering, "
                    "rescaling, splicing and cropping."),
        ("Verify", "Per-block hash tamper map shows what was altered; recovery is bit-exact "
                   "or a clear error."),
        ("Detect", "Analyst toolkit: chi-square, SPA, RS analysis and an ML steganalyser "
                   "with heatmaps and batch CSV."),
    ]
    for i, (h, b) in enumerate(cards):
        d.card(48 + i * (cw + 13), 170, cw, ch, h, b, PROFILE_COLORS[min(i, 3)] if i else INDIGO,
               size=15)
    d.para("Delivered as a CLI tool, a web console (Operator / Analyst / Model Insights) and a "
           "versioned REST API.  146 automated tests.", 48, 130, 860, 14, SLATE)
    d.next()

    # 4 Pipeline
    d._chrome("1. What we built")
    d.title("How a message travels through AIDE")
    steps = [("Analyse", "Block features"), ("Decide", "AI picks blocks"),
             ("Encrypt", "AES-256-GCM"), ("Protect", "Reed-Solomon"),
             ("Embed", "DCT pair QIM"), ("Transmit", "Attacks happen"),
             ("Recover", "Exact message")]
    bw, gap = 112, 12
    for i, (h, s) in enumerate(steps):
        x = 48 + i * (bw + gap)
        c.setFillColor(INDIGO if i in (1, 6) else INDIGO_SOFT)
        c.roundRect(x, 330, bw, 80, 8, stroke=0, fill=1)
        c.setFillColor(colors.white if i in (1, 6) else INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawCentredString(x + bw / 2, 378, h)
        c.setFont("Helvetica", 11)
        c.drawCentredString(x + bw / 2, 356, s)
        if i < len(steps) - 1:
            c.setFillColor(MUTED)
            c.drawCentredString(x + bw + gap / 2, 364, ">")
    labels = ["Original cover", "Stego image", "Blocks AIDE chose", "Change map (x10)"]
    for i, (name, lab) in enumerate(zip(["cover.png", "stego.png", "embedmap.png", "diff.png"],
                                        labels)):
        x = 48 + i * 220
        c.drawImage(str(ASSETS / name), x, 90, 200, 200)
        c.setFont("Helvetica", 11)
        c.setFillColor(SLATE)
        c.drawString(x, 72, lab)
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(GREEN)
    c.drawString(48, 50, "Real run: 76-byte message, balanced profile, PSNR 42.0 dB, recovered bit-exact")
    d.next()

    # 5 Technologies
    d._chrome("2. Technologies")
    d.title("Technology stack")
    d.table([("Area", "Technology")] + TECH, 48, H - 110, [210, 650], size=14, row_h=42)
    d.next()

    # 6 Core techniques
    d._chrome("2. Technologies")
    d.title("Core techniques")
    d.bullets([
        "Learned block score|S(b) = a*Texture + b*Entropy + g*Variance + d*Edge. Learned "
        "weights: texture 0.55, edge 0.44, entropy and variance near zero.",
        "Transform-domain embedding|DCT mid-frequency coefficient-pair QIM (main carrier), "
        "2-level Haar DWT, LSB only for very smooth blocks.",
        "Error correction|Reed-Solomon with shortened codewords, interleaving and GMD "
        "reliability-guided decoding; tamper map marks erasures.",
        "Security|scrypt memory-hard key derivation, AES-256-GCM authentication, keyed "
        "pseudo-random layout in blind mode.",
    ])
    d.next()

    # 7 Use
    d._chrome("3. What it is for")
    d.title("Who uses AIDE, and how")
    who = [
        ("Operator", "Send confidential messages through channels that compress or damage "
                     "images.\n\nEmbed with a shared secret, pick a profile, send. Receiver "
                     "extracts with the same secret."),
        ("Analyst", "Check intercepted images for hidden data, with no key.\n\nOne verdict "
                    "per image, payload-size estimate, suspicion heatmap, batch CSV "
                    "screening."),
        ("Researcher", "Study steganography and steganalysis with honest, reproducible "
                       "numbers.\n\nRun the benchmark, retrain models, compare profiles, "
                       "extend modules."),
    ]
    for i, (h, b) in enumerate(who):
        d.card(48 + i * 292, 90, 276, 340, h, b, size=16)
    d.next()

    # 8 Profiles + chart
    d._chrome("3. What it is for")
    d.title("Choose the trade-off: quality vs robustness")
    d.table(PROFILES, 48, H - 110, [170, 90, 80, 520], size=13, row_h=30)
    from reportlab.graphics import renderPDF
    renderPDF.draw(robustness_chart(860, 210, font_size=9), c, 48, 50)
    d.next()

    # 9 Predictions
    d._chrome("4. What we are predicting")
    d.title("What the models predict")
    d.para("Never the message itself (that is AES-encrypted). The models predict properties "
           "of images and blocks.", 48, H - 112, 860, 14, SLATE)
    preds = [
        ("Decision engine", "Per block: best method (LSB / DCT / DWT), safe bits per pixel, "
                            "or SKIP.", "87.7% validation accuracy\nbpp error 0.14 (MAE)"),
        ("ML steganalyser", "Per image: probability it hides an AIDE payload (126 SPAM + DCT "
                            "pair-gap features).", "AUC 0.999, 100 held-out covers\n1% false "
                                                   "positives"),
        ("SPA / RS analysis", "Per image: estimated embedding rate, i.e. how much data is "
                              "hidden (LSB family).", "Validated against ground truth"),
    ]
    for i, (h, b, r) in enumerate(preds):
        x = 48 + i * 292
        d.card(x, 90, 276, 320, h, b, size=16)
        c.setFont("Helvetica-Bold", 15)
        c.setFillColor(GREEN)
        yy = 140
        for line in r.split("\n"):
            c.drawString(x + 14, yy, line)
            yy -= 21
    d.next()

    # 10 Training
    d._chrome("4. What we are predicting")
    d.title("How the decision engine learns")
    d.bullets([
        "Real data|334 BOSSbase photographs, 342,016 image blocks.",
        "Method labels from the report's rule|Low texture -> LSB, strong edges -> DWT, "
        "otherwise DCT; thresholds at the 33rd/67th percentile so classes are balanced.",
        "Payload labels are measured, not assumed|Random bits embedded at 4/3/2/1 bpp; block "
        "PSNR measured; label = highest safe bpp (target 42 dB) or SKIP.",
        "Tiny and fast|Decision tree or MLP; 13 KB model, about 0.16 ms per block on one CPU "
        "core (target was < 5 ms).",
    ])
    d.next()

    # 11 Honest finding
    d._chrome("4. What we are predicting")
    d.title("Honest finding: can AIDE be detected?")
    d.card(48, 150, 420, 250, "Classical detectors: NO",
           "Chi-square, Sample Pairs and RS analysis flag 0 of 15 AIDE stego images. "
           "The report's 'negative detectability' claim holds against these tools.",
           GREEN, size=18)
    d.card(492, 150, 420, 250, "Targeted ML detector: YES",
           "Our own steganalyser detects every AIDE variant at 100% (AUC 0.998-1.000). "
           "Pair QIM forces |ci - cj| >= delta, leaving a gap natural photos do not have.",
           AMBER, size=18)
    d.para("We measure this and say it openly; fixing it is the top future-scope item.",
           48, 115, 860, 14, SLATE)
    d.next()

    # 12 Objectives
    d._chrome("Results")
    d.title("Project objectives: status")
    d.table(OBJECTIVES, 48, H - 105, [220, 300, 340], size=14, row_h=44)
    d.next()

    # 13 Future
    d._chrome("5. Future scope")
    d.title("Future scope")
    for i, (h, b) in enumerate(FUTURE):
        col, row = i % 3, i // 3
        d.card(48 + col * 292, 250 - row * 200, 276, 190, h, b, size=13.5)
    d.next()

    # 14 Close
    d.n += 1
    c.setFillColor(INK)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 40)
    c.setFillColor(colors.white)
    c.drawString(48, 360, "Summary")
    y = 310
    for line in [
        "Adaptive, encrypted, error-corrected image steganography on a plain CPU",
        "Meets the robustness and recovery targets with the 'robust' profile",
        "Built-in analyst toolkit: classical + ML steganalysis",
        "Open problem, measured honestly: resistance to targeted ML detection",
    ]:
        c.setFillColor(INDIGO)
        c.circle(56, y + 6, 4, stroke=0, fill=1)
        c.setFont("Helvetica", 18)
        c.setFillColor(colors.HexColor("#e2e8f0"))
        c.drawString(72, y, line)
        y -= 40
    c.setFont("Helvetica", 16)
    c.setFillColor(colors.HexColor("#c7d2fe"))
    c.drawString(48, 80, "Thank you  -  Questions?")
    d.next()
    d.save()


if __name__ == "__main__":
    build_report()
    build_deck()
    print(OUT_REPORT)
    print(OUT_DECK)
