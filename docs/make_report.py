"""Generate AIDE_System_Report.pdf - a detailed technical walkthrough of
what the project does, what runs in the backend after an image is
uploaded, and what has been discovered (measured) building it."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent / "AIDE_System_Report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#12243b")
ACCENT = colors.HexColor("#1f7a8c")
LIGHT = colors.HexColor("#eef3f6")
GREY = colors.HexColor("#555555")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=24, leading=28,
                           textColor=NAVY, spaceAfter=6))
styles.add(ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=11, leading=15,
                           textColor=GREY, spaceAfter=2))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, leading=19,
                           textColor=NAVY, spaceBefore=16, spaceAfter=6))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, leading=16,
                           textColor=ACCENT, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.7, leading=14,
                           alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle("BodyL", parent=styles["Body"], alignment=TA_LEFT))
styles.add(ParagraphStyle("AideBullet", parent=styles["Body"], alignment=TA_LEFT,
                           leftIndent=10, spaceAfter=3))
styles.add(ParagraphStyle("Mono", parent=styles["Normal"], fontName="Courier",
                           fontSize=8.3, leading=11, textColor=NAVY))
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, leading=11,
                           textColor=GREY, spaceAfter=10))
styles.add(ParagraphStyle("Cell", parent=styles["Normal"], fontSize=8.4, leading=11))
styles.add(ParagraphStyle("CellB", parent=styles["Cell"], fontName="Helvetica-Bold"))

story = []


def h1(t): story.append(Paragraph(t, styles["H1"]))
def h2(t): story.append(Paragraph(t, styles["H2"]))
def p(t): story.append(Paragraph(t, styles["Body"]))
def pl(t): story.append(Paragraph(t, styles["BodyL"]))
def sp(h=6): story.append(Spacer(1, h))
def caption(t): story.append(Paragraph(t, styles["Caption"]))


def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(i, styles["AideBullet"]), leftIndent=6, value="•") for i in items],
        bulletType="bullet", start="•", leftIndent=12,
    ))
    sp(4)


def table(data, col_widths, header=True):
    tbl_data = []
    for r_i, row in enumerate(data):
        style_name = "CellB" if (header and r_i == 0) else "Cell"
        tbl_data.append([Paragraph(str(c), styles[style_name]) for c in row])
    t = Table(tbl_data, colWidths=col_widths, hAlign="LEFT")
    ts = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd6de")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        ts += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT])]
    t.setStyle(TableStyle(ts))
    story.append(t)
    sp(8)


# ----------------------------------------------------------------------------
# TITLE
# ----------------------------------------------------------------------------
story.append(Paragraph("AIDE &mdash; System Report", styles["TitleBig"]))
story.append(Paragraph("Adaptive Intelligent Dynamic Embedding Framework", styles["Subtitle"]))
story.append(Paragraph("What the project does &nbsp;&bull;&nbsp; what runs in the backend after "
                       "an image is uploaded &nbsp;&bull;&nbsp; what we are discovering", styles["Subtitle"]))
sp(4)
story.append(Paragraph("P. Yogishwar &nbsp;&bull;&nbsp; 23261A6746 &nbsp;&bull;&nbsp; B.Tech CSE (Data Science), "
                       "MGIT Hyderabad &nbsp;&bull;&nbsp; CD753PC Project Stage&nbsp;I", styles["Subtitle"]))
story.append(Paragraph("Repository: github.com/yogishwar09/aide-steganography", styles["Subtitle"]))
sp(10)

table([
    ["Metric", "Measured value"],
    ["Decision-engine training set", "334 real BOSSbase images, ~342,000 image blocks"],
    ["Method-classifier validation accuracy", "0.877 (bpp regressor MAE ~0.14)"],
    ["Version", "AIDE 0.2 (robustness profiles, blind mode, GMD decoding, SPA/RS + ML steganalysis)"],
    ["Embed time (one CPU core, 512&times;512)", "~150 ms   |   Recover time ~40 ms   |   0.16 ms/block inference"],
    ["Decision-engine model size", "13 KB"],
    ["Stego quality by profile (30 covers)", "quality 43.4 dB / 0.984 &nbsp;|&nbsp; balanced 40.1 / 0.967 "
     "&nbsp;|&nbsp; robust 35.0 / 0.912 &nbsp;|&nbsp; fortress 30.5 / 0.803 (PSNR / SSIM)"],
    ["Report O4 (JPEG Q50 + Gaussian &sigma;=20)", "Met by the robust profile: 24/24 covers, BER 0"],
    ["Automated tests", "146 passing (no expected failures)"],
    ["Hardware requirement", "CPU only &mdash; no GPU / CUDA anywhere"],
], [190, 300])
caption("All numbers in this document are measured from the running system and locked by an "
        "automated test &mdash; nothing here is an aspiration or an estimate.")

story.append(PageBreak())

# ----------------------------------------------------------------------------
# 1. WHAT THE PROJECT DOES
# ----------------------------------------------------------------------------
h1("1.&nbsp;&nbsp;What the project does")

p("AIDE is a CPU-only image steganography system for defence and intelligence settings. It covers "
  "<b>both sides</b> of covert information transfer in a single tool:")

h2("1.1&nbsp;&nbsp;Operator side &mdash; secure covert communication")
p("Hide an encrypted message inside an ordinary-looking photograph so that interception reveals "
  "nothing, and so that the message can still be recovered after the image has been compressed, "
  "filtered, noised, cropped, or partially replaced in transit. This is the sender/receiver role: "
  "satellite &amp; drone imagery relay, military secure communication, government document exchange.")

h2("1.2&nbsp;&nbsp;Analyst side &mdash; blind threat detection (steganalysis)")
p("Take an <b>arbitrary intercepted image</b> &mdash; no shared secret, no side-channel data, no "
  "assumption about whether or how it was modified &mdash; and decide whether it is likely carrying "
  "hidden data. This is the counter-intelligence role: border and port media screening, cyber-defence "
  "command-centre monitoring for covert data exfiltration.")

h2("1.3&nbsp;&nbsp;How you use it")
bullets([
    "<b>Command line:</b> <font face='Courier'>aide embed / extract / capacity / attack / evaluate / "
    "detect / benchmark / train / train-detector / profiles</font>",
    "<b>Web console</b> (<font face='Courier'>python webapp/app.py</font> or gunicorn / Docker): "
    "Operator, Analyst (single image and batch screening) and Model Insights pages.",
    "<b>REST API:</b> versioned <font face='Courier'>/api/v1/*</font> endpoints with an OpenAPI "
    "spec, optional API-key auth, JSON errors and request IDs.",
    "<b>Python API:</b> <font face='Courier'>aide.pipeline.embed() / recover()</font>, "
    "<font face='Courier'>aide.steganalysis.analyze()</font>.",
])

p("The novelty is the <b>AIDE decision engine</b>: a lightweight trained model that decides, "
  "<i>per 8&times;8 or 16&times;16 image block</i>, four things at once &mdash; <i>where</i> to hide "
  "data, <i>how many</i> bits, <i>which</i> algorithm (LSB / DCT / DWT), and <i>whether</i> the block "
  "is reliable enough to use at all. No two images get the same embedding map.")

story.append(PageBreak())

# ----------------------------------------------------------------------------
# 2. BACKEND AFTER UPLOAD
# ----------------------------------------------------------------------------
h1("2.&nbsp;&nbsp;What runs in the backend after you upload an image")

p("Below is the exact sequence the web console / CLI executes. Each step names the module that does "
  "the work, so this doubles as a code map.")

h2("2.1&nbsp;&nbsp;EMBED path (Operator &rarr; &ldquo;Embed message&rdquo;)")

table([
    ["#", "Step", "Module", "What actually happens"],
    ["1", "Load &amp; plane split", "pipeline.load_image",
     "Read the uploaded file's raw bytes. If colour, convert RGB&rarr;YCbCr and take only the Y "
     "(luma) plane; Cb/Cr are kept untouched for later recombination."],
    ["2", "Block partition", "features.partition_blocks",
     "Split the plane into non-overlapping 8&times;8 blocks (16&times;16 for images &ge;512 px). "
     "Each block gets a stable index used everywhere downstream."],
    ["3", "Per-block features", "features.extract_image_features",
     "Four normalised signals per block: texture (Local Binary Pattern + GLCM energy), Shannon "
     "entropy, pixel variance, edge strength (Sobel + Canny)."],
    ["4", "AIDE decision engine", "model.AIDEModel.decide",
     "Trained decision tree (or tiny MLP) scores each block: S(b)=&alpha;&middot;texture + "
     "&beta;&middot;entropy + &gamma;&middot;variance + &delta;&middot;edge. Blocks below threshold "
     "&tau; are skipped; the rest are ranked and assigned a method and a bit-rate. The chosen "
     "<i>robustness profile</i> supplies the coefficient separation, RS strength and bpp cap."],
    ["5", "Encrypt", "crypto.encrypt",
     "AES-256-GCM. Key = scrypt(shared-secret, image-hash) &mdash; memory-hard, so captured images "
     "cannot be brute-forced cheaply against a weak passphrase. A GCM authentication tag is attached."],
    ["6", "Error-correction code", "ecc.plan / ecc.encode",
     "Reed-Solomon with profile-specific parity (e.g. 64 parity bytes per 255 for <i>balanced</i>). "
     "Short messages are split into &ge;4 shortened codewords at the same parity ratio so "
     "interleaving can still spread damage."],
    ["7", "Interleave", "ecc.interleave",
     "Scatter each codeword's bytes across many non-adjacent blocks, so a localised loss (a crop, a "
     "splice) damages a few bytes of many codewords instead of destroying whole codewords."],
    ["8", "Adaptive embedding", "embedding.dct (lsb / dwt optional)",
     "Write the interleaved bits as mid-frequency DCT coefficient-pair QIM, preferring the 8 "
     "lowest robust pairs per block (relaxed only if the message would not fit). LSB/DWT blocks "
     "are left untouched by default &mdash; measured to cause ~40% of errors on ~2% of bits."],
    ["9", "Tamper references", "tamper.build_references",
     "For every block actually used, store a SHA-256 hash and an 8-bin histogram in a sidecar JSON, "
     "for reference-based tamper localisation at recovery time."],
    ["10", "Output", "pipeline / webapp",
     "Return the stego image (colour recombined if needed) plus the sidecar JSON (profile, RS "
     "parameters and KDF recorded), and report PSNR, SSIM and blocks used."],
], [16, 74, 92, 300])

h2("2.2&nbsp;&nbsp;RECOVER path (Operator &rarr; &ldquo;Recover&rdquo;)")
p("The reverse of steps 8&rarr;5, driven entirely by the sidecar (no ML model needed at the "
  "receiver): recompute each block's hash and compare to the reference &rarr; a spatial "
  "<i>tamper map</i>; extract bits <i>with a soft reliability per bit</i> (the coefficient-pair "
  "margin); reverse the interleave; Reed-Solomon decode with tamper-flagged blocks as known erasures "
  "and <b>GMD reliability-guided decoding</b> (the least reliable bytes are tried as erasures, and "
  "the candidate with the smallest reliability-weighted distance wins, up to 2t errors instead of "
  "t); AES-256-GCM decrypt and verify the tag. Tag failure = &ldquo;tampered beyond recovery&rdquo;.")

h2("2.2b&nbsp;&nbsp;BLIND mode (no sidecar)")
p("A self-describing alternative: blocks are visited in a keyed pseudo-random order derived from the "
  "secret; a 14-byte header (profile, random salt, length) is RS-protected and stored three times with "
  "soft combining; the AIDE score sets each block's embedding <i>strength</i>, which the receiver does "
  "not need because extraction reads only the sign of each pair difference. The receiver needs only "
  "the image and the secret; a wrong secret finds no header at all.")

h2("2.3&nbsp;&nbsp;DETECT path (Analyst &rarr; &ldquo;Analyze&rdquo;)")
p("Completely independent of the sidecar. Takes only the image:")
bullets([
    "<b>Chi-square pairs-of-values test</b> (Westfeld &amp; Pfitzmann, 2001) &mdash; the primary "
    "signal. LSB embedding of an encrypted (near-random) payload equalises the counts of value "
    "pairs (2k, 2k+1); natural photos do not have this. Run on the whole image and per horizontal "
    "row-band.",
    "<b>LSB / next-bit-plane correlation</b> &mdash; a real photo's least-significant bit plane is "
    "weakly correlated with its finer structure; overwriting it with random payload collapses that "
    "correlation toward zero.",
    "<b>LSB bit-balance</b> &mdash; deviation of the LSB plane from a 50/50 split (weakest signal, "
    "low weight).",
    "<b>Sample Pairs Analysis + RS analysis</b> &mdash; quantitative estimators of the embedded "
    "fraction, validated against known embedding rates; they catch randomly scattered LSB payloads "
    "chi-square misses and report an estimated payload size.",
    "<b>Trained ML steganalyser</b> &mdash; SPAM + DCT coefficient-pair features, gradient boosting, "
    "with an IsolationForest guard that caps verdicts on images unlike its training covers.",
    "Combined into a verdict: <i>clean</i> / <i>suspicious</i> / <i>likely&nbsp;steganographic</i> "
    "(most severe of the statistical and ML verdicts).",
])

sp(6)

# ----------------------------------------------------------------------------
# 3. WHAT WE ARE DISCOVERING
# ----------------------------------------------------------------------------
h1("3.&nbsp;&nbsp;What we are discovering (measured findings)")

p("The project's method is &ldquo;measure, don't assume&rdquo;: every claim below is backed by a "
  "test on real BOSSbase images, and several of the original report's assumptions did <b>not</b> "
  "survive contact with measurement.")

h2("3.1&nbsp;&nbsp;Measured recovery by robustness profile")
p("Bit-exact, AES-GCM-verified recovery on 30 BOSSbase covers, 156-byte message "
  "(<font face='Courier'>aide benchmark</font>, reports/benchmark/benchmark.md):")
table([
    ["Attack", "quality", "balanced", "robust", "fortress"],
    ["JPEG Q50 (O4) / Q30", "100% / 20%", "100% / 100%", "100% / 100%", "100% / 100%"],
    ["Gaussian noise &sigma;=10 / &sigma;=20 (O4)", "0% / 0%", "100% / 0%", "100% / 100%", "100% / 100%"],
    ["Salt &amp; pepper 1%", "0%", "0%", "100%", "100%"],
    ["Median 3&times;3 / Gaussian blur", "52% / 100%", "100% / 100%", "100% / 100%", "100% / 100%"],
    ["Rescale 0.75&times; / 0.5&times;", "100% / 40%", "100% / 92%", "100% / 96%", "100% / 100%"],
    ["Splice 5% / Crop 10%", "100% / 84%", "100% / 88%", "100% / 100%", "100% / 100%"],
    ["Mean PSNR / SSIM", "43.4 dB / 0.984", "40.1 / 0.967", "35.0 / 0.912", "30.5 / 0.803"],
], [140, 85, 85, 85, 85])
p("For comparison, AIDE 0.1's fixed parameters on the same covers: &sigma;=20, salt &amp; pepper and "
  "0.5&times; rescale 0%, median 69%, splice 5% 85%, crop 10% 50% (PSNR 39.7 dB). Wrong secrets and "
  "tampered ciphertext are always rejected; colour covers keep full chroma fidelity.")

h2("3.2&nbsp;&nbsp;Finding: the DWT method as originally conceived did not work")
p("Single-level Haar DWT embedding in the horizontal/vertical <i>detail</i> sub-bands measured "
  "<b>~0% bit survival</b> against median filtering and geometric scaling &mdash; at <i>any</i> "
  "coefficient separation tested, up to delta = 400 (well past visible distortion). Both attacks "
  "resample the exact 2&times;2 neighbourhood a single-level Haar transform depends on. This directly "
  "contradicts the report's assumption that DWT gives &ldquo;superior robustness against geometric "
  "attacks&rdquo;. <b>Fix:</b> embed in the <i>coarse</i> 2-level approximation sub-band instead "
  "(a twice-averaged, low-frequency view of the block). Measured result: 100% survival against JPEG "
  "Q30/50, median filtering, and 2&times; downscale-then-restore, at delta = 40 and a <i>better</i> "
  "per-block PSNR (~37 dB vs ~29 dB) than the broken version needed.")

h2("3.3&nbsp;&nbsp;Finding: DCT has a hard capacity&ndash;versus&ndash;robustness trade-off")
p("The more DCT capacity a block is asked for, the further into higher, less-robust mid-frequencies "
  "it must reach. Measured on 16&times;16 blocks against a 3&times;3 median filter: 13 coefficient "
  "pairs &rarr; 20/20 blocks survive; 26 pairs &rarr; 10/20; 39+ pairs &rarr; 0/20 &mdash; and "
  "raising the coefficient separation does <b>not</b> help, because it is about <i>which</i> "
  "frequencies are used, not how far apart the pair is pushed. <b>Fix:</b> cap DCT to a "
  "measured-safe frequency band. Cost: less capacity per block, so more blocks are needed &mdash; "
  "which surfaces as an honest &ldquo;capacity insufficient&rdquo; error rather than a silent "
  "recovery failure.")

h2("3.4&nbsp;&nbsp;Finding: LSB is fragile to essentially everything")
p("Measured ~45&ndash;55% bit-error rate under every attack tried, including a plain re-save. LSB "
  "depends on exact pixel values. Its error rate is so much higher than DCT/DWT's that even 1% of "
  "the payload coming from LSB was measured to flip an otherwise-passing recovery into a failure. "
  "<b>Fix (0.2):</b> per-method attribution showed LSB and DWT causing ~40% of median-filter bit "
  "errors on ~2% of the bits, so every profile now routes the payload to DCT only; zeroing the "
  "budgets raised 10%-crop recovery from 52% to 88% at <i>better</i> PSNR. The engine still "
  "decides where, how much and which method; LSB/DWT blocks are left untouched.")

h2("3.5&nbsp;&nbsp;What 0.2 changed, and what still does not work")
bullets([
    "<b>Solved in 0.2:</b> &sigma;=20 noise and salt-and-pepper (robust profile), 0.5&times; rescaling "
    "(fortress 100%, balanced 92%), median filtering (100% from balanced up), single-codeword "
    "messages (shortened RS codewords), and the mandatory sidecar (blind mode).",
    "<b>Targeted detectability</b> &mdash; see 3.6: a trained detector catches AIDE. Fixing it needs "
    "side-informed embedding (e.g. syndrome-trellis codes); not attempted.",
    "<b>Geometric desynchronisation</b> &mdash; rotation and true (canvas-changing) cropping misalign "
    "the block grid; no registration template yet.",
    "<b>Capacity vs robustness</b> &mdash; messages near capacity lose the low-frequency layout "
    "preference and most 0.5&times;-rescale survival.",
    "<b>Scale</b> &mdash; trained on 334 BOSSbase images (not ALASKA2); CNN baselines (O7) and "
    "Raspberry Pi timings not measured.",
])

h2("3.6&nbsp;&nbsp;Finding: the detection side, and what it says about AIDE's own stealth")
table([
    ["Test", "Measured outcome"],
    ["SPA / RS accuracy vs ground truth", "True rate 0.25 &rarr; 0.27 / 0.28; 0.50 &rarr; 0.51 / 0.53; "
     "clean covers &asymp; 0.03"],
    ["Clean images falsely flagged (held out)", "1 in 40 (2.5%), thresholds calibrated on 200 disjoint covers"],
    ["Randomly scattered LSB at a 10% rate", "40 / 40 detected (chi-square alone: ~0)"],
    ["AIDE stego vs classical detectors", "0 of 15 flagged by chi-square, SPA or RS"],
    ["AIDE stego vs the trained ML detector", "100% detected for every profile and mode, AUC 0.998&ndash;1.000, "
     "1% false positives (100 held-out covers)"],
], [200, 290])
p("The honest conclusion: AIDE meets the report's &ldquo;negative detectability&rdquo; claim against "
  "classical blind tools only. Coefficient-pair QIM forces |c<sub>i</sub> &minus; c<sub>j</sub>| "
  "&ge; &delta;, leaving a deficit of small pair differences that a targeted detector learns "
  "immediately. A second finding came from the same detector: trained on photographs, it scored a "
  "clean synthetic pattern at P(stego)=0.99 (cover-source mismatch), which is why an "
  "out-of-distribution guard now caps such verdicts.")

h2("3.7&nbsp;&nbsp;A finding about method itself")
p("A false-positive regression in the detector was caught <i>by testing</i> before shipping: an "
  "early scoring change that folded a per-row-band breakdown into the verdict amplified 1/40 clean "
  "images being flagged into 16/20. It was reverted, and the row-band data kept as a labelled "
  "&ldquo;diagnostic only&rdquo; display. The general pattern &mdash; a plausible improvement that "
  "measurement rejects &mdash; recurred often enough to be the project's main working lesson. In 0.2 "
  "it recurred again: the first GMD decoder accepted the first successful decode and was measured to "
  "<i>mis-correct</i> on short codewords (all 13 failures of a 20-trial test were wrong-but-valid "
  "codewords, caught by the GCM tag); minimum-distance selection plus saturation-aware reliability "
  "took that test to 20/20.")

story.append(PageBreak())

# ----------------------------------------------------------------------------
# 4. SYSTEM FACTS
# ----------------------------------------------------------------------------
h1("4.&nbsp;&nbsp;System facts")

h2("4.1&nbsp;&nbsp;The trained decision engine")
p("Training is empirical, not circular: for each block, method labels come from the report's own "
  "texture/edge rule (with thresholds set from data percentiles), and the bit-rate label comes from "
  "actually embedding random bits at each level and <i>measuring</i> the resulting block PSNR. A "
  "DecisionTreeClassifier learns method; a DecisionTreeRegressor learns bit-rate; the classifier's "
  "normalised feature importances become the explainable weights (&alpha;,&beta;,&gamma;,&delta;). "
  "Measured learned weighting: texture &asymp; 0.55, edge &asymp; 0.44, entropy and variance &asymp; "
  "0.01 combined &mdash; i.e. the model relies almost entirely on texture and edge content.")

h2("4.2&nbsp;&nbsp;Cryptography &amp; coding parameters")
table([
    ["Component", "Parameter"],
    ["Cipher", "AES-256 in GCM mode (FIPS 197), 12-byte nonce, 16-byte tag"],
    ["Key derivation", "scrypt(shared secret, SHA-256(cover bytes)), N=2<super>14</super>, r=8, p=1; "
     "SHA-256 (report formula) for 0.1 files"],
    ["Error-correcting code", "Reed-Solomon, parity per 255 bytes: 64 / 64 / 104 / 176 by profile; "
     "shortened codewords for short messages; GMD reliability-guided decoding"],
    ["Interleaving", "Column-major over the codeword matrix &mdash; adjacent embedded bytes belong "
     "to different codewords"],
    ["DCT embedding", "Mid-frequency zigzag coefficient pairs, relative-magnitude QIM, separation "
     "&delta; = 20 / 36 / 64 / 80 by profile, preferring the 8 lowest robust pairs per block"],
    ["DWT embedding", "2-level Haar, coefficient pairs in the coarse approximation sub-band "
     "(available; payload budget 0 by default)"],
    ["Blind-mode header", "14 bytes (magic, version+profile, 8-byte salt, length), RS(40,18), 3&times; soft-combined"],
], [130, 360])

h2("4.3&nbsp;&nbsp;Attack simulation library")
p("Eight concrete simulations across the report's six categories: JPEG compression, Gaussian noise, "
  "salt-and-pepper noise, cropping (fixed-canvas region removal), splicing, median filter, Gaussian "
  "filter, and geometric scaling (downscale-then-restore). All operate on the luma plane and "
  "preserve image dimensions so block alignment for recovery stays valid.")

h2("4.4&nbsp;&nbsp;Technology stack")
p("Python 3, CPU only. NumPy, OpenCV, scikit-learn, scikit-image, SciPy, PyWavelets, PyCryptodome, "
  "reedsolo, Pillow for the core; Flask + gunicorn for the console and REST API; pytest for the "
  "146-test suite; ruff, Docker and GitHub Actions CI for engineering. No GPU, "
  "CUDA, or proprietary hardware anywhere &mdash; deployable on a Raspberry Pi 4 / Jetson-class "
  "board, as the report intends.")

sp(10)
story.append(Paragraph(
    "Generated from the running AIDE 0.2 system on 2026-10-11. Full source, tests, and the honest "
    "&ldquo;what works / what doesn't&rdquo; breakdown: github.com/yogishwar09/aide-steganography "
    "(see README.md).", styles["Caption"]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(20 * mm, 12 * mm,
                      "AIDE &mdash; Adaptive Intelligent Dynamic Embedding Framework  |  System Report"
                      .replace("&mdash;", "—"))
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm,
                        topMargin=18 * mm, bottomMargin=20 * mm,
                        title="AIDE System Report", author="P. Yogishwar")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
