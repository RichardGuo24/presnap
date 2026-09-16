# PreSnap — Build Plan

NFL win probability pipeline over ~15 seasons of play-by-play from `nflreadpy`
(Polars), stored in Postgres, evaluated on Brier score plus calibration curves
segmented by quarter and score margin.

The hard rule — **no feature may use information from after the play it
describes** — is the axis everything rotates around. This plan enforces that
rule with machinery, not diligence.

---

## Pre-phase contract (decided before any phase; lives in code as shared memory)

Several apparent "phase 3/4 decisions" are actually phase-1 decisions in
disguise (see the boundary critique at the end). Rather than let them leak
across boundaries, five decisions are hoisted into a **contract** that is
written first and imported by every later phase. This is the spine.

1. **Prediction grain** — one row = one pre-snap moment (one play). Join key =
   `(game_id, play_id)`.
2. **Label definition** — target is "did **`posteam`** win this game," derived
   from final score. This is a *post-play* value and that is fine — it is the
   label, not a feature. The perspective (posteam vs. home) is fixed here so it
   cannot silently flip later.
3. **Split policy** — temporal, by season, split on `game_id` (never on
   `play_id`). Plays within a game are correlated; a random play-level split
   leaks. A separate **calibration holdout** is reserved so phase 4's
   calibrator never sees train or test.
4. **Segmentation carry-through columns** — `qtr` and `score_differential`
   (binned) must survive untransformed all the way to evaluation, or phase 4
   cannot segment. Declared now.
5. **Pre-snap allowlist** — the frozenset of raw columns a feature is allowed
   to read. This *is* the hard rule, encoded.

These live in `presnap/contract.py`. Everything downstream references it.

---

## Phase 1 — Ingestion + schema

**Files**
- `presnap/contract.py` — the five decisions above; `PRESNAP_COLUMNS`,
  `SEGMENT_COLUMNS`, `LABEL_COLUMNS` frozensets.
- `presnap/db.py` — Postgres connection, DDL runner.
- `presnap/schema.sql` — raw `pbp` table DDL + a **`presnap_play` VIEW** that
  selects *only* allowlisted columns.
- `presnap/ingest.py` — `nflreadpy.load_pbp(seasons)` → Polars → Postgres,
  idempotent upsert on `(game_id, play_id)`.
- `tests/test_ingest.py` — row counts vs. source, non-null keys, all requested
  seasons present.

**Done means (verifiable):**
`python -m presnap.ingest --seasons 2010:2024` populates Postgres; a check
script prints the Postgres row count and it equals `nflreadpy`'s row count for
those seasons; `psql -c "SELECT * FROM presnap_play LIMIT 1"` succeeds and
`wp`/`vegas_wp`/`epa` are **absent** from it.

**How this phase silently produces wrong results:**
- **Non-idempotent re-runs** double rows; counts look "bigger, fine."
- **Schema drift across 15 seasons** — a column absent in older seasons gets
  coerced to null; features compute cleanly but are empty pre-2015 unnoticed.
- **Pre- vs. post-play score columns.** In nflfastR-derived data,
  `score_differential`/`posteam_score` are *start-of-play* while `*_post`
  variants are after. Verify this **empirically, not by assumption** — getting
  it backwards is a textbook silent leak. Phase-1 checklist item, not a claim.
- **Type coercion** Polars→Postgres silently truncating (e.g., `down` null on
  kickoffs vs. 0).

---

## Phase 2 — Feature construction  🚧 *stronger gate: walkthrough + wait before writing code*

**Mechanical no-lookahead enforcement (the thesis — three independent layers):**

1. **Logical** — `PRESNAP_COLUMNS` allowlist in `contract.py` is the single
   source of truth.
2. **Physical** — feature code connects only to the `presnap_play` **view**.
   Post-play columns are not hidden, they are *absent*; referencing `epa` is a
   SQL error, not a wrong number. `wp`/`vegas_wp` (the dataset's own answer) are
   excluded — using them would leak the target.
3. **Behavioral (fails loudly, cannot be forgotten)** — `tests/test_no_lookahead.py`:
   take real pbp, **null-and-randomize every column *not* in the allowlist**,
   run the feature pipeline on pristine vs. corrupted input, assert the two
   feature tables are **byte-identical**. If any feature secretly touched a
   post-play column, outputs diverge and CI goes red. Catches leakage even if
   someone bypasses the view.

The label is built by a separately-named `labels.py` that is *explicitly*
permitted to read final-score columns — keeping the "allowed to see the future"
surface to exactly one auditable file.

**Files**
- `presnap/features.py` — pure functions: presnap row → feature vector.
- `presnap/labels.py` — target (the only future-allowed module).
- `presnap/build_features.py` — read view, compute, write `features` table.
- `tests/test_no_lookahead.py` — the perturbation test.
- `tests/test_features.py` — known-play spot checks.

**Done means:** `python -m presnap.build_features` writes the feature table;
`pytest tests/test_no_lookahead.py` passes; a hand-verified play's features
match by hand.

**Silent-wrong risks:** a feature that is pre-snap "in spirit" but computed over
a window including the current play's outcome; null silently imputed to 0,
changing meaning (yardline 0 = goal line, timeouts 0 = none left); the label
accidentally merged into the feature frame and picked up.

---

## Phase 3 — Model training

**Files**
- `presnap/split.py` — temporal split by season, partitioned on `game_id`,
  emitting train/val/**cal**/test id sets.
- `presnap/model.py` — logistic-regression baseline, then gradient boosting.
- `presnap/train.py` — fit, print val Brier, save artifact.
- `tests/test_split.py` — assert **no `game_id` in two splits**, temporal
  ordering holds, test ids untouched.

**Done means:** `python -m presnap.train` saves a model and prints val Brier;
`pytest tests/test_split.py` passes.

**Silent-wrong risks:** game in both train and test (leakage); **fitting
scaler/imputer/calibrator on full data before splitting**; using test for early
stopping; **flipped label perspective** (posteam vs. home) — the model learns
the inverse and Brier stays deceptively mediocre rather than obviously broken.

---

## Phase 4 — Evaluation + calibration  🚧 *stronger gate: walkthrough + wait before writing code*

**Files**
- `presnap/calibrate.py` — Platt/isotonic, fit **only on the calibration
  holdout**.
- `presnap/evaluate.py` — Brier overall + segmented by `qtr` × margin bin.
- `presnap/plots.py` — reliability diagrams.
- `tests/test_evaluate.py` — Brier of constant-0.5 predictor equals the
  closed-form value; **segment bins cover every row** (sum of segment counts
  == N).

**Done means:** `python -m presnap.evaluate` prints Brier per segment and a
base-rate baseline for comparison, and writes reliability plots.

**Silent-wrong risks:** calibrating on training data (looks perfect, means
nothing); **segment bins that drop rows** (margin on a boundary, OT / `qtr=5`)
so the aggregate looks artificially good; Brier computed on the raw vs.
calibrated probability column; unweighted segment averaging producing a
misleading headline number.

---

## Phase 5 — Containerization + CI

**Files**
- `Dockerfile`, `docker-compose.yml` (app + Postgres), `.dockerignore`
- `pyproject.toml` / pinned `requirements.txt`
- `.github/workflows/ci.yml`, `Makefile`

**Done means:** `docker compose up` brings up Postgres and runs a smoke pipeline
on a 1-season subset end-to-end; CI is green on push.

**Silent-wrong risks:** **CI passing because tests silently no-op** without a DB
(skips read as passes); unpinned deps so the container's results differ from
dev; **the no-lookahead test not wired into CI**, so the core guarantee is not
actually enforced on every change.

---

## Phase 6 — Streamlit frontend *(optional, later — after the engine works)*

Not part of the core deliverable; added because the pipeline's real output is a
`predict(pre_snap_state) -> probability` function, and a frontend is just a face
over it. Every screen is the SAME model asked a different question — the
frontend contains no ML smarts of its own. Build with Streamlit (Python, fast,
learn-the-wiring) first; a React rebuild via the `frontend-design` skill is a
later polish step. Load the `dataviz` skill before writing any chart.

**Committed screens**
- **Game WP curve** — pick a `game_id`, draw win probability vs. game time,
  scrub to any moment for the number, show the actual winner for eyeballing.
- **Rankings** — games sorted by total win-probability swing (the historic
  collapses and comebacks).

**Buffet (pick as desired, each is cheap — one query / a few model calls)**
- Clutch plays (single plays that swung WP most)
- Comeback finder (lowest WP a team recovered from to win)
- Situation explorer (down/distance/score/clock sliders → live WP)
- 4th-down decision tool (go for it / punt / kick + WP of each)
- Model vs. Vegas (where our WP disagrees with the betting line)
- "Why this number?" (which factors drove a given prediction)

**Files (when built)**
- `presnap/inference.py` — the single `predict()` entry point every screen calls.
- `app/` — Streamlit pages (`Home.py`, `pages/1_Game.py`, `pages/2_Rankings.py`).

**Done means:** `streamlit run app/Home.py` opens a browser; picking a game
draws its WP curve; the rankings page lists games by swing.

**Silent-wrong risks:** **train/serve skew** — the frontend must compute features
via the exact same code path as training, or the displayed curve is subtly wrong
while looking plausible. `inference.py` must reuse `features.py`, never
reimplement it.

---

## Second-pass boundary critique (where the five phases are wrong)

The clean 1→5 boundary is a lie in three places; each is patched via the
contract:

1. **The split (phase 3) is really a phase-1 decision.** Splitting on `game_id`
   requires `game_id`/`season` to be first-class from ingestion, and the
   calibration holdout that phase 4 needs must be carved by the *same* split. →
   contract #3.
2. **Segmentation (phase 4) constrains the feature schema (phase 2).** If phase
   2 scales/drops `qtr` and `score_differential`, phase 4 cannot segment. →
   contract #4.
3. **The label perspective (phase 3/4) is a phase-1 decision.** posteam-vs-home
   determines table grain and the target column; deciding it late is how labels
   silently flip. → contract #2.

Net change: **a pre-phase contract file** absorbs the three cross-boundary
dependencies, so each phase is then a genuine hard stop.

---

## Execution strategy (subagent vs. main thread), model tier, /clear

| Phase | Subagent (tight spec, verifiable) | Main thread (reasoning *is* the artifact) | Model | /clear |
|---|---|---|---|---|
| 1 | ETL code, `test_ingest`, DDL | **the allowlist + label grain** | cheap (contract decisions brief-but-expensive) | `/clear` after |
| 2 | individual feature impls + unit tests | **no-lookahead mechanism + the perturbation test** | **expensive** | keep contract file; `/clear` after |
| 3 | training boilerplate, artifact I/O | **split/leakage logic** | mixed | `/clear` after |
| 4 | plot code | **calibration + segmentation** (interview-facing) | **expensive** | `/clear` before |
| 5 | Dockerfile, CI yaml, compose | what the CI smoke test asserts | cheap | — |

**Two places the default line is drawn wrong:**
- **The phase-1 allowlist should *not* go to a subagent**, even though phase 1
  is otherwise the most delegatable phase. It is reasoning-as-artifact — the
  hard rule encoded — so it stays main-thread.
- **Do not delegate the perturbation test**, even though it has a crisp
  verifiable output. Its *correctness* is subtle — "does identical output
  actually prove safety, and did I perturb the right column set?" — and that
  judgment is what you want to own. Delegate the feature unit tests and the
  Dockerfile; those are genuinely tight-spec.

**/clear rule of thumb:** clear whenever the next phase's reasoning does not need
the prior phase's exploration — safe *because* the contract file externalizes
the shared decisions. The files are the memory; the context is disposable.

---

## Stopping protocol (every phase)

After each phase, stop and do not start the next. Hand over: (a) a diff summary,
(b) the exact command to verify, (c) 2–3 questions with what a weak answer looks
like. Sample for phase 2:

- *"Why does nulling post-play columns and asserting identical output prove no
  leakage?"* — weak: "because the test passes." Strong: the test makes leakage
  *change observable output*, so absence of change is evidence of absence of
  dependence.
- *"Is `score_differential` pre- or post-play, and how do you know?"* — weak:
  "it's the score." Strong: cites the empirical check from phase 1.

Phases 2 and 4 additionally **begin with a walkthrough and wait for a response
before any code.**
