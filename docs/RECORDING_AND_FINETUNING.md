# Recording and fine-tuning guide

How to collect data with VisionICU, first in a developer test room and then in an ICU/IMCU,
and how that data becomes validated (and, where it makes sense, fine-tuned) models.

> **Read section 1 first.** Every session records numbers once a second, which is enough to
> **validate** every module. Tick **Capture training data** when starting a session to also
> record the per-frame inputs the breathing CNNs and the fall model learn from. Face video is
> never recorded, so EfficientPhys and the pain model cannot be fine-tuned with this yet.

---

## 1. What the website records

| | Every session | With **Capture training data** | Needed to fine-tune |
|---|---|---|---|
| Fused RR, HR, early-warning score, pain probability (1 Hz) | ✅ | ✅ | — |
| Every estimator's value, quality and raw network output (1 Hz) | ✅ | ✅ | — |
| Reference values (*Log reference*, or imported) | ✅ | ✅ | ✅ RR labels |
| Marked events (fall, chair exit, talking, CPOT/BPS score, …) | ✅ | ✅ | ✅ fall labels |
| Chest / abdomen depth means, every frame | ❌ | ✅ | ✅ RR CNN (D435) |
| Whole-frame mean green, every frame | ❌ | ✅ | ✅ RR CNN (LBRD-IC) |
| Pose keypoints + box, every frame | ❌ | ✅ | ✅ fall GRU |
| Face crops | ❌ never | ❌ never | EfficientPhys, pain model (not supported) |
| Continuous PPG / respiration waveform | via `scripts/import_reference.py` | same | beat/breath-level labels (future) |

Files per session (`data/sessions/<bed>_<time>_<label>/`, git-ignored):

| File | Contents |
|---|---|
| `meta.json` | label, condition, notes, times, fusion setting, `capture`, frame and event counts |
| `estimates.csv` | 1 Hz numbers per estimator |
| `references.csv` | references typed during the session, plus imported ones (`import:<source>` in the note); the original is kept as `references.manual.csv` after an import |
| `events.csv` | `t, kind, value, note` for every *Mark an event* press |
| `frames.csv` | capture only: one row per frame, columns in `src/serving/session_log.py → FRAME_COLUMNS` (~30 rows/s, about 40 MB per hour) |
| `ppg.csv` / `resp.csv` | an imported reference waveform |

Score sessions on the **Compare** page or with
`python scripts/compare_sessions.py --by condition --out reports/compare.md`.

---

## 2. What each module learns from, and what "fine-tuning" means for it

| Module | Has weights? | How to improve it for your setup | Input it needs | Identifiable data? | Label / reference |
|---|---|---|---|---|---|
| RR, depth (DSP) | No | Tune `config/signal.yaml` (band, high-pass, segment, gates) against sessions | — (sessions are enough) | No | Metronome / counted breaths / capnography |
| RR, thermal (DSP) | No | Same; align the thermal camera once per mount | — | No | Same |
| HR, POS (DSP) | No | Lighting and distance matter more than parameters (README → *Getting a strong signal*) | — | No | Pulse oximeter / ECG HR |
| RR CNN, D435 | Yes (`models/finetuned/rr_cnn1d_d435_finetuned.pth`) | Fine-tune on 10 s chest-depth windows → rate | Chest depth mean, 30 Hz | **No** (numbers) | Breath rate per window |
| RR CNN, LBRD-IC | Yes (public) | Fine-tune on 10 s green-channel windows → rate (new file in `models/finetuned/`) | Whole-frame mean green, 25 Hz | **No** (numbers) | Breath rate per window |
| EfficientPhys (HR) | Yes (public) | Fine-tune on face-frame differences → pulse waveform | 72×72 face crops, 30 Hz | **Yes (face video)** | Continuous PPG waveform, time-synced |
| Fall GRU | Yes (public) | Fine-tune on 32-frame keypoint windows → fall / no fall | Pose keypoints, 30 Hz | Low (skeleton only) | Event labels (start/end of each fall, exit, ADL) |
| Pain (MobileNetV3) | Yes (public, synthetic) | Retrain on real faces with clinical pain scores | Face crops | **Yes (faces)** | CPOT or BPS score by trained nurses |
| TAA | No | Tune threshold (1.2 rad) and sustain (10 s) against labelled episodes | — | No | Respiratory belts (RIP) or clinician-labelled episodes |

Public weights stay untouched; every fine-tuned file goes in `models/finetuned/` and is
switched on by pointing its path in `config/signal.yaml` at it.

---

## 3. Before any human recording: approvals (do not skip)

Not legal advice. Confirm every point with your institution.

- [ ] **Institutional Ethics Committee (IEC) approval** for the study protocol (ICMR *National
      Ethical Guidelines for Biomedical and Health Research*, 2017). One protocol can cover the
      developer-room volunteer study; the ICU study normally needs its own approval at the hospital.
- [ ] **Clinical Trials Registry – India (CTRI)** registration if the IEC classes the ICU study as
      clinical research.
- [ ] **Informed consent** forms in English and the local language. ICU patients who cannot
      consent: consent from the **legally authorised representative (LAR)**, and re-consent
      from the patient when they regain capacity.
- [ ] **Data protection** under the Digital Personal Data Protection Act, 2023: purpose stated in
      the consent, pseudonymous subject codes (S01, P01…), the code-to-name key kept on paper or a
      separate encrypted drive, data on an encrypted disk, never in git or a public cloud,
      a retention/deletion date.
- [ ] **Hospital permission** (medical superintendent / ICU head) to mount a camera, and a named
      clinician co-investigator in the ICU.
- [ ] A clear statement to staff that VisionICU is a **research prototype**: no treatment decision
      may use it.

Numbers-only sessions (section 1) contain no images. Face-video capture for EfficientPhys and
pain is identifiable data and needs explicit consent for video.

---

## 4. Equipment checklist

**Sensing**
- [ ] Intel RealSense D435 + USB 3 cable (≤ 2 m, or an active extension), rigid mount or tripod
- [ ] Waveshare ESP32-S3 thermal camera + 5 V supply (optional; enables the thermal RR path)
- [ ] Laptop or Jetson running `python main.py`; a second screen for the nurse view is useful

**References (ground truth)**
- [ ] Fingertip **pulse oximeter** with a readable HR (spot values for validation)
- [ ] For EfficientPhys fine-tuning: a **logging PPG sensor** (e.g. a pulse oximeter with PC data
      export, or a MAX30102 board on an ESP32) recording the waveform at ≥ 30 Hz with timestamps
- [ ] **Metronome app** (phone) for paced breathing
- [ ] For TAA: a pair of **respiratory inductance plethysmography (RIP) belts** (chest + abdomen),
      if the department has them; otherwise clinician-labelled episodes
- [ ] In the ICU: the **bedside monitor** (ECG HR, impedance or capnography RR). Note its values
      once a minute, or export them if the monitor allows it
- [ ] One clock source: set the laptop, phone and any logger to network time before each session

**Environment**
- [ ] Measuring tape and an angle app (camera distance and tilt)
- [ ] A lamp with a diffuser (or a white wall to bounce light off), placed behind the camera
- [ ] For fall recordings (developer room only): crash mats, a spotter, helmet/knee pads offered

---

## 5. Setting up the space

### 5a. Developer test room (chair)

1. **Chair**: a stable armchair, against a plain wall, nothing hanging behind the subject's head.
   No window behind the subject (backlight makes the face dark).
2. **Camera**: in front of the chair, **1.5–2.0 m** away (config assumes 1.6 m), lens at
   **chest-to-face height**, tilted **10–30° down** (config assumes 15°). The face should be at
   least ~120 px wide in the colour image and the whole torso (shoulders to hips) in view.
3. **Light**: diffuse light falling on the face from behind the camera. Avoid flickering LED strips.
   Then start the monitor: the camera meters the room at startup and locks exposure, so set the
   lights first.
4. **Thermal camera** (optional): next to the D435, same direction, about 1 m from the face.
5. **Website**: open the bed → **Set up chair** → click the four seat corners → *Start monitoring*.
   If thermal is on: **Align thermal** → hot mug at 4–6 spots (5 or more lets it measure its own
   accuracy).
6. Check on the bed page: face blurred, skeleton on the subject, signal bars above 40 % after
   ~30 s of sitting still. If not, fix light or distance before recording.

### 5b. ICU / IMCU (bed or chair)

1. Mount at the **foot of the bed or the side**, 1.5–2.0 m from the chest, high enough to look
   over bed rails and IV poles, tilted down so face, chest and abdomen are visible.
2. Bed head raised 30–45° is ideal. A blanket over the abdomen is fine; over the chest reduces
   breathing signal.
3. Nothing hanging between camera and patient (IV lines, curtains); agree with nursing staff where
   the camera goes so it does not block care.
4. **Set up chair** with *Bed* selected and the four mattress corners. Redo it whenever the bed
   is moved.
5. Use `--reuse-calibration` only when neither camera nor bed has moved.

---

## 6. The training tools

| Tool | What it does |
|---|---|
| **Capture training data** (session start form) | Writes `frames.csv` as described in section 1. Off by default. |
| **Mark an event** (shown while a session runs) | Fall, chair exit, sit-to-stand, lying down, talking, movement, breath-hold start/end, paradoxical breathing start/end, procedure start/end, pain score (CPOT 0–8 or BPS 3–12, the scale is stored at the start of the note), other. Press it **as it happens**, e.g. *Fall* as the person lands. |
| `scripts/import_reference.py` | Adds a pulse-oximeter or bedside-monitor export (time + hr/rr/spo2) to `references.csv`, or saves a PPG/respiration waveform as `ppg.csv` / `resp.csv`. `--offset-s` corrects the device clock, `--date` reads clock-only times. |
| `training/dataset.py` | Builds windows exactly like the live modules: 10 s chest depth → 300 samples, detrend, z-score (D435 CNN); 10 s green → 250 samples, z-score (LBRD-IC CNN); 32-frame 62-D pose features replayed through the live fall feature extractor. Skips the first 30 s and 20 s after each reference change, and windows with gaps. |
| `training/finetune.py` | Subject-wise train/val/test split (written to `data/splits/<module>_splits.json`, checked by `tests/test_leakage.py`), fine-tunes from the current weights with augmentation and early stopping, scores before/after on the held-out people, saves `models/finetuned/<module>_<date>.pth` + a `.json` report. Refuses with fewer than 3 people. |
| `training/eval.py` | Scores any weights on held-out people; for breathing also scores the signal-processing reading (`rr.depth`, fused `rr`) over the same seconds with MAE **and** coverage. |

Subject codes come from the session label: everything before the first underscore
(`S07_cabin_paced15_bright` → `S07`). Use the same code for every session of one person.

---

## 7. Who to record (subjects)

These are planning targets, not statistical guarantees. Use your IEC protocol's numbers.

| Dataset | People | Per person | Spread to cover |
|---|---|---|---|
| Developer validation (numbers only) | 10–15 healthy volunteers | 30–40 min (protocol in 8a) | Skin tone (Fitzpatrick III–VI matter most in India), sex, age 18–70, beards, glasses, BMI |
| RR CNN fine-tune (D435 and LBRD-IC) | ≥ 20 | ≥ 20 min of breathing across rates 6–30 /min | Same spread; sitting upright and reclined |
| EfficientPhys fine-tune | ≥ 30 (UBFC-rPPG had 42) | ≥ 5 min still + 5 min after light exercise, PPG waveform logged | Skin tone and lighting above all |
| Fall GRU fine-tune | ≥ 20 actors (developer room only) | ≥ 10 falls + ≥ 20 non-fall activities each | Fall directions; look-alikes (picking things up, sitting down hard) |
| Pain model | ≥ 50 ICU patients (observational only) | Every routine painful procedure + rest, CPOT/BPS by 2 nurses | Intubated and not; age; skin tone |
| ICU validation | ≥ 20–30 patients | 1–2 h each, bedside monitor as reference | Ventilated and spontaneous breathing; bed and chair |

Never put the same person in both training and testing.

---

## 8. Recording protocols (on the website)

General rules for every session:
- **Record & compare** page → choose the bed → *Condition*, *Label*, reference RR/HR, notes.
  **Label format:** `S07_cabin_paced15_bright` or `P03_icu_rest`. **No names** in label or notes.
- The first 30 s and 20 s after each reference change are not scored, so hold each block
  ≥ 2 min.

**Naming convention** (label = `<subject>_<site>_<block>_<light>`, optional `_r2` for a repeat):

| Part | Values | Notes |
|---|---|---|
| subject | `S01`–`S99` volunteers, `P01`–`P99` patients | Letters then digits, then `_`. Training reads the subject from this, so one person keeps one code for life |
| site | `cabin`, `dev`, `icu`, `imcu` | Where it was recorded |
| block | `rest`, `paced06`, `paced10`, `paced15`, `paced20`, `paced30`, `hold`, `talk`, `exercise`, `recline`, `taa`, `thermoff`, `fall` | Two-digit paced rates so files sort in order |
| light | `bright`, `dim` | Light at the time the monitor was started |

Examples: `S01_cabin_paced15_bright`, `S03_cabin_rest_dim`, `S05_cabin_fall_bright_r2`.
Keep the code-to-name list on paper only; names never go in labels or notes.
- Press **Log reference** every time the truth changes (new metronome rate, new oximeter value).
- One session per block keeps the Compare page readable.

### 8a. Developer-room protocol per volunteer (~35 min)

| # | Block | Condition (website) | Reference to enter | Time |
|---|---|---|---|---|
| 1 | Sit still, normal breathing | Rest, normal breathing | Counted breaths/min; pulse-ox HR every 30 s | 3 min |
| 2–6 | Paced breathing at 6, 10, 15, 20, 30 /min | Paced breathing (metronome) | The metronome rate; pulse-ox HR every 60 s | 2 min each |
| 7 | Breath-hold 15–20 s, three times | Breath-hold | Note each hold start/end | 2 min |
| 8 | Talking / small movements | Talking or moving | Counted breaths, pulse-ox HR | 2 min |
| 9 | Dim light (room lights off, lamp only) | Dim light | As 1 | 3 min |
| 10 | After 2 min of stair stepping | After exercise | Pulse-ox HR every 15 s (it falls fast) | 3 min |
| 11 | Reclined 30–45° | Other (note "reclined") | As 1 | 3 min |
| 12 | Voluntary paradoxical breathing (belly in while chest rises), only if comfortable | Other (note "TAA") | RIP belts if available, else note start/end | 2 min |
| 13 | Thermal camera on/off repeat of block 1 | Rest, normal breathing | As 1 | 3 min |

Fall block (separate session, developer room, mats + spotter, healthy actors only):
forward, sideways and backward slides from the chair to the mat, chair exit, sit-to-stand,
bending to pick something up, sitting down hard. Tick **Capture training data** and press
**Fall**, **Chair exit**, **Sit-to-stand** etc. under *Mark an event* the moment each happens.
For breathing sessions you intend to train on, tick **Capture training data** too.

### 8b. ICU / IMCU protocol per patient (with consent)

| # | Block | Reference | Time |
|---|---|---|---|
| 1 | Rest, patient settled, no procedures | Bedside monitor RR and HR noted every minute | 30–60 min |
| 2 | Routine care (turning, suctioning, physiotherapy) | Note start/end; CPOT/BPS before, during, after | As it happens |
| 3 | Sitting out in a chair (if the patient does) | As 1; recalibrate as *Chair* | 20–30 min |
| 4 | Night (lights dimmed) | As 1 | 30 min |

Do **not** stage anything with patients: no paced breathing, no breath-holds, no falls.

---

## 9. From recordings to models

**A. Validate (possible today):**
1. Record sessions (section 8) with the current models.
2. **Compare** page or `scripts/compare_sessions.py --by condition`: read **MAE together with
   coverage**, bias, % within ±2 br/min / ±5 bpm, Pearson r.
3. Tune DSP settings in `config/signal.yaml`, re-record on **new** volunteers, compare again.

**B. Fine-tune (breathing CNNs and the fall model):**
1. Record sessions with **Capture training data** on, references logged (or imported) and
   events marked. Copy the session folders of every site into one `data/sessions` folder.
2. Import device references if you have them:
   ```bash
   python scripts/import_reference.py data/sessions/<id> pulseox.csv --source pulse_ox --offset-s 0
   ```
3. Fine-tune (one module at a time):
   ```bash
   python training/finetune.py --module rr_d435 --sessions data/sessions
   python training/finetune.py --module rr_lbrd --sessions data/sessions
   python training/finetune.py --module fall    --sessions data/sessions
   ```
   Read the printed `test before` / `test after` lines: they score people the model never saw.
4. Compare with what the monitor already does:
   ```bash
   python training/eval.py --module rr_d435 --weights models/finetuned/rr_d435_<date>.pth
   ```
5. If it is better, point `config/signal.yaml` at the new file (`learned.rr_d435.weights`,
   `learned.rr_lbrd.weights` or `fall.gru_weights_primary`), restart, and record validation
   sessions on **new** people.
6. Fuse a learned breathing model (`learned.fuse` list) only if it beats the signal-processing
   path on MAE **and** coverage across conditions. Fused learned paths stay secondary.

EfficientPhys and the pain model need face video, which the recorder deliberately never stores.
Supporting them needs a separate, consented and encrypted capture path.

---

## 10. Printable session checklist

**Before**
- [ ] Consent signed (video consent ticked if capturing faces); subject code assigned
- [ ] Clocks synced; pulse oximeter charged; metronome ready
- [ ] Camera distance 1.5–2.0 m, tilt 10–30°, face ≥ 120 px wide, torso in view
- [ ] Lights set **before** starting `python main.py` (exposure locks at startup)
- [ ] Chair/bed calibrated; thermal aligned (if used); signal bars > 40 % at rest
- [ ] Alarm sound on; staff told it is a research prototype

**During**
- [ ] One session per block; label `Sxx_site_block_light`
- [ ] **Capture training data** ticked if the session is for fine-tuning
- [ ] **Mark an event** pressed the moment each fall / exit / talking / procedure happens
- [ ] *Log reference* at every change; notes for events (no names)
- [ ] Watch for "Patient moving" / "Face too dark" and fix the cause if it persists

**After**
- [ ] Stop the session; check row count on the Compare page
- [ ] Copy `data/sessions/<id>/` to the encrypted study drive; record the subject code on the
      paper log
- [ ] Note anything unusual (lighting change, camera bumped) in the study log
