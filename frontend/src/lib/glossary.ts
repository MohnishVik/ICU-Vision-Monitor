// glossary.ts — the one wording for every term on the monitor, in plain clinical language.
//
// Each entry says what a thing is, how the camera gets it, and how to read it (with the normal
// range where there is one). Tooltips and InfoTips read from here, so a term means the same
// thing on every page. Every number below is taken from the code that produces it:
// config/ews_rules.yaml, config/signal.yaml, src/serving/state.py, src/fusion/*,
// src/evaluation/compare.py. Change the code, change the sentence.

export interface Gloss { title: string; body: string }

const g = <T extends Record<string, Gloss>>(entries: T) => entries;

export const GLOSSARY = g({
  // ------------------------------------------------------------------ the product
  prototype: {
    title: 'Research prototype',
    body: 'VisionICU is a research system, not a certified medical device. Its numbers have not been '
      + 'validated for clinical use, so they must not guide treatment. Keep using the bedside monitor '
      + 'and your own observations; use this to see how camera-based monitoring behaves.',
  },
  demo: {
    title: 'Demo, simulated data',
    body: 'The server was started with --demo: no camera is used and the patients are simulated. The '
      + 'simulated readings still pass through the real fusion, early-warning and alarm code, so the '
      + 'screens behave as they would with a camera.',
  },
  live: {
    title: 'Live link to the server',
    body: '"Live" means an update arrived from the server within the last 3 s; the ward refreshes about '
      + 'once a second. "No update for N s" means the connection has stalled: the numbers on screen may '
      + 'be out of date until it reconnects, which it does by itself.',
  },
  clock: {
    title: 'Local time',
    body: 'This computer’s clock, 24-hour format. Alarm and session times use the same clock.',
  },
  sound: {
    title: 'Alarm sound',
    body: 'Browsers keep a page silent until you allow sound, so alarm tones start off. When on, an '
      + 'alarm that reaches the chime stage plays a tone in this browser: high priority is two bursts '
      + 'of five rising tones, medium priority three tones.',
  },
  d435: {
    title: 'RealSense D435 camera',
    body: 'Colour and depth camera at 30 frames a second. Colour gives heart rate from the skin; '
      + 'depth gives breathing from chest movement, and posture for falls.',
  },
  navHome: { title: 'Home', body: 'What VisionICU measures, how it works and how to run it.' },
  navWard: { title: 'Ward', body: 'Every bed at a glance, with the one that needs you first named at the top.' },
  navCompare: {
    title: 'Record & compare',
    body: 'Record a session against reference values you trust, then see how each method scored.',
  },
  navSystem: { title: 'System', body: 'Server health, model files and which parts of the pipeline are running.' },

  // ------------------------------------------------------------------ vitals
  hr: {
    title: 'Heart rate (bpm)',
    body: 'Beats per minute, read without contact from the face. The colour camera tracks the tiny '
      + 'change in skin colour with each heartbeat (rPPG, POS method) over a 20-second window, '
      + 'updated every second. The early-warning score treats 51–90 bpm as normal (0 points); outside '
      + 'that band it adds points.',
  },
  rr: {
    title: 'Respiratory rate (br/min)',
    body: 'Breaths per minute. The depth camera measures the chest rising and falling over a 30-second '
      + 'window; when the thermal camera is aligned it also counts the warm air breathed out at the '
      + 'nostrils. Normal adult range 12–20 br/min (0 points); outside it adds to the early-warning score.',
  },
  sqi: {
    title: 'Signal quality',
    body: 'How clearly the breathing or pulse rhythm stands out from noise, 0–100 %, drawn as five bars. '
      + 'Below 40 % the bars turn orange: the number is less certain. "None" means there is no current '
      + 'reading to rate.',
  },
  sqiFace: {
    title: 'Face quality',
    body: 'How usable the face is for the pain model: rises with face size in the image and with how many '
      + 'recent frames it could score. Below 40 % the bars turn orange.',
  },
  stale: {
    title: 'No fresh value',
    body: 'If no new value arrives for more than 5 s, the last number is withdrawn instead of being left '
      + 'on screen, and the time since the last good reading is shown. It never counts as a normal reading.',
  },
  cannotMeasure: {
    title: 'Cannot measure right now',
    body: 'A dash on a hatched background means the monitor has no trustworthy number, so it shows none '
      + 'rather than guess. The words beside it say why. A missing vital is never scored as normal.',
  },
  reason_camera_offline: {
    title: 'Camera offline',
    body: 'The camera is not delivering frames. The server retries every 5 s; check the USB cable and power.',
  },
  reason_awaiting_calibration: {
    title: 'Chair not set up',
    body: 'Monitoring starts once the four corners of the chair or bed have been marked (Set up chair).',
  },
  reason_starting: {
    title: 'Starting up',
    body: 'The server is opening the camera and loading the models. This usually takes a few seconds.',
  },
  reason_warming_up: {
    title: 'Collecting signal',
    body: 'Each method needs a full window of signal before it gives a number: about 20 s for breathing '
      + 'and 10 s for heart rate after monitoring starts or the patient comes into view.',
  },
  reason_paths_disagree: {
    title: 'Sensors disagree',
    body: 'Two signal-processing methods differ by more than 4 br/min (breathing) or 10 bpm (heart rate). '
      + 'Averaging them could produce a plausible but wrong number, so none is shown until they agree.',
  },
  reason_no_valid_input: {
    title: 'Signal too weak',
    body: 'Every method’s signal quality is below its minimum, for example when the face is turned away, '
      + 'the chest is covered or still, or the patient is moving.',
  },
  reason_motion: {
    title: 'Patient moving',
    body: 'The patient moved, so the analysis window restarted and is refilling. A number returns once '
      + 'they are still again, typically after 10–20 s.',
  },
  reason_low_light: {
    title: 'Face too dark',
    body: 'The face is too dark for a reliable heart-rate reading. Add diffuse light facing the patient; '
      + 'breathing from the depth camera is not affected by light.',
  },
  reason_stale: {
    title: 'No recent reading',
    body: 'No new value for more than 5 s, so the last one has been withdrawn.',
  },
  reason_waiting: { title: 'Waiting for data', body: 'No reading has arrived for this bed yet.' },
  apnoea: {
    title: 'Possible apnoea',
    body: 'Chest movement seen by the depth camera has dropped below 25 % of its recent size for at '
      + 'least 10 s. This raises a high-priority alarm and blanks the breathing rate, so a pause in '
      + 'breathing can never be shown as a normal rate.',
  },
  lastReading: {
    title: 'Time since last reading',
    body: 'How long ago this vital last had a valid value. After 5 s without one it is withdrawn.',
  },
  sparkline: {
    title: 'Last 30 minutes',
    body: 'The vital over the last 30 min. The shaded band is the range that scores 0 points (heart rate '
      + '51–90 bpm, breathing 12–20 br/min). Gaps are times when nothing could be measured; they are '
      + 'never joined up.',
  },

  // ------------------------------------------------------------------ status of a bed
  status_running: { title: 'Monitoring', body: 'The camera is running and the vitals are being measured.' },
  status_awaiting_calibration: {
    title: 'Needs chair setup',
    body: 'The chair or bed has not been marked yet, so monitoring has not started. Open the bed and choose Set up chair.',
  },
  status_camera_offline: {
    title: 'Camera offline',
    body: 'The camera is not delivering frames. The server retries every 5 s.',
  },
  status_starting: { title: 'Starting', body: 'The server is opening the camera and loading the models.' },
  status_error: {
    title: 'Engine error',
    body: 'The processing pipeline for this bed stopped with an error. The detail line says what failed; '
      + 'check the System page.',
  },
  thermalOn: {
    title: 'Thermal camera on',
    body: 'The ESP32 thermal camera (80 × 62 pixels, about 8 frames a second) is connected and adds a '
      + 'second breathing measurement from the air at the nostrils.',
  },
  thermalOff: {
    title: 'Thermal camera off',
    body: 'No thermal frames are arriving. Breathing continues from the depth camera alone.',
  },
  recording: {
    title: 'Recording a comparison session',
    body: 'Every heart-rate and breathing method is logged once a second next to the reference values '
      + 'you enter, for scoring on the Record & compare page. Numbers only: no video or images are saved.',
  },

  // ------------------------------------------------------------------ sources and fusion
  estimators: {
    title: 'Estimators',
    body: 'Each vital is measured several independent ways. Fused methods are blended, weighting each by '
      + 'how much it trusts its own reading, then smoothed so one bad second cannot jump the value. '
      + 'Shown-only methods run alongside for comparison.',
  },
  fused: {
    title: 'Fused',
    body: 'This method’s value is blended into the number on screen (inverse-variance weighting: a '
      + 'more certain reading counts more), and so into the early-warning score and alarms.',
  },
  shownOnly: {
    title: 'Shown only',
    body: 'Displayed and logged for comparison, but never part of the number on screen, the '
      + 'early-warning score or alarms.',
  },
  learned: {
    title: 'Experimental learned model',
    body: 'A neural network still being validated against reference measurements. As configured, the '
      + 'LBRD-IC breathing CNN and EfficientPhys heart-rate network are fused as secondary paths: they '
      + 'count only while within 4 br/min or 10 bpm of the signal-processing reading, and can stand in '
      + 'only when that reading is missing. The D435 breathing CNN is shown only.',
  },
  estimate: {
    title: 'Estimate',
    body: 'This method’s own latest value, before it is combined with the others. A word instead of a '
      + 'number says why it has none.',
  },
  role: {
    title: 'Role',
    body: 'Fused: counts towards the number on screen. Shown only: displayed for comparison and never alarms.',
  },
  compute: {
    title: 'Compute time',
    body: 'Milliseconds this method took for its last update on this computer.',
  },
  netOut: {
    title: 'Network output',
    body: 'What the network produced before the range and quality checks hid it. Logged so it can still '
      + 'be scored on the Record & compare page.',
  },
  src_flat_signal: { title: 'Flat signal', body: 'The input barely changed, so there is no rhythm to read.' },
  src_out_of_range: {
    title: 'Out of range',
    body: 'The estimate fell outside the plausible range for this vital, so it is hidden.',
  },
  src_low_quality: {
    title: 'Low quality',
    body: 'The pulse rhythm does not stand out clearly enough from noise to trust the number.',
  },
  src_no_face: { title: 'No face', body: 'The face could not be found, so this heart-rate method has no input.' },
  src_none: { title: 'No reading', body: 'This method has not produced a value in the last 5 s.' },

  // ------------------------------------------------------------------ early warning
  ews: {
    title: 'Early-warning score',
    body: 'A camera-derived subset of NEWS2: the points for respiratory rate and heart rate only (0–3 '
      + 'each, so 0–6). 0–1 routine, 2–4 watch closely, 5 or more urgent review. Not a full NEWS2: '
      + 'SpO₂, temperature, blood pressure and consciousness are not measured, so a low score does not '
      + 'rule out deterioration.',
  },
  ewsTotal: {
    title: 'Score out of 6',
    body: 'Respiratory-rate points plus heart-rate points. A vital that cannot be measured adds nothing '
      + 'and is listed as missing, never counted as normal.',
  },
  ews_green: { title: 'Routine (0–1)', body: 'Continue routine monitoring.' },
  ews_amber: {
    title: 'Watch closely (2–4)',
    body: 'Increase observation frequency. Raises a medium-priority alarm.',
  },
  ews_red: {
    title: 'Urgent review (5 or more)',
    body: 'Suggest urgent clinical assessment. Raises a high-priority alarm.',
  },
  ews_unknown: { title: 'No score', body: 'Neither heart rate nor breathing can be measured right now, so there is no score.' },
  ewsRR: {
    title: 'Respiratory-rate points',
    body: 'NEWS2 bands: ≤8 br/min +3, 9–11 +1, 12–20 0, 21–24 +2, ≥25 +3.',
  },
  ewsHR: {
    title: 'Heart-rate points',
    body: 'NEWS2 bands: ≤40 bpm +3, 41–50 +1, 51–90 0, 91–110 +1, 111–130 +2, ≥131 +3.',
  },
  notScored: {
    title: 'Not scored',
    body: 'This vital cannot be measured right now, so it adds no points. It is not treated as normal.',
  },
  bands: {
    title: 'NEWS2 band shading',
    body: 'Shaded ranges add early-warning points: pale orange +1, orange +2, red +3. Unshaded is 0 points '
      + '(heart rate 51–90 bpm, breathing 12–20 br/min).',
  },

  // ------------------------------------------------------------------ alarms
  alarms: {
    title: 'Alarms',
    body: 'High priority (red, fast flash): confirmed fall, possible apnoea, early-warning score 5 or more. '
      + 'Medium priority (orange, slow flash): score 2–4, paradoxical breathing. An unacknowledged '
      + 'alarm escalates: logged at once, highlighted after 10 s, chime from 30 s, phone push from 60 s.',
  },
  alarm_red: {
    title: 'High priority',
    body: 'Act now. Flashes about twice a second, with a red stop-sign icon. Raised by a confirmed fall, '
      + 'possible apnoea or an early-warning score of 5 or more.',
  },
  alarm_amber: {
    title: 'Medium priority',
    body: 'Look soon. Flashes about once every 2 s, with an orange triangle icon. Raised by an '
      + 'early-warning score of 2–4 or paradoxical breathing.',
  },
  escalation: {
    title: 'Escalation',
    body: 'An active alarm that nobody acknowledges escalates in stages: logged only at once, highlighted '
      + 'on screen after 10 s, chime after 30 s (when sound is on), phone push after 60 s.',
  },
  stage_log_only: { title: 'Stage: logged only', body: 'Recorded and listed, without highlighting or sound. The first 10 s of every alarm.' },
  stage_dashboard_highlight: { title: 'Stage: highlighted', body: 'Flashing in the alarm bar. Reached after 10 s unacknowledged.' },
  stage_chime: { title: 'Stage: chime', body: 'Tone in browsers with sound on. Reached after 30 s unacknowledged.' },
  stage_phone_push: {
    title: 'Stage: phone push',
    body: 'Reached after 60 s unacknowledged. The server marks this stage; no phone or pager is connected '
      + 'in this prototype.',
  },
  ack: {
    title: 'Acknowledge',
    body: 'Confirms someone has seen the alarm: it stops escalating and stops sounding. It stays listed '
      + 'until the condition clears, and becomes active again if it worsens from orange to red. A fall '
      + 'alarm stays up until acknowledged, even if the patient is back in the chair.',
  },
  acked: {
    title: 'Alarm acknowledged',
    body: 'Someone has seen this alarm. The condition is still present, so it stays open until it clears.',
  },
  staffSuppress: {
    title: 'Staff at bedside',
    body: 'Two or more other people are in the camera view, so staff are assumed to be with the patient. '
      + 'Alarms are still logged and listed, but do not flash, sound or escalate.',
  },
  staff: {
    title: 'Others in view',
    body: 'People in the camera view other than the patient, usually staff or visitors. They are ignored '
      + 'for the vitals. With two or more, alarms are logged without sound or escalation.',
  },
  shiftAlerts: {
    title: 'Alarms that sounded this shift',
    body: 'Alarms that reached the chime stage, without staff at the bedside, in the last 12 hours. The '
      + 'design target is no more than 3 per shift, to limit alarm fatigue.',
  },
  alarmTimes: {
    title: 'Alarm times',
    body: 'Started is when the condition was first seen (24-hour clock, with seconds). The alarm becomes '
      + 'active once the condition has lasted 1.5 s; falls, apnoea and paradoxical breathing are active at once.',
  },

  // ------------------------------------------------------------------ position and falls
  falls: {
    title: 'Position and falls',
    body: 'A pose model finds 17 points on the body. Where they sit relative to the marked chair gives the '
      + 'position. A fall is a fast downward movement followed by lying still, confirmed by a second, '
      + 'learned model; a confirmed fall raises a high-priority alarm that stays until acknowledged.',
  },
  state_IN_CHAIR: { title: 'In chair', body: 'At least 60 % of the patient’s tracked body points are inside the marked seat.' },
  state_SITTING_EDGE: {
    title: 'On chair edge',
    body: 'Between 20 % and 60 % of the patient’s tracked body points are inside the marked seat: sitting '
      + 'on the edge, leaning out or getting up. Shown, not alarmed.',
  },
  state_OUT_OF_CHAIR: {
    title: 'Out of chair',
    body: 'Fewer than 20 % of the patient’s body points are inside the marked seat and the fall detector '
      + 'does not see them lying. Shown on the bed; it does not raise an alarm by itself.',
  },
  state_ON_FLOOR: {
    title: 'On the floor',
    body: 'Out of the seat and the fall detector reads lying down. Check the patient.',
  },
  state_UNKNOWN: { title: 'Not in view', body: 'No patient is detected in the camera view.' },
  fallFsm: {
    title: 'Fall detector state',
    body: 'Upright or seated: normal. Falling motion: a fast downward movement is in progress. Lying down: '
      + 'the body is horizontal after a fall. A fall alarm needs the learned model to confirm it.',
  },
  inChairFrac: {
    title: 'In the chair',
    body: 'Share of the patient’s confidently tracked body points that are inside the marked seat. '
      + '60 % or more reads as in chair; below 20 % as out of chair.',
  },
  confirmedFall: { title: 'Last confirmed fall', body: 'Time of the last fall confirmed in this session, 24-hour clock.' },
  fallModel: {
    title: 'Model confidence',
    body: 'Probability from the learned fall model (a GRU network over the last 32 frames of body '
      + 'movement) that the recent movement is a fall.',
  },
  postureStrip: {
    title: 'Posture over time',
    body: 'The fall detector’s state across the selected trend window: blank for upright or seated, '
      + 'orange for falling motion, red for lying down.',
  },

  // ------------------------------------------------------------------ breathing mechanics
  taa: {
    title: 'Breathing synchrony',
    body: 'Thoraco-abdominal asynchrony. Chest and belly normally rise together. The depth camera compares '
      + 'their timing; when they see-saw (paradoxical breathing) it can signal respiratory muscle fatigue '
      + 'or airway obstruction, even while the rate looks normal.',
  },
  phaseGap: {
    title: 'Phase gap',
    body: 'How far out of step chest and belly are: 0° moving together, 180° exactly opposite. A gap '
      + 'above 68° (1.2 rad) held for 10 s flags paradoxical breathing. The seconds show how long the '
      + 'current gap has been held.',
  },
  taaFlag: {
    title: 'Paradoxical breathing',
    body: 'Chest and abdomen have moved out of phase (gap above 68°) for at least 10 s. Raises a '
      + 'medium-priority alarm.',
  },

  // ------------------------------------------------------------------ pain
  pain: {
    title: 'Pain expression (experimental)',
    body: 'Probability that the face shows a pain expression, averaged over the last 5 scores, from a '
      + 'MobileNetV3 model trained on synthetic faces (SynPAIN). The training set was almost all pain '
      + 'images, so it leans towards pain: even a blank image scores about 44 %. For research only: never '
      + 'scored and never raises an alarm.',
  },
  painThreshold: {
    title: 'Model threshold',
    body: 'Cut-off chosen on the training data (29.6 %). Above it the model calls the expression pain; '
      + 'because of the training imbalance, that is not a clinical pain assessment.',
  },

  // ------------------------------------------------------------------ trends
  trends: {
    title: 'Trends',
    body: 'Heart rate and breathing over 5 min, 15 min, 1 h or 4 h, sampled once a second. Shading marks '
      + 'the NEWS2 bands; gaps are times when nothing could be measured. Overlay every estimator to '
      + 'compare the individual methods with the number shown.',
  },
  trendWindow: {
    title: 'Trend window',
    body: 'How much history the charts show: 5 min, 15 min, 1 h or 4 h. The server keeps 4 hours at one '
      + 'value per second; longer windows are thinned to fit the chart.',
  },
  coverageTrend: {
    title: 'Measured share',
    body: 'Share of seconds in this window that had a valid reading. Low coverage means long gaps, so the '
      + 'trend says less.',
  },
  shownLine: {
    title: 'Shown value',
    body: 'The number displayed on the monitor: the fused, smoothed reading.',
  },
  overlay: {
    title: 'Overlay every estimator',
    body: 'Adds each individual method to the chart as a thin line (dashed for learned models), to see '
      + 'which ones agree with the number shown.',
  },

  // ------------------------------------------------------------------ live view
  overlayLegend: {
    title: 'Overlay',
    body: 'Drawn by the monitor on top of the video: the marked seat, the patient’s pose, other people, '
      + 'and the breathing region when the thermal camera is aligned.',
  },
  seatOutline: { title: 'Seat outline', body: 'The chair or bed corners marked during setup. Position and falls are judged against it.' },
  poseSkeleton: { title: 'Patient pose', body: 'The 17 body points the pose model found on the patient. Only confident points are drawn.' },
  staffBox: { title: 'Staff', body: 'Other people in view. They are ignored for the vitals; two or more silence alarm escalation.' },
  nostrilRegion: {
    title: 'Breathing region',
    body: 'Where the thermal camera reads breath temperature, below the nose. Shown once the thermal camera is aligned.',
  },
  faceBlur: {
    title: 'Faces blurred',
    body: 'The server blurs every detected face before the video leaves it, and the live view is not recorded.',
  },
  faceBlurOff: {
    title: 'Face blur unavailable',
    body: 'Faces are found by the pose model, which is not loaded, so they cannot be blurred. See the System page.',
  },
  fps: {
    title: 'Frame rate and processing time',
    body: 'Frames processed per second (the camera delivers 30) and the average time to process one frame. '
      + 'Low fps or high milliseconds mean the computer is struggling and values update more slowly.',
  },

  // ------------------------------------------------------------------ thermal
  thermal: {
    title: 'Thermal camera',
    body: 'An 80 × 62 pixel thermal sensor on an ESP32, about 8 frames a second. Breath out warms the '
      + 'skin below the nose, breath in cools it; counting those cycles gives a second breathing rate that '
      + 'works in the dark.',
  },
  thermalAlign: {
    title: 'Thermal alignment',
    body: 'The two cameras sit apart with different lenses. Marking the same spots in both images (at '
      + 'least 4) lets the monitor map one onto the other, so breathing is read at the nostrils found in '
      + 'the colour image instead of the warmest spot. Redo it only if a camera is moved.',
  },
  alignQuality: {
    title: 'Alignment quality',
    body: 'With more than 4 points the monitor measures how far the mapped points miss on average: under '
      + '10 px good, under 20 px fair, otherwise poor (realign). With exactly 4 the fit is always exact, '
      + 'so accuracy is unchecked.',
  },
  breathTrace: {
    title: 'Breath temperature',
    body: 'Temperature in the breathing region over the last 30 s: it rises on each breath out and falls '
      + 'on each breath in. Without alignment it is read at the warmest spot, which may be the forehead.',
  },
  thermalScale: {
    title: 'Colour scale',
    body: 'Dark is cool, bright is warm, in °C. The range follows this frame, ignoring up to two '
      + 'faulty hot pixels.',
  },

  // ------------------------------------------------------------------ chair setup
  chairCal: {
    title: 'Chair setup',
    body: 'Click the four corners of the chair or bed seat once per session. The monitor uses this outline '
      + 'to pick out the patient from staff, to work out how much of the patient is in the seat, and takes '
      + 'the seat’s distance as the reference for falls.',
  },
  furniture: {
    title: 'Chair or bed',
    body: 'Which kind of furniture you are marking. Both are traced the same way, by their four corners.',
  },
  seatDepth: {
    title: 'Seat distance',
    body: 'Distance from the camera to the marked seat, measured by the depth camera. A body much further '
      + 'away than the seat is treated as nearer the floor.',
  },
  reuseCalibration: {
    title: 'Use last calibration',
    body: 'Reuse the seat outline saved earlier. Only valid if neither the camera nor the chair has moved.',
  },

  // ------------------------------------------------------------------ record & compare
  results: {
    title: 'Scores against the reference',
    body: 'Each second the reference you logged is compared with what every method showed. Read MAE '
      + 'together with coverage: a method that answers rarely can look accurate. The first seconds, and '
      + 'the seconds right after each reference change, are skipped.',
  },
  condition: {
    title: 'Condition',
    body: 'What was happening during the recording, so sessions can be grouped (rest, paced breathing, '
      + 'dim light, talking, and so on).',
  },
  sessLabel: { title: 'Label', body: 'A short name for this session. Use a code, never a patient name.' },
  refRR: {
    title: 'Reference breathing rate (br/min)',
    body: 'The rate you trust: a metronome pace, or breaths counted over a full minute. Log a new value '
      + 'whenever it changes.',
  },
  refHR: {
    title: 'Reference heart rate (bpm)',
    body: 'The rate you trust, usually from a fingertip pulse oximeter. Log a new value every 30–60 s or '
      + 'when it moves by more than 3 bpm.',
  },
  sessNotes: { title: 'Notes', body: 'Lighting, distance, clothing, anything unusual. No names or identifying details.' },
  logReference: {
    title: 'Log reference',
    body: 'Records a new reference value at this moment. Scoring uses the latest one before each second.',
  },
  coverage: {
    title: 'Coverage',
    body: 'Share of scored seconds in which the method gave a number at all. Below 50 % the bar turns '
      + 'orange and the method cannot be marked lowest MAE.',
  },
  mae: {
    title: 'MAE, mean absolute error',
    body: 'The average size of the miss against the reference, in the vital’s unit, over the seconds '
      + 'where the method gave a value. Lower is better.',
  },
  bias: {
    title: 'Bias',
    body: 'Average of (estimate minus reference). Positive means it reads high on average, negative low.',
  },
  rmse: {
    title: 'RMSE, root-mean-square error',
    body: 'Like MAE, but large misses count more. RMSE well above MAE means occasional big errors.',
  },
  withinTol: {
    title: 'Within tolerance',
    body: 'Share of valued seconds within ±2 br/min (breathing) or ±5 bpm (heart rate) of the reference.',
  },
  pearson: {
    title: 'r, correlation',
    body: 'Pearson correlation between estimate and reference over time, from −1 to 1: does the method '
      + 'follow changes? Needs at least 10 s with values and a reference that changes.',
  },
  maeUnfiltered: {
    title: 'MAE unfiltered',
    body: 'For learned models: error and coverage of the network’s raw output before the range and '
      + 'quality checks, to show what it would score if nothing were hidden.',
  },
  warmup: {
    title: 'Skip first (warm-up)',
    body: 'Seconds at the start of a session left out of scoring, because each method’s analysis window '
      + 'is still filling. Default 30 s.',
  },
  settle: {
    title: 'Settle time',
    body: 'Seconds after each reference change left out of scoring, because the windows still hold the '
      + 'old rate. Default 20 s.',
  },
  lowestMae: {
    title: 'Lowest MAE',
    body: 'The individual method with the smallest error among those that answered at least half the time.',
  },
  screenShowed: { title: 'What the screen showed', body: 'The fused, smoothed value the monitor displayed during the session.' },

  // ------------------------------------------------------------------ system
  modules: {
    title: 'Pipeline status',
    body: 'Each part of the processing pipeline loads on its own. If one cannot run, only what depends on '
      + 'it stops; the rest keeps monitoring. The detail line says why and how to fix it.',
  },
  mod_ok: { title: 'Running', body: 'Loaded and producing output.' },
  mod_degraded: {
    title: 'Degraded',
    body: 'Running with a limitation, such as a missing optional model or a part it depends on. The detail line says which.',
  },
  mod_unavailable: {
    title: 'Not available',
    body: 'Not running: a weight file or library is missing, or it was switched off after repeated errors.',
  },
  mod_error: { title: 'Error', body: 'Its last update failed. The server keeps trying on the next frames.' },
  mod_pending: { title: 'Pending', body: 'Not started yet, or has not reported its state.' },
  moduleErrors: { title: 'Errors', body: 'How many times this part has failed since the server started.' },
  mod_pose: { title: 'Pose tracking', body: 'Finds people and 17 body points per person (YOLOv8 pose). Needed for falls, face blur and the face crops.' },
  mod_rr: { title: 'Respiratory rate', body: 'Breathing from chest depth and, when connected, nostril temperature.' },
  mod_hr: { title: 'Heart rate', body: 'Heart rate from skin colour changes in the face (POS rPPG).' },
  mod_fall: { title: 'Fall detection', body: 'Posture state machine plus a learned GRU model that confirms falls.' },
  mod_taa: { title: 'Breathing synchrony', body: 'Compares chest and abdomen movement timing from the depth camera.' },
  mod_pain: { title: 'Pain expression', body: 'Experimental face model. Never alarms.' },
  mod_rr_learned: { title: 'Learned breathing models', body: 'Experimental CNNs for breathing rate (LBRD-IC and the D435 fine-tune).' },
  mod_hr_learned: { title: 'Learned heart-rate model', body: 'Experimental EfficientPhys network on the face crop.' },
  weightFiles: {
    title: 'Weight files',
    body: 'The trained model files the pipeline loads. A missing file switches off only the part that needs it.',
  },
  wf_file: { title: 'File', body: 'Path of the model file inside the project.' },
  wf_usedBy: { title: 'Used by', body: 'The part of the pipeline that loads this file.' },
  wf_size: { title: 'Size', body: 'File size on disk, in kilobytes.' },
  wf_status: {
    title: 'Status',
    body: 'Present: found. Missing: required, so the part that uses it is off. Optional, absent: not '
      + 'needed for monitoring.',
  },

  // ------------------------------------------------------------------ home
  wardPreview: {
    title: 'On the ward now',
    body: 'Live heart rate (green, bpm) and breathing rate (yellow, br/min) for every bed, with the most '
      + 'important status in words. A dash means it cannot be measured right now. Select a bed to open it.',
  },
  raisesAlarms: {
    title: 'Raises alarms',
    body: 'Feeds the alarm system: falls, apnoea and the early-warning score from heart rate and breathing.',
  },
  flagsBedView: {
    title: 'Flags on the bed view',
    body: 'Shown on the ward strip and the bed page; sustained paradoxical breathing also raises a medium-priority alarm.',
  },
});

export type GlossKey = keyof typeof GLOSSARY;

/** Glossary key for a vital's "why no number" reason code (lib/format.ts). */
export function reasonKey(reason: string | null | undefined, stale: boolean, apnea = false): GlossKey {
  if (apnea) return 'apnoea';
  const k = `reason_${reason}`;
  if (reason && k in GLOSSARY) return k as GlossKey;
  return stale ? 'reason_stale' : 'reason_waiting';
}

/** Glossary key for one source's own reason code (SourcesPanel). */
export function sourceReasonKey(reason: string | null | undefined): GlossKey {
  for (const k of [`src_${reason}`, `reason_${reason}`]) if (reason && k in GLOSSARY) return k as GlossKey;
  return 'src_none';
}

export function glossKey(prefix: string, code: string | null | undefined, fallback: GlossKey): GlossKey {
  const k = `${prefix}${code}`;
  return code && k in GLOSSARY ? (k as GlossKey) : fallback;
}
