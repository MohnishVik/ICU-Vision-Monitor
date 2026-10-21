// sound.ts — alarm tones via WebAudio (no audio files to ship).
// Patterns follow the spirit of IEC 60601-1-8: high priority = two bursts of
// five rising pulses, medium = three pulses. Browsers only allow audio after a
// user gesture, so sound is enabled explicitly from the top bar.

let ctx: AudioContext | null = null;

export function enableSound(): boolean {
  try {
    const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext;
    if (!Ctor) return false;
    ctx = ctx ?? new Ctor();
    void ctx.resume();
    return true;
  } catch {
    return false;
  }
}

function pulse(at: number, freq: number, len: number) {
  if (!ctx) return;
  const o = ctx.createOscillator();
  const g = ctx.createGain();
  o.type = 'sine';
  o.frequency.value = freq;
  g.gain.setValueAtTime(0, at);
  g.gain.linearRampToValueAtTime(0.18, at + 0.015);
  g.gain.setValueAtTime(0.18, at + len - 0.03);
  g.gain.linearRampToValueAtTime(0, at + len);
  o.connect(g).connect(ctx.destination);
  o.start(at);
  o.stop(at + len + 0.02);
}

export function playAlarm(severity: 'red' | 'amber') {
  if (!ctx || ctx.state !== 'running') return;
  const t = ctx.currentTime + 0.05;
  if (severity === 'red') {
    const notes = [523, 659, 784, 659, 784];
    for (let burst = 0; burst < 2; burst++) {
      notes.forEach((f, i) => pulse(t + burst * 1.1 + i * 0.16, f, 0.12));
    }
  } else {
    [523, 659, 784].forEach((f, i) => pulse(t + i * 0.22, f, 0.16));
  }
}
