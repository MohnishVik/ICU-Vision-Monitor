/** @type {import('tailwindcss').Config} */
// Tokens live as CSS variables in src/index.css (dark + light); Tailwind only names them.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)',
        surface: 'var(--surface)',
        surface2: 'var(--surface-2)',
        surface3: 'var(--surface-3)',
        line: 'var(--line)',
        linestrong: 'var(--line-strong)',
        ink: 'var(--ink)',
        ink2: 'var(--ink-2)',
        mute: 'var(--mute)',
        accent: 'var(--accent)',
        hr: 'var(--hr)',
        rr: 'var(--rr)',
        amber: 'var(--amber)',
        amberink: 'var(--amber-ink)',
        red: 'var(--red)',
        redink: 'var(--red-ink)',
        ok: 'var(--ok)',
      },
      fontFamily: {
        sans: ['"Atkinson Hyperlegible"', 'system-ui', 'sans-serif'],
        cond: ['"IBM Plex Sans Condensed"', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
