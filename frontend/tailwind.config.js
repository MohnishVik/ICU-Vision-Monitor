/** @type {import('tailwindcss').Config} */
// Tokens live as CSS variables in src/index.css; Tailwind only names them.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ward: 'var(--ward)',
        panel: 'var(--panel)',
        raised: 'var(--raised)',
        hairline: 'var(--hairline)',
        ink: 'var(--ink)',
        ink2: 'var(--ink-2)',
        mute: 'var(--mute)',
        scrub: 'var(--scrub)',
        hr: 'var(--hr)',
        rr: 'var(--rr)',
        amber: 'var(--amber)',
        red: 'var(--red)',
      },
      fontFamily: {
        sans: ['Barlow', 'system-ui', 'sans-serif'],
        semi: ['"Barlow Semi Condensed"', 'Barlow', 'system-ui', 'sans-serif'],
        cond: ['"Barlow Condensed"', 'Barlow', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
