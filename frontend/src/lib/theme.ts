// theme.ts — dark (night ward, default) / light (daylight) theme, remembered per browser.
import { useEffect, useState } from 'react';

export type Theme = 'dark' | 'light';
const KEY = 'visionicu.theme';

function read(): Theme {
  try {
    const v = window.localStorage.getItem(KEY);
    if (v === 'dark' || v === 'light') return v;
  } catch { /* storage blocked */ }
  return 'dark';
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(read);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { window.localStorage.setItem(KEY, theme); } catch { /* storage blocked */ }
  }, [theme]);
  return [theme, setTheme];
}
