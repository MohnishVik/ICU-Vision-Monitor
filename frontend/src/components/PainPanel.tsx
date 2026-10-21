// PainPanel.tsx — honest placeholder until src/modules/pain.py is implemented.
// The server reports pain.available=false; we say so and what is needed.
import { Hourglass } from 'lucide-react';

export default function PainPanel({ pain }: { pain: { available: boolean; reason?: string } | null }) {
  if (pain?.available) {
    return (
      <section className="panel p-5" aria-labelledby="pain-h">
        <h2 id="pain-h" className="t-head">Pain assessment</h2>
        <p className="t-label text-ink2 mt-2">Pain output connected.</p>
      </section>
    );
  }
  return (
    <section className="panel p-5 flex flex-col gap-3" aria-labelledby="pain-h">
      <h2 id="pain-h" className="t-head">Pain assessment</h2>
      <div className="flex gap-3 items-start">
        <Hourglass className="h-5 w-5 text-mute mt-0.5 shrink-0" aria-hidden />
        <div className="flex flex-col gap-1">
          <p className="t-body text-ink">Not measured yet.</p>
          <p className="t-label text-ink2 max-w-[56ch]">
            The facial action-unit model (CPOT proxy, trained on SynPAIN) is still being integrated.
            No pain score is estimated, so none is shown.
          </p>
        </div>
      </div>
    </section>
  );
}
