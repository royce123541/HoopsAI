import { CalibrationChart } from "@/components/CalibrationChart";
import { SeasonComparison } from "@/components/SeasonComparison";
import { getModel } from "@/lib/api";
import { percent, seasonLabel } from "@/lib/format";

function Stat({
  label,
  value,
  baseline,
}: {
  label: string;
  value: string;
  baseline: string;
}) {
  return (
    <div>
      <dt className="text-sm text-ink-2">{label}</dt>
      <dd className="mt-1">
        <span className="font-condensed text-4xl font-bold leading-none">
          {value}
        </span>
        <span className="ml-2 text-sm text-ink-2">Elo: {baseline}</span>
      </dd>
    </div>
  );
}

export default async function ModelPage() {
  const model = await getModel();
  if (!model) {
    return (
      <div>
        <h1 className="font-condensed text-5xl font-bold tracking-tight">
          The model
        </h1>
        <p className="mt-2 max-w-prose text-ink-2">
          No predictions have been made yet. Run <code>hoopsai train</code> and
          then <code>hoopsai predict</code> to publish the first model.
        </p>
      </div>
    );
  }
  const wins = model.seasons.filter(
    (s) => s.model.logloss < s.elo.logloss,
  ).length;
  const n = model.seasons.length;
  const [first, last] = [model.seasons[0].season, model.seasons[n - 1].season];
  const maxShare = Math.max(...model.top_features.map((f) => f.share));

  return (
    <div className="flex flex-col gap-12">
      <header>
        <h1 className="max-w-3xl font-condensed text-5xl font-bold tracking-tight">
          {wins === n
            ? `More accurate than Elo in all ${n} test seasons`
            : `More accurate than Elo in ${wins} of ${n} test seasons`}
        </h1>
        <p className="mt-3 max-w-prose text-ink-2">
          Each season from {seasonLabel(first)} to {seasonLabel(last)} was
          predicted by a model trained only on the seasons before it, then
          compared with a classic Elo rating system on the same{" "}
          {model.pooled_model.n.toLocaleString("en-US")} games.
        </p>
        <dl className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-3">
          <Stat
            label="Log loss (lower is better)"
            value={model.pooled_model.logloss.toFixed(3)}
            baseline={model.pooled_elo.logloss.toFixed(3)}
          />
          <Stat
            label="Winner picked correctly"
            value={percent(model.pooled_model.accuracy, 1)}
            baseline={percent(model.pooled_elo.accuracy, 1)}
          />
          <Stat
            label="Calibration error"
            value={percent(model.pooled_model.ece, 1)}
            baseline={percent(model.pooled_elo.ece, 1)}
          />
        </dl>
      </header>

      <section aria-labelledby="seasons">
        <h2
          id="seasons"
          className="mb-4 font-condensed text-3xl font-bold tracking-tight"
        >
          Season by season
        </h2>
        <SeasonComparison seasons={model.seasons} />
      </section>

      <section aria-labelledby="calibration">
        <h2
          id="calibration"
          className="font-condensed text-3xl font-bold tracking-tight"
        >
          Can you trust a 70%?
        </h2>
        <p className="mt-1 mb-4 max-w-prose text-ink-2">
          Games are grouped by the predicted chance of a home win. If the dots
          sit on the diagonal, a team given 70% wins about 70% of the time.
        </p>
        <CalibrationChart bins={model.reliability} />
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-ink-2">
            Show as a table
          </summary>
          <table className="tabular mt-2 w-full max-w-md">
            <thead>
              <tr className="text-left text-ink-2">
                <th className="py-1 font-medium">Predicted</th>
                <th className="py-1 text-right font-medium">Home team won</th>
                <th className="py-1 text-right font-medium">Games</th>
              </tr>
            </thead>
            <tbody>
              {model.reliability.map((b) => (
                <tr key={b.bin_low}>
                  <td className="py-0.5">{percent(b.mean_pred, 1)}</td>
                  <td className="py-0.5 text-right">
                    {percent(b.observed, 1)}
                  </td>
                  <td className="py-0.5 text-right">
                    {b.count.toLocaleString("en-US")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      </section>

      <section aria-labelledby="inputs">
        <h2
          id="inputs"
          className="font-condensed text-3xl font-bold tracking-tight"
        >
          What the model relies on
        </h2>
        <p className="mt-1 mb-4 max-w-prose text-ink-2">
          Share of the model&apos;s decisions that used each input, across all
          games.
        </p>
        <ul className="flex flex-col gap-2">
          {model.top_features.map((f) => (
            <li
              key={f.feature}
              className="grid grid-cols-[minmax(0,15rem)_1fr] items-center gap-4 text-[15px]"
            >
              <span className="truncate text-ink-2">{f.label}</span>
              <span className="flex items-center gap-2">
                <span
                  className="h-3 rounded-r bg-home"
                  style={{ width: `${(f.share / maxShare) * 85}%` }}
                />
                <span className="tabular text-ink">{percent(f.share, 1)}</span>
              </span>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="about" className="text-[15px] text-ink-2">
        <h2
          id="about"
          className="mb-2 font-condensed text-3xl font-bold tracking-tight text-ink"
        >
          About this version
        </h2>
        <p className="max-w-prose">
          Model v{model.version}, a gradient-boosted tree model trained on every
          game from the {model.train_seasons.replace("-", " to ")} seasons,
          using feature set {model.feature_version}
          {model.calibration === "none"
            ? ". Its probabilities are used as-is: an extra calibration step made them worse in testing."
            : ", with isotonic calibration."}
        </p>
      </section>
    </div>
  );
}
