import { connection } from "next/server";

import { getHealth } from "@/lib/api";

const STATUS_STYLES = {
  ok: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
  degraded: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  offline: "bg-red-500/15 text-red-600 dark:text-red-400",
} as const;

export default async function Home() {
  await connection(); // render per request; never bake API status in at build time
  const health = await getHealth();
  const status = health?.status ?? "offline";

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-8 px-4 py-16">
      <header className="flex flex-col gap-2">
        <h1 className="text-4xl font-semibold tracking-tight">HoopsAI</h1>
        <p className="text-foreground/70">
          Machine-learned win probabilities for every NBA game: before tip-off and live.
        </p>
      </header>

      <section
        aria-labelledby="system-status"
        className="rounded-xl border border-foreground/10 p-6"
      >
        <div className="flex items-center justify-between gap-4">
          <h2 id="system-status" className="text-lg font-medium">
            System status
          </h2>
          <span
            data-testid="api-status"
            className={`rounded-full px-3 py-1 text-sm font-medium ${STATUS_STYLES[status]}`}
          >
            {status}
          </span>
        </div>
        {health ? (
          <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
            <dt className="text-foreground/60">API version</dt>
            <dd className="font-mono">{health.version}</dd>
            {Object.entries(health.checks).map(([name, result]) => (
              <div key={name} className="contents">
                <dt className="capitalize text-foreground/60">{name}</dt>
                <dd className="font-mono">{result}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="mt-4 text-sm text-foreground/70">
            The API is unreachable. Start it with <code>docker compose up</code>.
          </p>
        )}
      </section>
    </main>
  );
}
