// Server-side API client. API_INTERNAL_URL is read at request time, so one built image
// works in any environment (compose: http://api:8000, local dev: http://localhost:8000).
// Types are hand-written until openapi-typescript generation lands in M3.

export type HealthResponse = {
  status: "ok" | "degraded";
  version: string;
  checks: Record<string, string>;
};

function apiBaseUrl(): string {
  return process.env.API_INTERNAL_URL ?? "http://localhost:8000";
}

export async function getHealth(): Promise<HealthResponse | null> {
  try {
    const res = await fetch(`${apiBaseUrl()}/api/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return null;
    return (await res.json()) as HealthResponse;
  } catch {
    return null; // API unreachable: the page renders an "offline" state instead of erroring
  }
}
