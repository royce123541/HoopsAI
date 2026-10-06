// Server-side API client. API_INTERNAL_URL is read at request time, so one built image works
// in any environment (compose: http://api:8000, local dev: http://127.0.0.1:8000; not
// localhost, whose IPv6 Docker port forwarding can stall server-side requests).
// Types come from the API's OpenAPI schema: `npm run gen:api` regenerates lib/api-types.ts.
import { connection } from "next/server";

import type { components } from "./api-types";

type Schemas = components["schemas"];
export type Slate = Schemas["Slate"];
export type GameSummary = Schemas["GameSummary"];
export type GameDetail = Schemas["GameDetail"];
export type TeamRef = Schemas["TeamRef"];
export type TeamListItem = Schemas["TeamListItem"];
export type TeamDetail = Schemas["TeamDetail"];
export type TeamGame = Schemas["TeamGame"];
export type ModelInfo = Schemas["ModelInfo"];
export type Factor = Schemas["Factor"];
export type ComparisonRow = Schemas["ComparisonRow"];
export type EloPoint = Schemas["EloPoint"];
export type ReliabilityBin = Schemas["ReliabilityBin"];
export type SeasonMetrics = Schemas["SeasonMetrics"];
export type FeatureImportance = Schemas["FeatureImportance"];
export type WinProbSeries = Schemas["WinProbSeries"];
export type WinProbPoint = Schemas["WinProbPoint"];
export type LiveSummary = Schemas["LiveSummary"];
export type MonitoringResult = Schemas["MonitoringResult"];

function apiBaseUrl(): string {
  return process.env.API_INTERNAL_URL ?? "http://127.0.0.1:8000";
}

/** GET a JSON resource; null on 404, an error (shown by app/error.tsx) otherwise. */
async function get<T>(path: string): Promise<T | null> {
  await connection(); // always render at request time; never bake API data into the build
  const res = await fetch(`${apiBaseUrl()}/api${path}`, {
    cache: "no-store",
    signal: AbortSignal.timeout(5000),
  });
  if (res.status === 404) return null;
  if (!res.ok)
    throw new Error(`The HoopsAI API returned ${res.status} for ${path}.`);
  return (await res.json()) as T;
}

export const getSlate = (date?: string) =>
  get<Slate>(date ? `/games?date=${encodeURIComponent(date)}` : "/games");
export const getGame = (id: string) =>
  get<GameDetail>(`/games/${encodeURIComponent(id)}`);
export const getTeams = () => get<TeamListItem[]>("/teams");
export const getTeam = (id: string) =>
  get<TeamDetail>(`/teams/${encodeURIComponent(id)}`);
export const getModel = () => get<ModelInfo>("/model");
export const getMonitoring = () => get<MonitoringResult[]>("/model/monitoring");
export const getWinProb = (id: string) =>
  get<WinProbSeries>(`/games/${encodeURIComponent(id)}/winprob`);

/** WebSocket base URL as the browser sees the API (read at request time, not build time). */
export function publicWsUrl(): string {
  return process.env.API_PUBLIC_WS_URL ?? "ws://localhost:8000";
}
