"use client";

import { useEffect, useMemo, useState } from "react";

import type { TeamRef, WinProbPoint, WinProbSeries } from "@/lib/api";
import { gameClock } from "@/lib/format";
import { SplitBar } from "./SplitBar";
import { WinProbChart } from "./WinProbChart";

type Message =
  | ({ type: "snapshot" } & WinProbSeries)
  | {
      type: "points";
      game_id: string;
      source: "live" | "replay";
      points: WinProbPoint[];
    }
  | { type: "reset"; game_id: string };

function merge(current: Map<number, WinProbPoint>, points: WinProbPoint[]) {
  const next = new Map(current);
  for (const p of points) next.set(p.action_id, p);
  return next;
}

/**
 * Subscribes to the game's WebSocket stream: a snapshot of every point so far, then new points
 * as the live poller (or a replay) records them. Points are keyed by action_id, so a point
 * that arrives in both the snapshot and an update is counted once. Reconnects with backoff.
 */
function useLiveWinProb(
  gameId: string,
  wsBase: string,
  initial: WinProbSeries,
) {
  const [points, setPoints] = useState(() => merge(new Map(), initial.points));
  const [source, setSource] = useState(initial.source);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    let closed = false;

    function connect() {
      socket = new WebSocket(
        `${wsBase}/api/ws/games/${encodeURIComponent(gameId)}`,
      );
      socket.onopen = () => {
        attempts = 0;
        setConnected(true);
      };
      socket.onmessage = (event: MessageEvent<string>) => {
        const message = JSON.parse(event.data) as Message;
        if (message.type === "snapshot") {
          setPoints(merge(new Map(), message.points));
          setSource(message.source);
        } else if (message.type === "points") {
          setPoints((cur) => merge(cur, message.points));
          setSource(message.source);
        } else if (message.type === "reset") setPoints(new Map());
      };
      socket.onclose = () => {
        setConnected(false);
        if (closed) return;
        attempts += 1;
        retry = setTimeout(connect, Math.min(30_000, 1000 * 2 ** attempts));
      };
    }

    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      socket?.close();
    };
  }, [gameId, wsBase]);

  const sorted = useMemo(
    () => [...points.values()].sort((a, b) => a.action_id - b.action_id),
    [points],
  );
  return { points: sorted, connected, source };
}

export function LiveWinProb({
  gameId,
  wsBase,
  initial,
  home,
  away,
  live,
}: {
  gameId: string;
  wsBase: string;
  initial: WinProbSeries;
  home: TeamRef;
  away: TeamRef;
  live: boolean;
}) {
  const { points, connected, source } = useLiveWinProb(gameId, wsBase, initial);
  if (points.length === 0) {
    return live ? (
      <p className="text-ink-2">Waiting for the first play of the game.</p>
    ) : null;
  }
  const last = points[points.length - 1];
  const over = last.home_win_prob === 0 || last.home_win_prob === 1;

  return (
    <section aria-labelledby="in-game" className="flex flex-col gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
        <h2
          id="in-game"
          className="font-condensed text-3xl font-bold tracking-tight"
        >
          {over ? "How the game unfolded" : "Win probability right now"}
        </h2>
        <p className="tabular text-ink-2" aria-live="polite">
          {last.action_id === 0
            ? "Tip-off"
            : gameClock(last.period, last.clock_seconds)}
          {", "}
          {away.abbreviation} {last.score_away}, {home.abbreviation}{" "}
          {last.score_home}
          {!over && live && (
            <span className="ml-3 text-sm text-muted">
              {connected ? "Updating live" : "Reconnecting"}
            </span>
          )}
          {source === "replay" && (
            <span className="ml-3 text-sm text-muted">Replay</span>
          )}
        </p>
      </div>
      <SplitBar home={home} away={away} homeProb={last.home_win_prob} />
      <WinProbChart points={points} home={home} away={away} />
      <details className="text-sm">
        <summary className="cursor-pointer text-ink-2">Show as a table</summary>
        <table className="tabular mt-2 w-full max-w-md">
          <thead>
            <tr className="text-left text-ink-2">
              <th className="py-1 font-medium">Moment</th>
              <th className="py-1 text-right font-medium">Score</th>
              <th className="py-1 text-right font-medium">
                {home.abbreviation} chance
              </th>
            </tr>
          </thead>
          <tbody>
            {points
              .filter(
                (p, i) =>
                  i === 0 || i === points.length - 1 || p.clock_seconds === 0,
              )
              .map((p) => (
                <tr key={p.action_id}>
                  <td className="py-0.5">
                    {p.action_id === 0
                      ? "Tip-off"
                      : gameClock(p.period, p.clock_seconds)}
                  </td>
                  <td className="py-0.5 text-right">
                    {p.score_away}–{p.score_home}
                  </td>
                  <td className="py-0.5 text-right">
                    {Math.round(p.home_win_prob * 100)}%
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
