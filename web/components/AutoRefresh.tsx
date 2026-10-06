"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

/** Re-render the server page every `seconds` (used while games on the slate are live). */
export function AutoRefresh({ seconds }: { seconds: number }) {
  const router = useRouter();
  useEffect(() => {
    const id = setInterval(() => router.refresh(), seconds * 1000);
    return () => clearInterval(id);
  }, [router, seconds]);
  return null;
}
