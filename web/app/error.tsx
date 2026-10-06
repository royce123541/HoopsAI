"use client";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  return (
    <div>
      <h1 className="font-condensed text-5xl font-bold tracking-tight">
        Data unavailable
      </h1>
      <p className="mt-2 max-w-prose text-ink-2">
        {error.message || "The HoopsAI API could not be reached."} Check that
        the API is running, then try again.
      </p>
      <button
        type="button"
        onClick={reset}
        className="mt-4 rounded-md border border-line px-4 py-2 font-medium hover:bg-surface"
      >
        Try again
      </button>
    </div>
  );
}
