import Link from "next/link";

export default function NotFound() {
  return (
    <div>
      <h1 className="font-condensed text-5xl font-bold tracking-tight">
        Page not found
      </h1>
      <p className="mt-2 text-ink-2">
        That game or team isn&apos;t in the HoopsAI database.{" "}
        <Link href="/" className="text-ink underline underline-offset-4">
          Go to today&apos;s games
        </Link>
      </p>
    </div>
  );
}
