"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  {
    href: "/",
    label: "Games",
    match: (p: string) => p === "/" || p.startsWith("/games"),
  },
  {
    href: "/teams",
    label: "Teams",
    match: (p: string) => p.startsWith("/teams"),
  },
  {
    href: "/model",
    label: "Model",
    match: (p: string) => p.startsWith("/model"),
  },
];

export function NavLinks() {
  const pathname = usePathname();
  return (
    <nav aria-label="Main">
      <ul className="flex gap-5 text-[15px] font-medium">
        {LINKS.map(({ href, label, match }) => {
          const active = match(pathname);
          return (
            <li key={href}>
              <Link
                href={href}
                aria-current={active ? "page" : undefined}
                className={
                  active
                    ? "text-ink underline decoration-2 underline-offset-[6px]"
                    : "text-ink-2 hover:text-ink"
                }
              >
                {label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
