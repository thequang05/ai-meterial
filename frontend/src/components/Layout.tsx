import { useEffect, useState } from "react";
import { Outlet, NavLink } from "react-router-dom";
import { Beaker, Boxes, GitBranch, MessageSquare, Info, Moon, Sun, CircleDot } from "lucide-react";
import { fetchHealth } from "@/api/campaigns";
import { cn } from "@/lib/utils";
import type { HealthResponse } from "@/lib/types";

const NAV = [
  { to: "/",          label: "Overview",  icon: Beaker },
  { to: "/campaigns", label: "Campaigns", icon: Boxes },
  { to: "/pipeline",  label: "Pipeline",  icon: GitBranch },
  { to: "/nl-query",  label: "NL Query",  icon: MessageSquare },
  { to: "/about",     label: "About",     icon: Info },
];

export function Layout() {
  const [dark, setDark] = useState<boolean>(() => {
    const saved = localStorage.getItem("theme");
    return saved ? saved === "dark" : true;
  });
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    fetchHealth().then(setHealth);
    const t = setInterval(() => fetchHealth().then(setHealth), 30_000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-40 border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-zinc-950/80 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 h-14 flex items-center gap-6">
          <NavLink to="/" className="flex items-center gap-2 font-bold text-base">
            <Beaker className="h-5 w-5 text-emerald-500" />
            <span>AI Material Discovery</span>
          </NavLink>
          <nav className="hidden md:flex items-center gap-1">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition",
                    isActive
                      ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                      : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="flex-1" />
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs">
              <CircleDot
                className={cn(
                  "h-3.5 w-3.5",
                  health?.neo4j === "up" ? "text-emerald-500" : "text-amber-500"
                )}
              />
              <span className="text-zinc-500 dark:text-zinc-400 hidden sm:inline">
                Neo4j {health?.neo4j ?? "..."}
              </span>
            </div>
            <button
              onClick={() => setDark((v) => !v)}
              className="rounded-md p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800"
              aria-label="Toggle theme"
            >
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
        <nav className="md:hidden border-t border-zinc-200 dark:border-zinc-800 px-2 py-1 overflow-x-auto flex gap-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-1 rounded-md px-3 py-1.5 text-xs font-medium whitespace-nowrap",
                  isActive ? "bg-emerald-500/10 text-emerald-600" : "text-zinc-600 dark:text-zinc-400"
                )
              }
            >
              <Icon className="h-3.5 w-3.5" />{label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
      <footer className="border-t border-zinc-200 dark:border-zinc-800 py-4 text-center text-xs text-zinc-500">
        End-to-end AI pipeline · W–C refractory campaign v1 · {new Date().getFullYear()}
      </footer>
    </div>
  );
}
