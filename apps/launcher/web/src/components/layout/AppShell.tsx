import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router";
import {
  Boxes,
  ChartColumnBig,
  CloudDownload,
  LayoutDashboard,
  Menu,
  MessageSquare,
  Monitor,
  Moon,
  Pin,
  Plug,
  ScrollText,
  Settings as SettingsIcon,
  Sun,
  WifiOff,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { SessionMenu } from "@/components/admin/SignIn";
import { CopyButton } from "@/components/CopyButton";
import { IncidentIcon, IncidentLine } from "@/components/IncidentBanner";
import { Logo } from "@/components/Logo";
import { StateDot } from "@/components/StateBadge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { Hint } from "@/components/ui/tooltip";
import { useLive } from "@/lib/events";
import { STATE_LABELS, formatTokS } from "@/lib/format";
import { useEndpoints, useMediaQuery, useNow } from "@/lib/hooks";
import { useTheme, type Theme } from "@/lib/theme";
import { cn } from "@/lib/utils";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
}

const NAV: NavItem[] = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/discover", label: "Discover", icon: CloudDownload },
  { to: "/connect", label: "Connect", icon: Plug },
  { to: "/usage", label: "Usage", icon: ChartColumnBig },
  { to: "/logs", label: "Logs", icon: ScrollText },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

function pageTitle(pathname: string): string {
  if (pathname.startsWith("/models/")) return "Configure model";
  const item = NAV.find((nav) => (nav.end ? pathname === nav.to : pathname.startsWith(nav.to)));
  return item?.label ?? "Infernix";
}

function NavList({ compact = false, onNavigate }: { compact?: boolean; onNavigate?: () => void }) {
  return (
    <nav className={cn("flex flex-col gap-0.5", compact ? "items-center" : "px-2")} aria-label="Main">
      {NAV.map((item) => {
        const link = (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            onClick={onNavigate}
            aria-label={compact ? item.label : undefined}
            className={({ isActive }) =>
              cn(
                "group flex items-center rounded-md text-sm font-medium transition-colors",
                compact ? "size-10 justify-center" : "gap-2.5 px-2.5 py-2",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground/75 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground",
              )
            }
          >
            {({ isActive }) => (
              <>
                <item.icon className={cn("size-4", isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground")} />
                {compact ? null : item.label}
              </>
            )}
          </NavLink>
        );
        // The tooltip trigger wraps the link: Radix's Slot would join NavLink's className function
        // into a string.
        return compact ? (
          <Hint key={item.to} label={item.label} side="right">
            <span className="flex">{link}</span>
          </Hint>
        ) : (
          link
        );
      })}
    </nav>
  );
}

/** The status card's state as one dot, for the compact sidebar; it links to the dashboard. */
function LiveStatusDot() {
  const instance = useLive((live) => live.instance);
  const state = instance?.state ?? "idle";
  const name = instance?.alias ?? instance?.name ?? null;
  return (
    <Hint label={name ? `${STATE_LABELS[state]} · ${name}` : "No model loaded"} side="right">
      <NavLink
        to="/"
        aria-label={name ? `${STATE_LABELS[state]}: ${name}` : "No model loaded"}
        className="flex size-8 items-center justify-center rounded-md hover:bg-sidebar-accent/60"
      >
        <StateDot state={state} />
      </NavLink>
    </Hint>
  );
}

function LiveStatus() {
  const instance = useLive((live) => live.instance);
  const metrics = useLive((live) => live.metrics);
  const launcher = useLive((live) => live.launcher);
  const state = instance?.state ?? "idle";
  const rate = metrics?.live.tok_s ?? null;
  const generating = metrics?.live.state === "generating" || metrics?.live.state === "reading";
  return (
    <div className="mx-2 rounded-lg border bg-card/60 p-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <StateDot state={state} />
        <span>{STATE_LABELS[state]}</span>
        {instance && instance.waiting > 0 ? <span className="ml-auto tabular-nums">{instance.waiting} waiting</span> : null}
      </div>
      <div className="mt-1.5 truncate text-sm font-medium" title={instance?.name ?? undefined}>
        {instance?.alias ?? instance?.name ?? "No model loaded"}
      </div>
      <div className="mt-1 text-xs text-muted-foreground tabular-nums">
        {state === "ready" ? (generating && rate !== null ? formatTokS(rate) : "Idle, ready for requests") : instance?.phase || " "}
      </div>
      <IncidentLine />
      {launcher ? <div className="mt-2 text-[10px] text-muted-foreground/70">Launcher {launcher.version}</div> : null}
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col gap-4 py-4">
      <div className="px-4">
        <Logo />
      </div>
      <NavList onNavigate={onNavigate} />
      <div className="mt-auto">
        <LiveStatus />
      </div>
    </div>
  );
}

/** The sidebar as an icon rail, for windows between the phone layout and lg (half a 1920-pixel
 * screen, for one), so the page keeps its width. */
function CompactSidebarContent() {
  return (
    <div className="flex h-full flex-col items-center gap-4 py-4">
      <Logo compact />
      <NavList compact />
      <div className="mt-auto flex flex-col items-center gap-1">
        <IncidentIcon />
        <LiveStatusDot />
      </div>
    </div>
  );
}

function ThemeMenu() {
  const { theme, setTheme, resolved } = useTheme();
  const Icon = theme === "system" ? Monitor : resolved === "dark" ? Moon : Sun;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label="Theme">
          <Icon />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>Theme</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={theme} onValueChange={(value) => setTheme(value as Theme)}>
          <DropdownMenuRadioItem value="dark">
            <Moon /> Dark
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="light">
            <Sun /> Light
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="system">
            <Monitor /> System
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function StatusPill() {
  const instance = useLive((live) => live.instance);
  const state = instance?.state ?? "idle";
  return (
    <NavLink
      to="/"
      className="hidden min-w-0 items-center gap-2 rounded-full border bg-card/60 px-3 py-1 text-xs sm:flex"
    >
      <StateDot state={state} />
      <span className="font-medium">{STATE_LABELS[state]}</span>
      {instance?.alias ? <span className="max-w-48 truncate text-muted-foreground">{instance.alias}</span> : null}
      {instance?.pinned && state !== "idle" ? (
        <Pin className="size-3 shrink-0 text-muted-foreground" aria-label="pinned: every request goes to this model" />
      ) : null}
      {instance?.pending_model ? <span className="text-muted-foreground">· switching</span> : null}
    </NavLink>
  );
}

const bootTime = Date.now();

function OfflineBanner() {
  const connected = useLive((live) => live.connected);
  const everConnected = useLive((live) => live.everConnected);
  const since = useLive((live) => live.disconnectedAt);
  const now = useNow(1000);
  // Give the first connection (and brief reconnects) a moment before alarming anyone.
  const show = !connected && (everConnected ? since !== null && now - since > 1500 : now - bootTime > 2500);
  if (!show) return null;
  return (
    <div className="flex items-center gap-2 border-b border-critical/40 bg-critical/10 px-4 py-2 text-sm md:px-6">
      <WifiOff className="size-4 shrink-0 text-critical" />
      <span className="font-medium">Launcher offline</span>
      <span className="text-muted-foreground">Reconnecting… the page updates again once infernix-launcher is reachable.</span>
    </div>
  );
}

export function AppShell({ children }: { children?: ReactNode }) {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const endpoints = useEndpoints();
  const title = pageTitle(location.pathname);
  const full = useMediaQuery("(min-width: 64rem)");

  useEffect(() => {
    document.title = `${title} · Infernix Launcher`;
  }, [title]);

  return (
    <div className="flex min-h-dvh w-full">
      <aside className={cn("sticky top-0 hidden h-dvh shrink-0 border-r bg-sidebar md:block", full ? "w-60" : "w-16")}>
        {full ? <SidebarContent /> : <CompactSidebarContent />}
      </aside>

      <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
        <SheetContent side="left" className="bg-sidebar p-0">
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <SheetDescription className="sr-only">Pages of the Infernix Launcher</SheetDescription>
          <SidebarContent onNavigate={() => setMenuOpen(false)} />
        </SheetContent>
      </Sheet>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b bg-background/80 px-4 backdrop-blur-md md:px-6">
          <Button variant="ghost" size="icon-sm" className="md:hidden" aria-label="Open the menu" onClick={() => setMenuOpen(true)}>
            <Menu />
          </Button>
          <div className="min-w-0 truncate text-sm font-semibold md:text-base">{title}</div>
          <div className="ml-auto flex min-w-0 items-center gap-2">
            <StatusPill />
            <div className="hidden items-center gap-1 rounded-md border bg-card/60 py-0.5 pr-0.5 pl-2.5 lg:flex">
              <span className="text-[11px] font-medium text-muted-foreground">Endpoint</span>
              <code className="max-w-56 truncate font-mono text-xs">{endpoints.openai}</code>
              <Hint label="Copy the OpenAI base URL">
                <span>
                  <CopyButton text={endpoints.openai} size="icon-xs" toastMessage="Endpoint copied" />
                </span>
              </Hint>
            </div>
            <ThemeMenu />
            <SessionMenu />
          </div>
        </header>
        <OfflineBanner />
        <main className="mx-auto w-full max-w-[1600px] min-w-0 flex-1 px-4 py-5 md:px-6 md:py-6">{children ?? <Outlet />}</main>
      </div>
    </div>
  );
}
