import type { ReactNode } from "react";
import { HashRouter, Link, Route, Routes } from "react-router";
import { Toaster } from "sonner";
import { Compass } from "lucide-react";

import { SignInPage } from "@/components/admin/SignIn";
import { EmptyState } from "@/components/common";
import { AppShell } from "@/components/layout/AppShell";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useLive } from "@/lib/events";
import { ThemeProvider, useTheme } from "@/lib/theme";
import { ChatPage } from "@/pages/Chat";
import { ConnectPage } from "@/pages/Connect";
import { DashboardPage } from "@/pages/Dashboard";
import { DiscoverPage } from "@/pages/Discover";
import { LogsPage } from "@/pages/Logs";
import { ModelConfigurePage } from "@/pages/ModelConfigure";
import { ModelsPage } from "@/pages/Models";
import { SettingsPage } from "@/pages/Settings";
import { UsagePage } from "@/pages/Usage";

function ThemedToaster() {
  const { resolved } = useTheme();
  return <Toaster theme={resolved} position="bottom-right" richColors closeButton duration={5000} />;
}

function NotFound() {
  return (
    <EmptyState
      icon={Compass}
      title="Nothing here"
      description="This page does not exist."
      actions={
        <Button asChild>
          <Link to="/">Back to the dashboard</Link>
        </Button>
      }
    />
  );
}

/** The whole page becomes the sign-in card while the launcher wants an admin key from this
 * browser; the route the user was on comes back after signing in. */
function SignInGate({ children }: { children: ReactNode }) {
  const required = useLive((live) => live.signInRequired);
  return required ? <SignInPage /> : children;
}

export function App() {
  return (
    <ThemeProvider>
      <TooltipProvider>
        <SignInGate>
          <HashRouter>
            <Routes>
              <Route element={<AppShell />}>
                <Route index element={<DashboardPage />} />
                <Route path="chat" element={<ChatPage />} />
                <Route path="models" element={<ModelsPage />} />
                <Route path="models/:key" element={<ModelConfigurePage />} />
                <Route path="discover" element={<DiscoverPage />} />
                <Route path="connect" element={<ConnectPage />} />
                <Route path="usage" element={<UsagePage />} />
                <Route path="logs" element={<LogsPage />} />
                <Route path="settings" element={<SettingsPage />} />
                <Route path="*" element={<NotFound />} />
              </Route>
            </Routes>
          </HashRouter>
        </SignInGate>
        <ThemedToaster />
      </TooltipProvider>
    </ThemeProvider>
  );
}
