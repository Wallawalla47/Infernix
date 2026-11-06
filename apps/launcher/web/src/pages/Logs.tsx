import { useSearchParams } from "react-router";
import { FolderOpen, ListTree, Terminal } from "lucide-react";

import { ConsolePanel } from "@/components/ConsolePanel";
import { EventLogPanel } from "@/components/EventLogPanel";
import { PageHeader } from "@/components/common";
import { StateBadge } from "@/components/StateBadge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Hint } from "@/components/ui/tooltip";
import { useLive } from "@/lib/events";
import { useOpenLogFolder } from "@/lib/hooks";

type LogTab = "console" | "events";

/** The server console and the launcher's event log; `?tab=events` opens the second. */
export function LogsPage() {
  const instance = useLive((live) => live.instance);
  const openFolder = useOpenLogFolder();
  const [params, setParams] = useSearchParams();
  const tab: LogTab = params.get("tab") === "events" ? "events" : "console";

  return (
    <div className="flex h-[calc(100dvh-7.5rem)] min-h-[28rem] flex-col gap-4">
      <PageHeader
        title="Logs"
        description={
          tab === "events"
            ? "The launcher's own events: model loads and stops, the watchdog, downloads, settings changes and one record per API request."
            : instance?.alias
              ? `infernix-serve for ${instance.alias}${instance.pid ? ` · pid ${instance.pid}` : ""}; launcher events are marked in the left margin.`
              : "The managed infernix-serve's console and the launcher's own events."
        }
        actions={
          tab === "events" ? (
            <Hint label="Opens it in the file manager of the computer running the launcher">
              <Button variant="outline" size="sm" disabled={openFolder.isPending} onClick={() => openFolder.mutate()}>
                <FolderOpen />
                Open log folder
              </Button>
            </Hint>
          ) : instance ? (
            <StateBadge state={instance.state} />
          ) : null
        }
      />
      <Tabs
        value={tab}
        onValueChange={(value) => setParams(value === "events" ? { tab: "events" } : {}, { replace: true })}
        className="min-h-0 flex-1"
      >
        <TabsList>
          <TabsTrigger value="console">
            <Terminal />
            Server console
          </TabsTrigger>
          <TabsTrigger value="events">
            <ListTree />
            Launcher events
          </TabsTrigger>
        </TabsList>
        <TabsContent value="console" className="flex min-h-0 flex-col">
          <ConsolePanel className="min-h-0 flex-1" heightClass="flex-1 min-h-0" />
        </TabsContent>
        <TabsContent value="events" className="flex min-h-0 flex-col">
          <EventLogPanel className="min-h-0 flex-1" heightClass="flex-1 min-h-0" />
        </TabsContent>
      </Tabs>
    </div>
  );
}
