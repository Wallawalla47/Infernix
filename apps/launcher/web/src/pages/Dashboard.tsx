import { ActivityCard } from "@/pages/dashboard/ActivityCard";
import { DashboardConsole } from "@/pages/dashboard/DashboardConsole";
import { InferenceGauges } from "@/pages/dashboard/InferenceGauges";
import { InferenceStats } from "@/pages/dashboard/InferenceStats";
import { RecentRequests } from "@/pages/dashboard/RecentRequests";
import { RequestCharts } from "@/pages/dashboard/RequestCharts";
import { StatusHero } from "@/pages/dashboard/StatusHero";
import { SystemGauges } from "@/pages/dashboard/SystemGauges";

/** Top to bottom: the loaded model, the inference dials, the system dials, the statistics and
 * request detail, and the server console in a resizable box at the bottom. */
export function DashboardPage() {
  return (
    <div className="flex flex-col gap-6">
      <StatusHero />
      <InferenceGauges />
      <SystemGauges />

      <div className="grid gap-5 xl:grid-cols-5">
        <InferenceStats className="xl:col-span-2" />
        <RequestCharts className="xl:col-span-3" />
      </div>
      <div className="grid gap-5 xl:grid-cols-5">
        <RecentRequests className="xl:col-span-3" />
        <ActivityCard className="xl:col-span-2" />
      </div>

      <DashboardConsole />
    </div>
  );
}
