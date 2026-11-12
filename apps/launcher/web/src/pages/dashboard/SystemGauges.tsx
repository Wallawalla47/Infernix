import { useMemo } from "react";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  ChevronDown,
  ChevronUp,
  CircuitBoard,
  Cpu,
  HardDrive,
  MemoryStick,
  Monitor,
  Thermometer,
  Zap,
} from "lucide-react";

import { Sparkline, type SparkPoint } from "@/components/charts";
import { GaugeCard, gaugeTone, useGrowingMax } from "@/components/Gauge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Hint } from "@/components/ui/tooltip";
import type { CpuTempSource, GpuSample, SystemSample } from "@/lib/api-types";
import { useLive } from "@/lib/events";
import { formatByteRate, formatGiB } from "@/lib/format";
import { useStoredFlag } from "@/lib/hooks";

const GiB = 1024 ** 3;

// Limits of the dials that have them: a reading at or above the first is "High", the second "Critical".
const CPU_TEMP_LIMITS = { warnAt: 85, criticalAt: 95 };
const GPU_TEMP_LIMITS = { warnAt: 80, criticalAt: 90 };
// Memory is judged by what is left, not by how full it is: models with offloaded experts fill the
// RAM on purpose. 2 GiB available is where the Engine starts giving RAM back and the launcher's
// default floor for a load.
const MEMORY_WARN_FREE = 4 * GiB;
const MEMORY_CRITICAL_FREE = 2 * GiB;

function memoryLimits(total: number | null): { warnAt?: number; criticalAt?: number } {
  if (total === null || total <= MEMORY_WARN_FREE) return {};
  return { warnAt: (1 - MEMORY_WARN_FREE / total) * 100, criticalAt: (1 - MEMORY_CRITICAL_FREE / total) * 100 };
}

const CPU_SOURCE_TEXT: Record<CpuTempSource, string> = {
  hwinfo: "CPU package sensor via HWiNFO",
  coretemp: "CPU package sensor via Core Temp",
  "thermal-zone":
    "Windows ACPI thermal zone — a motherboard sensor, not the CPU package; running HWiNFO or Core Temp gives the package temperature",
  hwmon: "Linux hwmon",
};

const NO_CPU_SENSOR =
  "This PC exposes no temperature sensor to Windows; HWiNFO or Core Temp, if running, provide one";

/** A number NVML or a sensor may not have read (null or missing). */
function num(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function firstGpu(sample: SystemSample): GpuSample | null {
  return sample.gpus.length > 0 ? sample.gpus[0] : null;
}

function percentOf(part: number | null, whole: number | null): number | null {
  return part !== null && whole !== null && whole > 0 ? (part / whole) * 100 : null;
}

interface Series {
  points: SparkPoint[];
  peak: number | null;
}

function track(samples: readonly SystemSample[], pick: (sample: SystemSample) => number | null): Series {
  let peak: number | null = null;
  const points = samples.map((sample) => {
    const v = pick(sample);
    if (v !== null && (peak === null || v > peak)) peak = v;
    return { t: sample.t, v };
  });
  return { points, peak };
}

const whole = (value: number) => value.toFixed(0);
const oneDecimal = (value: number) => value.toFixed(1);

function Spark({ series, label, format, domain }: { series: Series; label: string; format: (v: number) => string; domain?: [number, number] }) {
  return <Sparkline data={series.points} label={label} format={format} domain={domain} height={30} />;
}

/** The parts of a dial's sub-line that have a value, joined by a middle dot. */
function facts(...parts: (string | null)[]): string | null {
  const present = parts.filter((part): part is string => part !== null);
  return present.length > 0 ? present.join(" · ") : null;
}

const SHOW_ALL_KEY = "infernix.dashboard.system-all";

/** Host and GPU readings as dials, each with its ten-minute history beneath: processor, memory,
 * GPU and VRAM use always; the temperatures, GPU power and PCIe traffic with "All gauges" (this
 * browser remembers the choice). A temperature past its limit is named in the header either way. */
export function SystemGauges() {
  const system = useLive((live) => live.system);
  const history = useLive((live) => live.history);
  const launcher = useLive((live) => live.launcher);
  const info = launcher?.system ?? null;
  const [showAll, setShowAll] = useStoredFlag(SHOW_ALL_KEY, false);

  const series = useMemo(
    () => ({
      cpu: track(history, (s) => num(s.cpu.usage)),
      cpuTemp: track(history, (s) => num(s.cpu.temp_c)),
      ram: track(history, (s) => percentOf(num(s.ram.used), num(s.ram.total))),
      gpu: track(history, (s) => num(firstGpu(s)?.usage)),
      gpuTemp: track(history, (s) => num(firstGpu(s)?.temp_c)),
      vram: track(history, (s) => {
        const gpu = firstGpu(s);
        return percentOf(num(gpu?.vram_used), num(gpu?.vram_total));
      }),
      power: track(history, (s) => num(firstGpu(s)?.power_w)),
      rx: track(history, (s) => {
        const rx = num(firstGpu(s)?.pcie_rx_bps);
        return rx === null ? null : rx / 1e9;
      }),
      tx: track(history, (s) => {
        const tx = num(firstGpu(s)?.pcie_tx_bps);
        return tx === null ? null : tx / 1e9;
      }),
    }),
    [history],
  );

  const gpu = system ? firstGpu(system) : null;
  // Keep the GPU dials' places until the first sample; once samples show no NVIDIA GPU, leave them out.
  const showGpu = system === null || gpu !== null;

  const cpuTemp = num(system?.cpu.temp_c);
  const cpuSource = system?.cpu.temp_source ?? info?.cpu_temp_source ?? null;
  const cpuHint = cpuTemp === null || cpuSource === null ? NO_CPU_SENSOR : CPU_SOURCE_TEXT[cpuSource];
  const gpuTemp = num(gpu?.temp_c);

  const ramUsed = num(system?.ram.used);
  const ramTotal = num(system?.ram.total);
  const vramUsed = num(gpu?.vram_used);
  const vramTotal = num(gpu?.vram_total);
  const power = num(gpu?.power_w);
  const powerLimit = num(gpu?.power_limit_w);
  const rx = num(gpu?.pcie_rx_bps);
  const tx = num(gpu?.pcie_tx_bps);
  const pcieGen = num(gpu?.pcie_gen);
  const pcieWidth = num(gpu?.pcie_width);
  const link = pcieGen !== null && pcieWidth !== null ? `Gen${pcieGen} x${pcieWidth}` : null;
  const fan = num(gpu?.fan_pct);
  const clock = num(gpu?.sm_clock_mhz);

  const powerMax = powerLimit !== null && powerLimit > 0 ? powerLimit : 600;
  const rxMax = useGrowingMax(series.rx.peak, 32);
  const txMax = useGrowingMax(series.tx.peak, 32);
  const server = system?.server ?? null;

  // The temperature dials are among the ones "All gauges" adds: a hot reading shows in the header too.
  const hot: string[] = [];
  if (gaugeTone(cpuTemp, CPU_TEMP_LIMITS.warnAt, CPU_TEMP_LIMITS.criticalAt) !== "normal") hot.push(`CPU ${whole(cpuTemp ?? 0)} °C`);
  if (gaugeTone(gpuTemp, GPU_TEMP_LIMITS.warnAt, GPU_TEMP_LIMITS.criticalAt) !== "normal") hot.push(`GPU ${whole(gpuTemp ?? 0)} °C`);

  return (
    <section className="flex flex-col gap-3" aria-labelledby="system-gauges">
      <div className="flex flex-wrap items-center gap-2">
        <h2 id="system-gauges" className="flex items-center gap-2 text-sm font-semibold">
          <Monitor className="size-4 text-muted-foreground" />
          System
        </h2>
        <span className="min-w-0 truncate text-xs text-muted-foreground">
          {info ? `${info.cpu_name}${gpu ? ` · ${gpu.name ?? "GPU"}` : ""}` : "Reading the hardware…"}
        </span>
        <div className="ml-auto flex items-center gap-1.5">
          {hot.map((text) => (
            <Badge key={text} variant="warning">
              <Thermometer />
              {text}
            </Badge>
          ))}
          <Hint label={showAll ? "Show processor, memory, GPU and VRAM only" : "Also show the temperatures, GPU power and PCIe traffic"}>
            <Button variant="ghost" size="xs" aria-expanded={showAll} onClick={() => setShowAll(!showAll)}>
              {showAll ? <ChevronUp /> : <ChevronDown />}
              {showAll ? "Fewer gauges" : "All gauges"}
            </Button>
          </Hint>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-[repeat(auto-fill,minmax(170px,1fr))]">
        <GaugeCard
          title="CPU"
          icon={Cpu}
          value={num(system?.cpu.usage)}
          max={100}
          unit="%"
          format={whole}
          sub={
            <span>{facts(cpuTemp !== null ? `${whole(cpuTemp)} °C` : null, info ? `${info.cores} cores · ${info.threads} threads` : null) ?? "—"}</span>
          }
          footer={<Spark series={series.cpu} label="CPU" format={(v) => `${whole(v)}%`} domain={[0, 100]} />}
        />
        <GaugeCard
          title="Memory"
          icon={MemoryStick}
          value={percentOf(ramUsed, ramTotal)}
          max={100}
          unit="%"
          format={whole}
          {...memoryLimits(ramTotal)}
          sub={
            <>
              <span>{ramUsed !== null && ramTotal !== null ? `${(ramUsed / GiB).toFixed(1)} of ${formatGiB(ramTotal, 0)}` : "—"}</span>
              {server ? <span>server {formatGiB(server.working_set)}</span> : null}
            </>
          }
          footer={<Spark series={series.ram} label="Memory in use" format={(v) => `${whole(v)}%`} domain={[0, 100]} />}
        />
        {showGpu ? (
          <>
            <GaugeCard
              title="GPU"
              icon={CircuitBoard}
              value={num(gpu?.usage)}
              max={100}
              unit="%"
              format={whole}
              sub={<span>{facts(gpuTemp !== null ? `${whole(gpuTemp)} °C` : null, power !== null ? `${whole(power)} W` : null) ?? "—"}</span>}
              footer={<Spark series={series.gpu} label="GPU" format={(v) => `${whole(v)}%`} domain={[0, 100]} />}
            />
            <GaugeCard
              title="VRAM"
              icon={HardDrive}
              value={percentOf(vramUsed, vramTotal)}
              max={100}
              unit="%"
              format={whole}
              sub={
                <span>
                  {vramUsed !== null && vramTotal !== null ? `${(vramUsed / GiB).toFixed(1)} of ${formatGiB(vramTotal, 0)}` : "—"}
                </span>
              }
              footer={<Spark series={series.vram} label="VRAM in use" format={(v) => `${whole(v)}%`} domain={[0, 100]} />}
            />
          </>
        ) : null}
        {showAll ? (
          <GaugeCard
            title="CPU temperature"
            icon={Thermometer}
            hint={cpuHint}
            value={cpuTemp}
            max={100}
            unit="°C"
            format={whole}
            {...CPU_TEMP_LIMITS}
            sub={
              <>
                <Hint label={cpuHint}>
                  <span className="cursor-default underline decoration-dotted underline-offset-4">
                    {cpuTemp === null ? "no sensor" : cpuSource === "thermal-zone" ? "motherboard zone" : "package"}
                  </span>
                </Hint>
                {series.cpuTemp.peak !== null ? <span>peak {whole(series.cpuTemp.peak)} °C</span> : null}
              </>
            }
            footer={<Spark series={series.cpuTemp} label="CPU temperature" format={(v) => `${whole(v)} °C`} domain={[0, 100]} />}
          />
        ) : null}
        {showAll && showGpu ? (
          <>
            <GaugeCard
              title="GPU temperature"
              icon={Thermometer}
              hint={gpuTemp === null ? "The driver reports no GPU temperature" : "GPU core sensor via NVML"}
              value={gpuTemp}
              max={100}
              unit="°C"
              format={whole}
              {...GPU_TEMP_LIMITS}
              sub={
                <span>
                  {facts(fan !== null ? `fan ${whole(fan)}%` : null, series.gpuTemp.peak !== null ? `peak ${whole(series.gpuTemp.peak)} °C` : null) ?? "—"}
                </span>
              }
              footer={<Spark series={series.gpuTemp} label="GPU temperature" format={(v) => `${whole(v)} °C`} domain={[0, 100]} />}
            />
            <GaugeCard
              title="GPU power"
              icon={Zap}
              value={power}
              max={powerMax}
              unit="W"
              format={whole}
              sub={
                <span>
                  {facts(powerLimit !== null ? `of ${whole(powerLimit)} W` : "no power limit reported", clock !== null ? `${clock} MHz` : null)}
                </span>
              }
              footer={<Spark series={series.power} label="GPU power" format={(v) => `${whole(v)} W`} domain={[0, powerMax]} />}
            />
            <GaugeCard
              title="PCIe to GPU"
              icon={ArrowDownToLine}
              hint="Host-to-GPU traffic: offloaded experts, n-gram rows, prompt uploads."
              value={rx === null ? null : rx / 1e9}
              max={rxMax}
              unit="GB/s"
              format={oneDecimal}
              sub={<span>{facts(link ?? formatByteRate(rx), series.rx.peak !== null ? `peak ${series.rx.peak.toFixed(1)}` : null)}</span>}
              footer={<Spark series={series.rx} label="Host to GPU" format={(v) => `${v.toFixed(2)} GB/s`} />}
            />
            <GaugeCard
              title="PCIe from GPU"
              icon={ArrowUpFromLine}
              hint="GPU-to-host traffic: results, CPU-served expert inputs, cache writes."
              value={tx === null ? null : tx / 1e9}
              max={txMax}
              unit="GB/s"
              format={oneDecimal}
              sub={<span>{facts(link ?? formatByteRate(tx), series.tx.peak !== null ? `peak ${series.tx.peak.toFixed(1)}` : null)}</span>}
              footer={<Spark series={series.tx} label="GPU to host" format={(v) => `${v.toFixed(2)} GB/s`} />}
            />
          </>
        ) : null}
      </div>
    </section>
  );
}
