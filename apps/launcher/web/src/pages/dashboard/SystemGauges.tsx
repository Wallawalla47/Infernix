import { useMemo } from "react";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  CircuitBoard,
  Cpu,
  HardDrive,
  MemoryStick,
  Monitor,
  Thermometer,
  Zap,
} from "lucide-react";

import { Sparkline, type SparkPoint } from "@/components/charts";
import { GaugeCard, accentZones, severityZones, useGrowingMax } from "@/components/Gauge";
import { Hint } from "@/components/ui/tooltip";
import type { CpuTempSource, GpuSample, SystemSample } from "@/lib/api-types";
import { useLive } from "@/lib/events";
import { formatByteRate, formatGiB } from "@/lib/format";

const GiB = 1024 ** 3;

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

/** Host and GPU readings as dials, each with its ten-minute history beneath. */
export function SystemGauges() {
  const system = useLive((live) => live.system);
  const history = useLive((live) => live.history);
  const launcher = useLive((live) => live.launcher);
  const info = launcher?.system ?? null;

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

  const powerMax = powerLimit !== null && powerLimit > 0 ? powerLimit : 600;
  const rxMax = useGrowingMax(series.rx.peak, 32);
  const txMax = useGrowingMax(series.tx.peak, 32);
  const server = system?.server ?? null;

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
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-[repeat(auto-fit,minmax(170px,1fr))]">
        <GaugeCard
          title="CPU"
          icon={Cpu}
          value={num(system?.cpu.usage)}
          max={100}
          unit="%"
          format={whole}
          zones={accentZones(100)}
          peak={series.cpu.peak}
          sub={info ? <span>{`${info.cores} cores · ${info.threads} threads`}</span> : null}
          footer={<Spark series={series.cpu} label="CPU" format={(v) => `${whole(v)}%`} domain={[0, 100]} />}
        />
        <GaugeCard
          title="CPU temperature"
          icon={Thermometer}
          hint={cpuHint}
          value={cpuTemp}
          max={100}
          unit="°C"
          format={whole}
          zones={severityZones(100, 80, 95)}
          valueColor="zone"
          peak={series.cpuTemp.peak}
          peakLabel="hottest"
          sub={
            <Hint label={cpuHint}>
              <span className="cursor-default underline decoration-dotted underline-offset-4">
                {cpuTemp === null ? "no sensor" : cpuSource === "thermal-zone" ? "motherboard zone" : "package"}
              </span>
            </Hint>
          }
          footer={<Spark series={series.cpuTemp} label="CPU temperature" format={(v) => `${whole(v)} °C`} domain={[0, 100]} />}
        />
        <GaugeCard
          title="Memory"
          icon={MemoryStick}
          value={percentOf(ramUsed, ramTotal)}
          max={100}
          unit="%"
          format={whole}
          zones={severityZones(100, 85, 95)}
          valueColor="zone"
          peak={series.ram.peak}
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
              zones={accentZones(100)}
              peak={series.gpu.peak}
              sub={num(gpu?.sm_clock_mhz) !== null ? <span>{`${num(gpu?.sm_clock_mhz)} MHz`}</span> : null}
              footer={<Spark series={series.gpu} label="GPU" format={(v) => `${whole(v)}%`} domain={[0, 100]} />}
            />
            <GaugeCard
              title="GPU temperature"
              icon={Thermometer}
              hint={num(gpu?.temp_c) === null ? "The driver reports no GPU temperature" : "GPU core sensor via NVML"}
              value={num(gpu?.temp_c)}
              max={100}
              unit="°C"
              format={whole}
              zones={severityZones(100, 70, 85)}
              valueColor="zone"
              peak={series.gpuTemp.peak}
              peakLabel="hottest"
              sub={num(gpu?.fan_pct) !== null ? <span>{`fan ${whole(num(gpu?.fan_pct) ?? 0)}%`}</span> : null}
              footer={<Spark series={series.gpuTemp} label="GPU temperature" format={(v) => `${whole(v)} °C`} domain={[0, 100]} />}
            />
            <GaugeCard
              title="VRAM"
              icon={HardDrive}
              value={percentOf(vramUsed, vramTotal)}
              max={100}
              unit="%"
              format={whole}
              zones={accentZones(100)}
              peak={series.vram.peak}
              sub={
                <span>
                  {vramUsed !== null && vramTotal !== null ? `${(vramUsed / GiB).toFixed(1)} of ${formatGiB(vramTotal, 0)}` : "—"}
                </span>
              }
              footer={<Spark series={series.vram} label="VRAM in use" format={(v) => `${whole(v)}%`} domain={[0, 100]} />}
            />
            <GaugeCard
              title="GPU power"
              icon={Zap}
              value={power}
              max={powerMax}
              unit="W"
              format={whole}
              zones={accentZones(powerMax)}
              peak={series.power.peak}
              sub={<span>{powerLimit !== null ? `of ${whole(powerLimit)} W` : "no power limit reported"}</span>}
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
              zones={accentZones(rxMax)}
              peak={series.rx.peak}
              sub={<span>{link ?? formatByteRate(rx)}</span>}
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
              zones={accentZones(txMax)}
              peak={series.tx.peak}
              sub={<span>{link ?? formatByteRate(tx)}</span>}
              footer={<Spark series={series.tx} label="GPU to host" format={(v) => `${v.toFixed(2)} GB/s`} />}
            />
          </>
        ) : null}
      </div>
    </section>
  );
}
