// The dashboard's memory bar (MemoryBreakdown's poolView): the `reserved` part, VRAM the server's plan keeps but has
// not allocated, is free on the card. It must not shrink "Infernix other" (it is no part of the process's memory) and
// must come out of "Free" (which then sits near the server's headroom), never past the free space the card reports.
// Run by `npm test` (scripts/test.mjs).
import { poolView, type Segment } from "@/pages/dashboard/MemoryBreakdown";

const part = (id: string, bytes: number, unallocated = false): Segment => ({
  id,
  kind: "part",
  label: id,
  bytes,
  color: "red",
  striped: false,
  unallocated,
  what: "",
});

const text = { infernix: "", others: "", free: "" };

function bytesOf(view: ReturnType<typeof poolView>, id: string): number | undefined {
  return view.segments.find((segment) => segment.id === id)?.bytes;
}

export function run(): string[] {
  const failures: string[] = [];
  const check = (ok: boolean, what: string) => {
    if (!ok) failures.push(what);
  };

  // A card of 10,000 bytes with 400 free: the process holds 900 (800 in named parts), other programs 8,700, and the
  // plan keeps 150 of the free 400.
  const kept = poolView([part("weights", 800), part("reserved", 150, true)], 10_000, 9_600, 900, text);
  check(bytesOf(kept, "infernix-other") === 100, `Infernix other ${bytesOf(kept, "infernix-other")}, expected 100`);
  check(bytesOf(kept, "other-processes") === 8_700, `Other processes ${bytesOf(kept, "other-processes")}, expected 8700`);
  check(bytesOf(kept, "reserved") === 150, `Reserved ${bytesOf(kept, "reserved")}, expected 150`);
  check(bytesOf(kept, "free") === 250, `Free ${bytesOf(kept, "free")}, expected 250`);
  check(kept.infernix === 1_050, `Infernix ${kept.infernix}, expected 1050 (process and what its plan keeps)`);
  check(kept.used === 9_600, `in use ${kept.used}, expected 9600 (the card's reading: the reserved share is free)`);
  const sum = kept.segments.reduce((acc, segment) => acc + segment.bytes, 0);
  check(sum === 10_000, `segments sum to ${sum}, expected the card's 10000`);

  // A reading that lags (the plan keeps more than is free now): the reserved share is cut to the free space.
  const lagging = poolView([part("weights", 800), part("reserved", 600, true)], 10_000, 9_600, 900, text);
  check(bytesOf(lagging, "reserved") === 400, `lagging Reserved ${bytesOf(lagging, "reserved")}, expected 400`);
  check(bytesOf(lagging, "free") === 0, `lagging Free ${bytesOf(lagging, "free")}, expected 0`);

  return failures;
}
