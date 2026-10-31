// Collects the licence of every runtime package the UI bundles (the dependency tree of
// package.json's "dependencies") into apps/launcher/assets/THIRD_PARTY_LICENSES.txt, which the
// Windows release ships beside the launcher. Sorted, so identical inputs give an identical file.
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const output = resolve(root, "../assets/THIRD_PARTY_LICENSES.txt");

// Node's resolution: the package's own node_modules, then each ancestor's.
function locate(name, from) {
  let dir = from;
  for (;;) {
    const candidate = join(dir, "node_modules", name);
    if (existsSync(join(candidate, "package.json"))) return candidate;
    const parent = dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

const seen = new Map();
function visit(name, from) {
  const dir = locate(name, from);
  if (!dir || seen.has(dir)) return;
  const manifest = JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
  const licenceFile = readdirSync(dir).find((file) => /^(licen[cs]e|copying)(\.|$)/i.test(file));
  seen.set(dir, {
    name: manifest.name,
    version: manifest.version,
    license: typeof manifest.license === "string" ? manifest.license : manifest.license?.type ?? "see text",
    text: licenceFile ? readFileSync(join(dir, licenceFile), "utf8").trim() : null,
  });
  for (const dependency of Object.keys(manifest.dependencies ?? {})) visit(dependency, dir);
}

const top = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
for (const dependency of Object.keys(top.dependencies ?? {})) visit(dependency, root);
// Tailwind is a build tool, but its generated CSS (preflight) is part of the page.
visit("tailwindcss", root);

const entries = [...seen.values()].sort((a, b) => `${a.name}@${a.version}`.localeCompare(`${b.name}@${b.version}`));
const missing = entries.filter((entry) => !entry.text).map((entry) => `${entry.name}@${entry.version}`);
let text =
  "Third-party licences of the Infernix Launcher web UI\n" +
  "These packages are bundled into the web UI that infernix-launcher serves.\n\n";
for (const entry of entries) {
  text += `${"=".repeat(78)}\n${entry.name} ${entry.version} (${entry.license})\n${"=".repeat(78)}\n`;
  text += `${entry.text ?? `Licence: ${entry.license} (the package ships no licence file).`}\n\n`;
}
writeFileSync(output, text.replace(/\r\n/g, "\n"));
console.log(`${entries.length} packages -> ${output}${missing.length ? `; no licence file: ${missing.join(", ")}` : ""}`);
