// Compresses the single-file build into the asset the launcher embeds
// (apps/launcher/assets/index.html.gz). The output is byte-identical for identical input.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { constants, gzipSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const input = resolve(here, "../dist/index.html");
const output = resolve(here, "../../assets/index.html.gz");

const html = readFileSync(input);
const gz = gzipSync(html, { level: constants.Z_BEST_COMPRESSION });
// zlib writes MTIME 0; pin the OS byte as well so every platform produces the same file.
gz[9] = 255;
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, gz);
console.log(`${input} -> ${output}: ${html.length} -> ${gz.length} bytes`);
