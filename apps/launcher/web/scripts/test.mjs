// `npm test`: loads every tests/*.test.ts through Vite (the app's own config, so `@/` and TypeScript resolve as in
// the build) and runs its exported run(), which returns the failed checks. No browser and no extra dependency.
import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const tests = readdirSync(new URL("../tests", import.meta.url)).filter((name) => name.endsWith(".test.ts")).sort();
const server = await createServer({
  root,
  logLevel: "error",
  appType: "custom",
  server: { middlewareMode: true, hmr: false, ws: false },
  // No dependency pre-bundling: its scan of index.html runs in the background and is cut off by close().
  optimizeDeps: { noDiscovery: true, include: [] },
});
let failed = 0;
try {
  for (const name of tests) {
    const module = await server.ssrLoadModule(`/tests/${name}`);
    const failures = module.run();
    for (const failure of failures) console.error(`FAIL ${name}: ${failure}`);
    console.log(`${name}: ${failures.length === 0 ? "passed" : `${failures.length} failed`}`);
    failed += failures.length;
  }
} finally {
  await server.close();
}
if (tests.length === 0) {
  console.error("no tests/*.test.ts found");
  process.exit(1);
}
process.exit(failed === 0 ? 0 : 1);
