# Infernix Launcher web UI

The browser interface of `infernix-launcher`: React 19 + TypeScript, Vite, Tailwind CSS 4,
shadcn/ui-style components on Radix, Recharts, TanStack Query and react-router (hash routes).
`npm run build` produces one self-contained `dist/index.html` and packs it into
`../assets/index.html.gz`, which the C++ build embeds, so building the launcher needs no Node.js.

```bash
npm ci
npm run dev        # http://localhost:5173, API forwarded to a launcher on 127.0.0.1:1234
npm run build      # type-check, bundle, write ../assets/index.html.gz
```

`INFERNIX_LAUNCHER_URL` points the dev server at a launcher elsewhere. The API contract is
[`src/lib/api-types.ts`](src/lib/api-types.ts); the backend in `src/launcher` produces exactly those
shapes.

Layout: `src/components/ui` holds the primitives, `src/components` the shared pieces (charts,
console, folder picker), `src/pages` one module per route, `src/lib` the API client, the live event
store, formatting and launch-argument helpers. Charts follow the validated reference palette in
`src/index.css` (`--series-*`, `--chart-*`, `--status-*`).
