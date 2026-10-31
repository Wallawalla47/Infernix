import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// `npm run dev` serves the UI with hot reload and forwards the launcher's API (including the
// /launcher/api/events stream) to a running infernix-launcher.
const launcher = process.env.INFERNIX_LAUNCHER_URL ?? "http://127.0.0.1:1234";

export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  build: {
    target: "es2022",
    reportCompressedSize: false,
    chunkSizeWarningLimit: 4096,
  },
  server: {
    port: 5173,
    proxy: {
      "/launcher": { target: launcher, changeOrigin: true },
      "/v1": { target: launcher, changeOrigin: true },
      "/health": { target: launcher, changeOrigin: true },
    },
  },
});
