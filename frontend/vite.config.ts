import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard talks to FastAPI over absolute URLs taken from
// VITE_API_BASE_URL, so no dev proxy is used: the browser's real origin
// (http://localhost:5173) must reach the API, which is what exercises the
// backend's explicit CORS allow-list during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
