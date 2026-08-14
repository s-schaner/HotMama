import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev-time: the Python host runs on :8000; the PWA dev server proxies to it.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
