/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies /api to the demo API on 127.0.0.1:8000 (the browser never calls Orthanc).
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { environment: "jsdom", globals: true },
});
