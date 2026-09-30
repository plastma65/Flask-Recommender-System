import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Front-end của đề tài Recommendation System.
// Mọi request /api/... được proxy sang FastAPI backend đang chạy ở cổng 8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
