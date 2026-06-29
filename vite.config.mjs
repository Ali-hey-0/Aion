import { realpathSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const projectRoot = realpathSync(process.cwd());

export default defineConfig({
  root: projectRoot,
  cacheDir: "node_modules/.vite-aion",
  plugins: [react()],
  clearScreen: false,
  server: {
    host: "127.0.0.1",
    port: 7391,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
});
