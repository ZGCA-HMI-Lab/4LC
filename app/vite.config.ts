import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import type { IncomingMessage } from "node:http";

const publicURL = process.env.APP_PUBLIC_URL
  ? new URL(process.env.APP_PUBLIC_URL)
  : null;
const base = publicURL?.pathname ?? "/";
const proxy = {
  [`${base}api`]: {
    target: "http://127.0.0.1:7860",
    rewrite: (path: string) => "/" + path.slice(base.length),
    ...(publicURL
      ? {
          headers: {
            host: publicURL.host,
            "x-forwarded-proto": publicURL.protocol.slice(0, -1),
          },
        }
      : {}),
  },
};

export default defineConfig({
  base: publicURL ? base : "./",
  plugins: [
    react(),
    {
      name: "forwarded-dev-prefix",
      configureServer(server) {
        if (!publicURL) return;
        // code-server strips its proxy prefix before forwarding HTTP and WebSocket traffic.
        const restorePrefix = (request: IncomingMessage) => {
          if (request.url && !request.url.startsWith(base))
            request.url = base.slice(0, -1) + request.url;
        };
        server.middlewares.use((request, _response, next) => {
          restorePrefix(request);
          next();
        });
        server.httpServer?.prependListener("upgrade", restorePrefix);
      },
    },
  ],
  server: {
    proxy,
    ...(publicURL
      ? {
          allowedHosts: [publicURL.hostname],
          hmr: {
            host: publicURL.hostname,
            protocol: publicURL.protocol === "https:" ? "wss" : "ws",
            clientPort:
              Number(publicURL.port) ||
              (publicURL.protocol === "https:" ? 443 : 80),
          },
        }
      : {}),
  },
  preview: { proxy },
});
