import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // DuckDB ships platform-specific native bindings.  Without marking it
  // external, Turbopack walks every `require('@duckdb/node-bindings-<os>-<arch>')`
  // branch and fails on the platforms the host isn't.
  serverExternalPackages: ["@duckdb/node-api", "@duckdb/node-bindings"],
  // Allow hot-reload requests from other hosts on the LAN so the app
  // can be previewed from another machine on the same wifi.
  allowedDevOrigins: ["192.168.1.192", "localhost"],
};

export default nextConfig;
