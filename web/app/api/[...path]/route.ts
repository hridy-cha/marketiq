// Runtime proxy to the FastAPI backend.
//
// A Next.js `rewrites()` entry is evaluated at build time, so a destination taken from an
// environment variable gets baked into the image; in Docker that meant the dashboard kept calling
// 127.0.0.1 (itself) instead of the `api` container. This route handler reads the backend URL on
// every request, so the same image works locally (`marketiq serve` on 127.0.0.1:8000) and in
// Compose (`API_URL=http://api:8000`).

import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function backendUrl(): string {
  return (process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
}

async function proxy(req: NextRequest, { params }: { params: { path: string[] } }) {
  const target = `${backendUrl()}/api/${params.path.join("/")}${req.nextUrl.search}`;
  const headers: Record<string, string> = {};
  const ct = req.headers.get("content-type");
  if (ct) headers["content-type"] = ct;
  const init: RequestInit = { method: req.method, headers, cache: "no-store" };
  if (req.method !== "GET" && req.method !== "HEAD") init.body = await req.text();
  try {
    const res = await fetch(target, init);
    const body = await res.arrayBuffer();
    return new NextResponse(body, {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch (err) {
    const code = (err as { cause?: { code?: string } })?.cause?.code ?? (err as Error).message;
    return NextResponse.json(
      {
        detail:
          `The MarketIQ backend is not reachable at ${backendUrl()} (${code}). ` +
          "Start it with MarketIQ.command / `marketiq serve`, or check the `api` container in Docker Compose.",
      },
      { status: 502 },
    );
  }
}

export { proxy as GET, proxy as POST, proxy as PUT, proxy as DELETE, proxy as PATCH };
