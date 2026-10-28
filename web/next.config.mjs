/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // /api/* is proxied to the FastAPI backend by app/api/[...path]/route.ts at request time
  // (see that file for why a build-time `rewrites()` entry is not used).
};
export default nextConfig;
