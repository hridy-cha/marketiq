import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 50: "#f5f6f8", 100: "#e8eaef", 200: "#cfd3dc", 300: "#a9b0bf", 400: "#7c8596", 500: "#5b6475", 600: "#454c5a", 700: "#343a45", 800: "#22262e", 900: "#15181d", 950: "#0d0f12" },
        bull: "#1f9d55",
        bear: "#d64545",
        watch: "#d9a21b",
        accent: "#3b6ff0",
      },
      fontFamily: { mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"] },
    },
  },
  plugins: [],
};
export default config;
