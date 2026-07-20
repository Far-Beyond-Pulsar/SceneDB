import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        github: {
          bg: "#0d1117",
          sidebar: "#161b22",
          card: "#1c2128",
          border: "#30363d",
          "border-muted": "#21262d",
          text: "#f0f6fc",
          "text-secondary": "#9198a1",
          "text-muted": "#6e7681",
          accent: "#58a6ff",
          green: "#3fb950",
          yellow: "#d29922",
          red: "#f85149",
          orange: "#f78166",
          blue: "#79c0ff",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "SF Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
