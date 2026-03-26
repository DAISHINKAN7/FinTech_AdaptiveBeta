import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: "#060B16",
          900: "#0A0F1E",
          800: "#0F1629",
          700: "#141D3A",
          600: "#1A2550",
        },
        teal: {
          500: "#0F7A56",
          400: "#1D9E75",
          300: "#5DCAA5",
          200: "#9FE1CB",
          100: "#D0F5EA",
        },
        purple: {
          500: "#5750C4",
          400: "#7F77DD",
          300: "#AFA9EC",
          200: "#CECBF6",
          100: "#ECEAFD",
        },
        amber: {
          500: "#C47A0A",
          400: "#EF9F27",
          300: "#FAC775",
          200: "#FDE4B3",
        },
        coral: {
          500: "#A83A18",
          400: "#D85A30",
          300: "#F0997B",
          200: "#FACCBD",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "system-ui", "sans-serif"],
        body: ["var(--font-body)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      animation: {
        "draw-line": "drawLine 2s ease-in-out forwards",
        "count-up": "countUp 1.5s ease-out forwards",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-up": "fadeUp 0.6s ease-out forwards",
        "fade-in": "fadeIn 0.4s ease-out forwards",
        "glow": "glow 2s ease-in-out infinite alternate",
        "float": "float 3s ease-in-out infinite",
      },
      keyframes: {
        drawLine: {
          "0%": { strokeDashoffset: "1000" },
          "100%": { strokeDashoffset: "0" },
        },
        countUp: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        glow: {
          "0%": { boxShadow: "0 0 20px rgba(29,158,117,0.3)" },
          "100%": { boxShadow: "0 0 40px rgba(29,158,117,0.6)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" },
        },
      },
      backgroundImage: {
        "grid-pattern":
          "linear-gradient(rgba(29,158,117,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.05) 1px, transparent 1px)",
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "hero-gradient":
          "radial-gradient(ellipse at 20% 50%, rgba(29,158,117,0.15) 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, rgba(127,119,221,0.1) 0%, transparent 50%)",
      },
      backgroundSize: {
        "grid": "60px 60px",
      },
    },
  },
  plugins: [],
};

export default config;
