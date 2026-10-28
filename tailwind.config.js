/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#08131c",
        panel: "#0d1d28",
        cyan: "#66e3ff",
        lime: "#b9f36b",
      },
    },
  },
  plugins: [],
};
