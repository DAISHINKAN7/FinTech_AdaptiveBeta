import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "AdaptiveBeta — AI-Powered Dynamic Beta Portfolio",
  description:
    "LSTM-based system that predicts forward beta volatility and triggers adaptive portfolio rebalancing on NIFTY50. M.Tech AI & ML thesis, Symbiosis Institute of Technology, Pune.",
  keywords: ["portfolio optimisation", "LSTM", "beta prediction", "NIFTY50", "quantitative finance", "adaptive beta", "HMM regime"],
  authors: [{ name: "Kunal Ajgaonkar", url: "https://github.com/daishinkan7" }],
  openGraph: {
    title: "AdaptiveBeta — AI-Powered Dynamic Beta Portfolio",
    description:
      "Walk-forward backtest 2015–2025 | 6 strategies compared | Interactive live simulator | NIFTY50 universe.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-navy-900 text-gray-100 antialiased">
        <div className="relative">
          {/* Background grid pattern */}
          <div
            className="fixed inset-0 bg-grid-pattern bg-grid opacity-30 pointer-events-none"
            aria-hidden="true"
          />
          {/* Radial glow effects */}
          <div
            className="fixed inset-0 pointer-events-none"
            style={{
              background:
                "radial-gradient(ellipse at 15% 40%, rgba(29,158,117,0.08) 0%, transparent 55%), radial-gradient(ellipse at 85% 15%, rgba(127,119,221,0.06) 0%, transparent 50%)",
            }}
            aria-hidden="true"
          />
          <div className="relative z-10">
            <Navbar />
            <main>{children}</main>
            <Footer />
          </div>
        </div>
      </body>
    </html>
  );
}
