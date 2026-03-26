"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Github, Menu, X } from "lucide-react";

const navLinks = [
  { href: "/research", label: "Research" },
  { href: "/results", label: "Results" },
  { href: "/architecture", label: "Architecture" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 32);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
        scrolled
          ? "bg-navy-900/90 backdrop-blur-md border-b border-navy-700/60 shadow-lg shadow-black/20"
          : "bg-transparent"
      )}
    >
      <nav className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-7 h-7 bg-teal-400 rounded flex items-center justify-center">
            <span className="text-navy-900 font-bold text-sm">AB</span>
          </div>
          <span className="font-semibold text-white group-hover:text-teal-300 transition-colors">
            AdaptiveBeta
          </span>
        </Link>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-6">
          {navLinks.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "nav-link",
                pathname === href && "text-white font-medium"
              )}
            >
              {label}
            </Link>
          ))}
        </div>

        {/* Right actions */}
        <div className="hidden md:flex items-center gap-3">
          <a
            href="https://github.com/daishinkan7/ai_fintech"
            target="_blank"
            rel="noopener noreferrer"
            className="nav-link flex items-center gap-1.5"
          >
            <Github className="w-4 h-4" />
            GitHub
          </a>
          <Link
            href="/results"
            className="px-4 py-1.5 bg-teal-400 hover:bg-teal-300 text-navy-900 font-medium text-sm rounded-lg transition-colors"
          >
            View Results
          </Link>
        </div>

        {/* Mobile menu button */}
        <button
          className="md:hidden text-gray-400 hover:text-white"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle mobile menu"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </nav>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="md:hidden bg-navy-900/95 backdrop-blur-md border-b border-navy-700 px-4 pb-4">
          {navLinks.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className="block py-2.5 text-gray-300 hover:text-white border-b border-navy-800 last:border-0"
              onClick={() => setMobileOpen(false)}
            >
              {label}
            </Link>
          ))}
          <a
            href="https://github.com/daishinkan7/ai_fintech"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 mt-3 text-gray-400"
          >
            <Github className="w-4 h-4" /> GitHub
          </a>
        </div>
      )}
    </header>
  );
}
