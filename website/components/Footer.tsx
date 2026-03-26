import Link from "next/link";
import { Github } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-navy-700/60 mt-24 py-12 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 bg-teal-400 rounded flex items-center justify-center">
                <span className="text-navy-900 font-bold text-xs">AB</span>
              </div>
              <span className="font-semibold text-white">AdaptiveBeta</span>
            </div>
            <p className="text-sm text-gray-500 leading-relaxed">
              AI-powered dynamic beta prediction and portfolio optimisation for the Indian equity market.
            </p>
          </div>

          {/* Navigation */}
          <div>
            <p className="text-xs font-semibold tracking-widest uppercase text-gray-500 mb-3">Navigation</p>
            <ul className="space-y-2">
              {[
                { href: "/", label: "Home" },
                { href: "/research", label: "Research" },
                { href: "/results", label: "Results" },
                { href: "/architecture", label: "Architecture" },
              ].map(({ href, label }) => (
                <li key={href}>
                  <Link href={href} className="text-sm text-gray-400 hover:text-white transition-colors">
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Project info */}
          <div>
            <p className="text-xs font-semibold tracking-widest uppercase text-gray-500 mb-3">Project</p>
            <div className="space-y-2 text-sm text-gray-400">
              <p><span className="text-gray-300">Author:</span> Kunal</p>
              <p><span className="text-gray-300">Degree:</span> M.Tech AI & ML</p>
              <p><span className="text-gray-300">Institute:</span> Symbiosis Institute of Technology, Pune</p>
              <p><span className="text-gray-300">Year:</span> 2026</p>
            </div>
            <a
              href="https://github.com/daishinkan7/ai_fintech"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 mt-4 text-sm text-teal-400 hover:text-teal-300 transition-colors"
            >
              <Github className="w-4 h-4" /> View on GitHub
            </a>
          </div>
        </div>

        <div className="border-t border-navy-800 pt-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-xs text-gray-600">
            © 2026 Kunal. M.Tech Thesis — Symbiosis Institute of Technology, Pune. MIT License.
          </p>
          <p className="text-xs text-gray-600">
            Built with Next.js 14 · Tailwind CSS · Framer Motion · Recharts
          </p>
        </div>
      </div>
    </footer>
  );
}
