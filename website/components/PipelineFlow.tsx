"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { pipelineStages } from "@/lib/sampleData";
import { ChevronRight, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface PipelineFlowProps {
  showDetails?: boolean;
}

export default function PipelineFlow({ showDetails = false }: PipelineFlowProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const selectedStage = pipelineStages.find((s) => s.id === selected);

  return (
    <div className="relative">
      {/* Pipeline nodes */}
      <div className="flex flex-wrap items-center gap-2 md:gap-0">
        {pipelineStages.map((stage, i) => (
          <div key={stage.id} className="flex items-center">
            <motion.button
              whileHover={{ scale: 1.04, y: -2 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setSelected(selected === stage.id ? null : stage.id)}
              className={cn(
                "relative flex flex-col items-center gap-2 px-4 py-3 rounded-xl border transition-all cursor-pointer text-center w-28 md:w-32",
                selected === stage.id
                  ? "border-opacity-60 shadow-lg"
                  : "border-navy-700 hover:border-navy-600 bg-navy-800/60"
              )}
              style={
                selected === stage.id
                  ? {
                      borderColor: stage.color,
                      background: `${stage.color}15`,
                      boxShadow: `0 0 20px ${stage.color}20`,
                    }
                  : {}
              }
            >
              {/* Step number */}
              <span
                className="text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center"
                style={{ background: `${stage.color}25`, color: stage.color }}
              >
                {i + 1}
              </span>
              <span className="text-xs font-semibold text-white leading-tight">{stage.label}</span>
              <span className="text-[10px] text-gray-500 leading-tight line-clamp-2">
                {stage.description.split(" ").slice(0, 5).join(" ")}…
              </span>
            </motion.button>

            {/* Arrow between stages */}
            {i < pipelineStages.length - 1 && (
              <div className="hidden md:flex items-center mx-1 text-navy-600">
                <ChevronRight className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Detail panel */}
      <AnimatePresence>
        {selectedStage && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.2 }}
            className="mt-4 rounded-xl border p-5 relative"
            style={{
              borderColor: `${selectedStage.color}40`,
              background: `${selectedStage.color}08`,
            }}
          >
            <button
              onClick={() => setSelected(null)}
              className="absolute top-3 right-3 text-gray-500 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-2 mb-2">
              <div
                className="text-xs font-bold px-2 py-0.5 rounded"
                style={{ background: `${selectedStage.color}25`, color: selectedStage.color }}
              >
                Stage {pipelineStages.indexOf(selectedStage) + 1}
              </div>
              <h3 className="font-semibold text-white">{selectedStage.label}</h3>
            </div>
            <p className="text-sm text-gray-300 mb-2">{selectedStage.description}</p>
            <p className="text-xs font-mono text-gray-500">{selectedStage.detail}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
