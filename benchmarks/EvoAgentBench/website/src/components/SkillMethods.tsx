import { ALL_METHODS } from "@/data/leaderboard-data";

const APPROACH_COLORS: Record<string, { bg: string; text: string }> = {
  "Memory-based extraction": { bg: "#d1fae5", text: "#065f46" },
  "Evolutionary optimization": { bg: "#dbeafe", text: "#1e40af" },
  "Retrieval-based (CBR)": { bg: "#fce7f3", text: "#9d174d" },
  "Continuous accumulation": { bg: "#fef3c7", text: "#92400e" },
  "Memory + reasoning": { bg: "#fef9c3", text: "#713f12" },
  "Expert-crafted": { bg: "#e0f2fe", text: "#0369a1" },
};

export function SkillMethods() {
  return (
    <section id="methods" className="py-16 md:py-24 px-6 bg-secondary">
      <div className="max-w-6xl mx-auto">
        <p className="font-mono text-sm text-muted-foreground mb-2 tracking-wider">
          04 — SELF-EVOLUTION
        </p>
        <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
          Skill Extraction Methods
        </h2>
        <p className="text-muted-foreground mb-10 max-w-2xl">
          EvoAgentBench provides a standardized protocol for evaluating agent
          self-evolution methods — techniques that let agents learn from past
          experience and improve future performance. 5 methods are currently
          integrated.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {ALL_METHODS.map((method) => {
            const colors = APPROACH_COLORS[method.approach] || { bg: "#f3f4f6", text: "#374151" };
            return (
              <div
                key={method.name}
                className="rounded-xl border bg-card p-5 transition-shadow hover:shadow-md flex flex-col"
              >
                <div className="flex items-center gap-2 mb-3 flex-wrap">
                  <h3 className="text-base font-semibold text-foreground">
                    {method.name}
                  </h3>
                  <span
                    className="inline-block px-2 py-0.5 rounded-md text-[11px] font-medium"
                    style={{ background: colors.bg, color: colors.text }}
                  >
                    {method.approach}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed flex-1">
                  {method.description}
                </p>
                {method.github && (
                  <a
                    href={method.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 mt-3 text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" /></svg>
                    GitHub
                  </a>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
