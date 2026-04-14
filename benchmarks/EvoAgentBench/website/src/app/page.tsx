import Link from "next/link";
import { Leaderboard } from "@/components/Leaderboard";
import { BenchmarkDomains } from "@/components/BenchmarkDomains";
import { SkillMethods } from "@/components/SkillMethods";

export default function Home() {
  return (
    <main className="flex-1">
      {/* Hero */}
      <section className="bg-secondary py-20">
        <div className="mx-auto max-w-4xl px-6 text-center">
          <p className="font-mono text-sm text-muted-foreground mb-4 tracking-wider">
            01 — OVERVIEW
          </p>
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight leading-tight mb-6">
            EvoAgentBench
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto mb-6 leading-relaxed italic">
            A Unified Evaluation Framework for AI Agent Self-Evolution
          </p>
          <p className="text-sm text-muted-foreground max-w-3xl mx-auto mb-10 leading-relaxed">
            EvoAgentBench enables standardized comparison of agent self-evolution
            methods — techniques that allow agents to improve their performance
            by learning from past experience. It provides pluggable abstractions
            for domains, agents, and skill evaluation methods, making it easy to
            evaluate how different self-evolution approaches generalize across
            information retrieval, reasoning, software engineering, code
            implementation, and knowledge work.
          </p>
          <div className="flex items-center justify-center gap-3 flex-wrap">
            <Link
              href="/leaderboard"
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-full bg-foreground text-background text-sm font-medium hover:opacity-90 transition-opacity"
            >
              Full Leaderboard
            </Link>
            <a
              href="#domains"
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-full border border-border bg-background text-sm font-medium hover:bg-secondary transition-colors"
            >
              Domains
            </a>
            <a
              href="#methods"
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-full border border-border bg-background text-sm font-medium hover:bg-secondary transition-colors"
            >
              Self-Evolution
            </a>
          </div>
        </div>
      </section>

      {/* Key Features */}
      <section className="mx-auto max-w-5xl px-6 py-16">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="rounded-xl border bg-card p-6">
            <div className="text-2xl mb-3">🌐</div>
            <h3 className="font-semibold text-foreground mb-2">Multi-Domain Evaluation</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              5 diverse evaluation domains — information retrieval, reasoning,
              software engineering, code implementation, and knowledge work —
              with clustered train/test splits and unified evaluation pipeline.
            </p>
          </div>
          <div className="rounded-xl border bg-card p-6">
            <div className="text-2xl mb-3">🤖</div>
            <h3 className="font-semibold text-foreground mb-2">Multi-Agent Support</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Plug in any CLI-based agent — Nanabot, OpenClaw, or your own.
              Each task runs in isolated config with independent workspace,
              supporting concurrent execution and automatic retry.
            </p>
          </div>
          <div className="rounded-xl border bg-card p-6">
            <div className="text-2xl mb-3">🧬</div>
            <h3 className="font-semibold text-foreground mb-2">Self-Evolution Comparison</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Standardized train → extract → evaluate protocol for comparing
              skill-based self-evolution methods. Supports both offline
              (batch extraction) and online (learn-as-you-go) evaluation modes.
            </p>
          </div>
        </div>
      </section>

      {/* Results */}
      <section id="results" className="mx-auto max-w-6xl px-6 py-16">
        <div className="flex items-center justify-between mb-8">
          <div>
            <p className="font-mono text-sm text-muted-foreground mb-2 tracking-wider">
              02 — RESULTS
            </p>
            <h2 className="text-2xl font-bold">Agent Performance</h2>
          </div>
          <Link
            href="/leaderboard"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            View all &rarr;
          </Link>
        </div>
        <Leaderboard compact />
      </section>

      {/* Benchmark Domains */}
      <BenchmarkDomains />

      {/* Skill Extraction Methods */}
      <SkillMethods />
    </main>
  );
}
