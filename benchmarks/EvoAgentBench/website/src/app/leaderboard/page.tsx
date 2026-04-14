import type { Metadata } from "next";
import { Leaderboard } from "@/components/Leaderboard";

export const metadata: Metadata = {
  title: "Leaderboard — EvoAgentBench",
  description:
    "Compare AI agent self-evolution performance across multiple domains.",
};

export default function LeaderboardPage() {
  return (
    <main className="flex-1">
      <section className="mx-auto max-w-6xl px-6 py-12">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Agent Performance</h1>
          <p className="text-sm text-muted-foreground">
            Pass rates on{" "}
            <span className="font-semibold text-foreground">EvoAgentBench</span>{" "}
            · 5 domains · EverOS skill extraction
          </p>
        </div>
        <Leaderboard />
      </section>
    </main>
  );
}
