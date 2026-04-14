"use client";

import { useMemo, useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { FilterButton } from "@/components/FilterButton";
import {
  leaderboardData,
  AGENTS,
  DOMAINS,
  AGENT_COLORS,
  AGENT_BAR_COLORS,
  DOMAIN_LABELS,
  DOMAIN_TAG_COLORS,
  SKILL_METHOD_TAG_COLORS,
  type Agent,
  type Domain,
  type LeaderboardEntry,
} from "@/data/leaderboard-data";

type SortKey = "withSkills" | "delta" | "without";

interface DisplayRow {
  agent: Agent;
  model: string;
  domain: string;
  skillMethod: string;
  without: number;
  withSkills: number;
  efficiency: string;
  isOverall: boolean;
  domainTag?: Domain;
}

function buildOverallRows(data: LeaderboardEntry[]): DisplayRow[] {
  const groups = new Map<string, LeaderboardEntry[]>();
  for (const entry of data) {
    const key = `${entry.agent}|${entry.model}|${entry.skillMethod}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(entry);
  }

  const overalls: DisplayRow[] = [];
  for (const [, entries] of groups) {
    if (entries.length < 2) continue;
    const avgWithout = entries.reduce((s, e) => s + e.without, 0) / entries.length;
    const avgWith = entries.reduce((s, e) => s + e.withSkills, 0) / entries.length;

    // Parse efficiency percentages and average them
    let effSum = 0;
    let effCount = 0;
    for (const e of entries) {
      const match = e.efficiency.match(/([↑↓])\s*([\d.]+)%/);
      if (match) {
        const val = parseFloat(match[2]);
        // ↓ means cost reduced (negative), ↑ means cost increased (positive)
        effSum += match[1] === "↓" ? -val : val;
        effCount++;
      }
    }
    let effStr = "—";
    if (effCount > 0) {
      const avgEff = effSum / effCount;
      const arrow = avgEff <= 0 ? "↓" : "↑";
      effStr = `${arrow} ${Math.abs(avgEff).toFixed(1)}%`;
    }

    overalls.push({
      agent: entries[0].agent,
      model: entries[0].model,
      domain: "Overall",
      skillMethod: entries[0].skillMethod,
      without: Math.round(avgWithout * 10) / 10,
      withSkills: Math.round(avgWith * 10) / 10,
      efficiency: effStr,
      isOverall: true,
    });
  }
  return overalls;
}

export function Leaderboard({ compact = false }: { compact?: boolean }) {
  const [agentFilter, setAgentFilter] = useState<Agent | "All">("All");
  const [domainFilter, setDomainFilter] = useState<Domain | "All" | "Overall">("All");
  const [sortBy, setSortBy] = useState<SortKey>("withSkills");

  const filtered = useMemo(() => {
    let data = [...leaderboardData];
    if (agentFilter !== "All") {
      data = data.filter((d) => d.agent === agentFilter);
    }
    if (domainFilter !== "All" && domainFilter !== "Overall") {
      data = data.filter((d) => d.domain === domainFilter);
    }

    // Build display rows
    const overalls = buildOverallRows(data);

    if (domainFilter === "Overall") {
      // Only show overall rows
      const all = [...overalls];
      return all;
    }

    const individual: DisplayRow[] = data.map((d) => ({
      ...d,
      domain: DOMAIN_LABELS[d.domain],
      isOverall: false,
      domainTag: d.domain,
    }));

    const all = [...individual];

    if (sortBy === "withSkills") {
      all.sort((a, b) => b.withSkills - a.withSkills);
    } else if (sortBy === "delta") {
      all.sort(
        (a, b) =>
          (b.withSkills - b.without) - (a.withSkills - a.without)
      );
    } else {
      all.sort((a, b) => b.without - a.without);
    }

    return all;
  }, [agentFilter, domainFilter, sortBy]);

  const maxScore = Math.max(...filtered.map((d) => d.withSkills), 1);

  const bestScore = Math.max(...leaderboardData.map((d) => d.withSkills));
  const avgDelta =
    leaderboardData.reduce((s, d) => s + (d.withSkills - d.without), 0) /
    leaderboardData.length;

  const displayData = compact ? filtered.slice(0, 10) : filtered;

  return (
    <div>
      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="text-center">
          <p className="font-mono text-3xl font-bold text-foreground">
            {bestScore.toFixed(1)}%
          </p>
          <p className="text-sm text-muted-foreground mt-1">Best With Skills</p>
        </div>
        <div className="text-center">
          <p className="font-mono text-3xl font-bold text-foreground">
            +{avgDelta.toFixed(1)}%
          </p>
          <p className="text-sm text-muted-foreground mt-1">Avg. Improvement</p>
        </div>
        <div className="text-center">
          <p className="font-mono text-3xl font-bold text-foreground">
            {leaderboardData.length}
          </p>
          <p className="text-sm text-muted-foreground mt-1">Configurations</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-6 mb-6">
        {/* Agent filter */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-muted-foreground uppercase tracking-widest font-medium">
            Filter · Agent
          </span>
          <div className="flex gap-1.5">
            <FilterButton
              active={agentFilter === "All"}
              onClick={() => setAgentFilter("All")}
            >
              All
            </FilterButton>
            {AGENTS.map((agent) => (
              <FilterButton
                key={agent}
                active={agentFilter === agent}
                onClick={() => setAgentFilter(agent)}
              >
                {agent}
              </FilterButton>
            ))}
          </div>
        </div>

        {/* Domain filter */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-muted-foreground uppercase tracking-widest font-medium">
            Filter · Domain
          </span>
          <div className="flex gap-1.5 flex-wrap">
            <FilterButton
              active={domainFilter === "All"}
              onClick={() => setDomainFilter("All")}
            >
              All
            </FilterButton>
            <FilterButton
              active={domainFilter === "Overall"}
              onClick={() => setDomainFilter("Overall")}
            >
              Overall
            </FilterButton>
            {DOMAINS.map((domain) => (
              <FilterButton
                key={domain}
                active={domainFilter === domain}
                onClick={() => setDomainFilter(domain)}
              >
                {DOMAIN_LABELS[domain]}
              </FilterButton>
            ))}
          </div>
        </div>

        {/* Sort */}
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-[11px] text-muted-foreground uppercase tracking-widest font-medium">
            Sort by
          </span>
          <div className="flex gap-1.5">
            <FilterButton
              active={sortBy === "withSkills"}
              onClick={() => setSortBy("withSkills")}
            >
              With skills
            </FilterButton>
            <FilterButton
              active={sortBy === "delta"}
              onClick={() => setSortBy("delta")}
            >
              Δ gain
            </FilterButton>
            <FilterButton
              active={sortBy === "without"}
              onClick={() => setSortBy("without")}
            >
              Without
            </FilterButton>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="border-b-2">
              <TableHead className="w-10 text-center">#</TableHead>
              <TableHead>Agent</TableHead>
              <TableHead>Base Model</TableHead>
              <TableHead>Domain</TableHead>
              <TableHead>Skill Method</TableHead>
              <TableHead className="text-right">Without</TableHead>
              <TableHead className="text-right font-semibold">
                With Skills
              </TableHead>
              <TableHead className="text-right">Δ</TableHead>
              <TableHead className="w-24"></TableHead>
              <TableHead className="text-right">Efficiency</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {displayData.map((row, i) => {
              const delta = row.withSkills - row.without;
              const barWidth = (row.withSkills / maxScore) * 100;
              const deltaColor = delta >= 0 ? "text-emerald-600" : "text-red-500";
              const deltaStr = delta >= 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1);

              return (
                <TableRow
                  key={`${row.agent}-${row.model}-${row.domain}-${row.skillMethod}`}
                  className={row.isOverall ? "bg-muted/60 font-semibold border-t-2" : ""}
                >
                  <TableCell className="text-center font-mono text-muted-foreground">
                    {i + 1}
                  </TableCell>
                  <TableCell className="font-semibold">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full mr-2 align-middle"
                      style={{ background: AGENT_COLORS[row.agent] }}
                    />
                    {row.agent}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {row.model}
                  </TableCell>
                  <TableCell>
                    {row.isOverall ? (
                      <span className="inline-block px-3 py-0.5 rounded-md text-xs font-bold bg-foreground text-background">
                        Overall
                      </span>
                    ) : row.domainTag ? (
                      <span
                        className="inline-block px-3 py-0.5 rounded-md text-xs font-medium"
                        style={{
                          background: DOMAIN_TAG_COLORS[row.domainTag].bg,
                          color: DOMAIN_TAG_COLORS[row.domainTag].text,
                        }}
                      >
                        {row.domain}
                      </span>
                    ) : (
                      row.domain
                    )}
                  </TableCell>
                  <TableCell>
                    <span
                      className="inline-block px-3 py-0.5 rounded-md text-xs font-medium"
                      style={{
                        background: SKILL_METHOD_TAG_COLORS[row.skillMethod as keyof typeof SKILL_METHOD_TAG_COLORS]?.bg ?? "#f3f4f6",
                        color: SKILL_METHOD_TAG_COLORS[row.skillMethod as keyof typeof SKILL_METHOD_TAG_COLORS]?.text ?? "#374151",
                      }}
                    >
                      {row.skillMethod}
                    </span>
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {row.without}%
                  </TableCell>
                  <TableCell className="text-right font-bold text-lg">
                    {row.withSkills}%
                  </TableCell>
                  <TableCell className={`text-right font-semibold ${deltaColor}`}>
                    {deltaStr}
                  </TableCell>
                  <TableCell>
                    <div
                      className="h-2.5 rounded-full transition-all duration-500"
                      style={{
                        width: `${barWidth}%`,
                        background: AGENT_BAR_COLORS[row.agent],
                      }}
                    />
                  </TableCell>
                  <TableCell className={`text-right font-semibold text-sm whitespace-nowrap ${
                    row.efficiency === "—" ? "text-muted-foreground"
                    : row.efficiency.startsWith("↓") ? "text-emerald-600"
                    : "text-red-500"
                  }`}>
                    {row.efficiency}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
