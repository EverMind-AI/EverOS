import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { DOMAIN_INFO, DOMAIN_TAG_COLORS } from "@/data/leaderboard-data";

export function BenchmarkDomains() {
  return (
    <section id="domains" className="py-16 md:py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <p className="font-mono text-sm text-muted-foreground mb-2 tracking-wider">
          03 — DOMAINS
        </p>
        <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
          Evaluation Domains
        </h2>
        <p className="text-muted-foreground mb-10 max-w-2xl">
          EvoAgentBench builds on existing benchmarks by clustering tasks into
          domains with train/test splits for self-evolution training and
          evaluation.
        </p>

        <div className="rounded-lg border overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-b-2">
                <TableHead>Domain</TableHead>
                <TableHead>Base Benchmark</TableHead>
                <TableHead>Description</TableHead>
                <TableHead>Clusters</TableHead>
                <TableHead className="text-right">Train</TableHead>
                <TableHead className="text-right">Test</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {DOMAIN_INFO.map((domain) => {
                const colors = DOMAIN_TAG_COLORS[domain.id];
                return (
                  <TableRow key={domain.id}>
                    <TableCell className="font-semibold whitespace-nowrap">
                      <span
                        className="inline-block px-2.5 py-0.5 rounded-md text-xs font-medium"
                        style={{ background: colors.bg, color: colors.text }}
                      >
                        {domain.name}
                      </span>
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {domain.baseBenchmark}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {domain.description}
                    </TableCell>
                    <TableCell className="text-sm whitespace-nowrap">
                      {domain.clusters}
                    </TableCell>
                    <TableCell className="text-right font-mono font-semibold">
                      {domain.train}
                    </TableCell>
                    <TableCell className="text-right font-mono font-semibold">
                      {domain.test}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </div>
    </section>
  );
}
