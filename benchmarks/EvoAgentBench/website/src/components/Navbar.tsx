import Link from "next/link";

const HF_URL = "https://huggingface.co/datasets/EverMind-AI/EvoAgentBench";

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
        <Link href="/" className="text-lg font-bold tracking-tight">
          EvoAgentBench
        </Link>
        <nav className="flex items-center gap-5 text-sm">
          <a
            href="/#results"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Results
          </a>
          <a
            href="/#domains"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Domains
          </a>
          <a
            href="/#methods"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Self-Evolution
          </a>

          <span className="h-4 w-px bg-border" />

          <a
            href={HF_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-border text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
          >
            🤗 Hugging Face
          </a>
        </nav>
      </div>
    </header>
  );
}
