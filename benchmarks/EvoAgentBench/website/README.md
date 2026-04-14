# EvoAgentBench Website

Leaderboard and project website for EvoAgentBench — a unified evaluation framework for AI agent self-evolution across diverse task domains.

## Features

- Interactive leaderboard with filtering and sorting
- Filter by agent (OpenClaw, Nanabot) and domain (Information Retrieval, Reasoning, Software Engineering, Code Implementation, Knowledge Work)
- Sort by score with skills, improvement delta, or baseline score
- Overall aggregation for same agent + model + method combinations
- Efficiency column showing cost change after skill injection
- 5 self-evolution methods with GitHub links (EverOS, EvoSkill, Memento, OpenSpace, ReasoningBank)
- Domain overview table with cluster/train/test statistics

## Tech Stack

- [Next.js](https://nextjs.org/) 16 (App Router, static export)
- [TypeScript](https://www.typescriptlang.org/)
- [Tailwind CSS](https://tailwindcss.com/) v4
- [shadcn/ui](https://ui.shadcn.com/) components

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the site.

## Project Structure

```
src/
├── app/
│   ├── layout.tsx          # Root layout with Navbar + Footer
│   ├── page.tsx            # Home: hero, features, results, domains, methods
│   └── leaderboard/
│       └── page.tsx        # Full leaderboard page
├── components/
│   ├── Leaderboard.tsx     # Main leaderboard with filters, sorting, overall rows
│   ├── BenchmarkDomains.tsx # Domain overview table
│   ├── SkillMethods.tsx    # Self-evolution method cards with GitHub links
│   ├── FilterButton.tsx    # Reusable filter toggle button
│   ├── Navbar.tsx          # Top navigation bar
│   ├── Footer.tsx          # Page footer
│   └── ui/                 # shadcn/ui primitives
└── data/
    └── leaderboard-data.ts # Domain info, method info, and benchmark results
```

## Deployment

### GitHub Pages

Push to main branch — GitHub Actions will automatically build and deploy.

### Vercel / Netlify

Import the GitHub repository. No extra configuration needed.

## License

MIT
