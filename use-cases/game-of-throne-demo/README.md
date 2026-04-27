# EverMem Story Memory Demo

> Built on [EverCore](https://github.com/EverMind-AI/EverOS/) - Open-source AI memory infrastructure

A demonstration web application showcasing [EverMem](https://evermind.ai)'s AI memory infrastructure through an interactive Q&A experience with "A Game of Thrones" (Book 1).

Ask questions about the book and watch two AI responses stream side-by-side: one **with memory** using EverMem to retrieve relevant passages, and one **without memory** using only the LLM's training data. See the difference memory makes.

![Demo Screenshot](https://github.com/user-attachments/assets/54a7cf8f-62c4-4fbc-9d50-b214d034e051)

## Features

- **Side-by-Side Comparison**: Watch two responses stream simultaneously - with and without memory context
- **Memory-Grounded Responses**: See exactly which book passages are used to answer questions
- **Real-time Streaming**: Token-by-token AI response streaming via SSE
- **Interactive Memory Chips**: Hover over memory chips to see full excerpt details and metadata
- **Follow-up Suggestions**: AI-generated follow-up questions after each response
- **Dark Theme UI**: Modern, clean interface inspired by EverMind's design

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Backend**: Node.js + Express + Bun
- **AI**: Claude Haiku (via OpenRouter)
- **Memory**: [EverMind Cloud API](https://evermind.ai)

## Quick Start

### Prerequisites

- [Bun](https://bun.sh/) (latest version)
- OpenAI API key (or OpenRouter API key)
- EverMind Cloud API key (apply at [EverCore Cloud](https://console.evermind.ai/))

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd evermem-story-demo

# Install dependencies
bun install
```

### Configuration

**Backend** (`backend/.env`):

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:
```bash
OPENAI_API_KEY=your-openrouter-api-key
OPENAI_MODEL=anthropic/claude-3-haiku
PORT=3001
FRONTEND_URL=http://localhost:3000

# EverMind Cloud
USE_EVERMEMOS=true
EVERMEMOS_URL=https://api.evermind.ai
EVERMEMOS_API_KEY=your-evermind-api-key
```

**Frontend** (`frontend/.env`):

```bash
cp frontend/.env.example frontend/.env
```

The default `VITE_API_URL=http://localhost:3001` should work for local development.

### Running

```bash
# Start both frontend and backend
bun run dev
```

- Frontend: http://localhost:3000
- Backend: http://localhost:3001

## Loading Novel Content

Before using the demo, you need to load novel content into EverMind Cloud.

### Quick Test with Sample

A sample file with 5 chapters is included for testing:

```bash
bun run load-novel-cloud \
  --file sample/got-sample.txt \
  --book-title "A Game of Thrones" \
  --book-abbrev "got" \
  --api-key YOUR_EVERMIND_API_KEY
```

### Full Book

For the complete experience, obtain the full novel text file and load it:

```bash
bun run load-novel-cloud \
  --file path/to/got.txt \
  --book-title "A Game of Thrones" \
  --book-abbrev "got" \
  --api-key YOUR_EVERMIND_API_KEY
```

The script:
- Detects chapter boundaries automatically (PROLOGUE, character names in caps)
- Splits text into paragraphs
- Uploads to EverMind Cloud with metadata
- Supports resumption if interrupted

## Project Structure

```
├── frontend/                # React frontend
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── hooks/          # Custom React hooks
│   │   ├── services/       # API client
│   │   └── types/          # TypeScript types
│   └── public/             # Static assets
├── backend/                 # Express backend
│   ├── src/
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utilities
├── scripts/                 # CLI tools
│   ├── load-novel-cloud.ts # Load novel to EverMind
│   └── clear-memories-cloud.ts
└── sample/                  # Sample data for testing
    └── got-sample.txt      # 5 chapters from Book 1
```

## Development

```bash
# Start dev servers
bun run dev

# Frontend only
bun run dev:frontend

# Backend only
bun run dev:backend

# Type check
bun run type-check

# Lint
bun run lint
```

## License

MIT

## Acknowledgments

- [EverMind](https://evermind.ai) for the memory infrastructure
- George R.R. Martin for "A Song of Ice and Fire"
