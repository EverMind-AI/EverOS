# Use official Bun image
FROM oven/bun:1 AS base
WORKDIR /app

# Install dependencies
FROM base AS deps
COPY package.json bun.lockb* ./
RUN bun install --frozen-lockfile || bun install

# Build stage (copy source)
FROM base AS runner
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Cloud Run uses PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

# Run the server
CMD ["bun", "run", "src/server.ts"]
