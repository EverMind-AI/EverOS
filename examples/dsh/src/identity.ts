/** Stable, path-safe EverOS identity mapping for DSH sessions. */

import { createHash } from 'node:crypto'
import { userInfo } from 'node:os'
import { basename, resolve } from 'node:path'

import type { Session } from '@deepseek-ai/dsh-session'

const PATH_SAFE = /^[a-zA-Z0-9_.@+-]+$/u
const RESERVED = new Set(['.', '..'])
const MAX_ID_CHARS = 128

function shortHash(value: string): string {
  return createHash('sha256').update(value).digest('hex').slice(0, 12)
}

/** Keep readable identifiers while making collisions and path traversal unlikely. */
export function safeId(raw: string | undefined, fallback: string): string {
  const source = raw?.trim() || fallback
  if (source.length <= MAX_ID_CHARS && PATH_SAFE.test(source) && !RESERVED.has(source)) {
    return source
  }
  const stem = source
    .normalize('NFKD')
    .replace(/[^a-zA-Z0-9_.@+-]+/gu, '-')
    .replace(/^[.-]+|[.-]+$/gu, '')
  const suffix = shortHash(source)
  const readable = !stem || RESERVED.has(stem) ? fallback : stem
  const head = readable.slice(0, MAX_ID_CHARS - suffix.length - 1)
  return `${head}-${suffix}`
}

export function sessionIdOf(session: Pick<Session, 'id'>): string {
  return safeId(String(session.id), 'dsh-session')
}

export function projectIdOf(session: Pick<Session, 'header'>, configured?: string): string {
  if (configured) return safeId(configured, 'default')
  const cwd = session.header.cwd
  if (!cwd) return 'default'
  const absolute = resolve(cwd)
  const readable = safeId(basename(absolute), 'workspace')
  const suffix = shortHash(absolute)
  const head = readable.slice(0, MAX_ID_CHARS - suffix.length - 1)
  return `${head}-${suffix}`
}

export function agentIdOf(session: Pick<Session, 'header'>, configured?: string): string {
  if (configured) return safeId(configured, 'dsh')
  const preset = session.header.agentPreset?.trim()
  return safeId(preset ? `dsh-${preset}` : 'dsh', 'dsh')
}

function osUsername(): string | undefined {
  try {
    return userInfo().username?.trim() || undefined
  } catch {
    return undefined
  }
}

export function resolveUserId(configured?: string): string {
  const raw = configured || process.env.USER || process.env.USERNAME || osUsername()
  return safeId(raw, 'local-user')
}
