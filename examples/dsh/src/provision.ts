/** Detect-then-provision support for a local EverOS server. */

import { type ChildProcess, type SpawnOptions, spawn } from 'node:child_process'

import { createEverosClient, type EverosClient } from './everos-client.js'
import type { PluginLogger } from './types.js'

export interface ProvisionOptions {
  baseUrl: string
  apiVersion?: 'auto' | 'v1' | 'v2'
  startCommand?: string[]
  cwd?: string
  readinessTimeoutMs?: number
  readinessIntervalMs?: number
  logger: PluginLogger
  client?: EverosClient
  spawnFn?: typeof spawn
  onStop?: (stop: () => void) => void
}

export interface ProvisionResult {
  status: 'already-running' | 'started' | 'skipped' | 'failed'
  detail: string
  stop?: () => void
}

export function portFromUrl(baseUrl: string): string {
  try {
    const url = new URL(baseUrl)
    if (url.port) return url.port
    return url.protocol === 'https:' ? '443' : '80'
  } catch {
    return '8000'
  }
}

export function isLoopbackUrl(baseUrl: string): boolean {
  try {
    const host = new URL(baseUrl).hostname.toLowerCase()
    return host === 'localhost' || host === '127.0.0.1' || host === '[::1]' || host === '::1'
  } catch {
    return false
  }
}

const sleep = (milliseconds: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, milliseconds))

export async function waitForHealthy(
  client: EverosClient,
  timeoutMs: number,
  intervalMs: number,
  shouldAbort?: () => boolean,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    if (shouldAbort?.()) return false
    try {
      await client.health({ timeoutMs: Math.min(2_000, Math.max(500, intervalMs * 2)) })
      return true
    } catch {
      if (shouldAbort?.() || Date.now() >= deadline) return false
      await sleep(intervalMs)
    }
  }
}

/** Start only loopback endpoints. The plugin never opens or mutates remote services. */
export async function provision(options: ProvisionOptions): Promise<ProvisionResult> {
  const client =
    options.client ??
    createEverosClient({ baseUrl: options.baseUrl, apiVersion: options.apiVersion })
  try {
    await client.health({ timeoutMs: 2_000 })
    options.logger.info(`everos-memory: EverOS is healthy at ${options.baseUrl}`)
    return { status: 'already-running', detail: 'health check passed' }
  } catch {
    options.logger.info(`everos-memory: EverOS is not reachable at ${options.baseUrl}`)
  }

  if (!isLoopbackUrl(options.baseUrl)) {
    const detail = 'auto-start is restricted to loopback EverOS endpoints'
    options.logger.warn(`everos-memory: ${detail}; continuing without memory`)
    return { status: 'skipped', detail }
  }

  const command = options.startCommand ?? ['everos', 'server', 'start']
  const [executable, ...argumentsList] = command
  if (!executable) return { status: 'failed', detail: 'empty start command' }

  const spawnOptions: SpawnOptions = {
    cwd: options.cwd,
    env: {
      ...process.env,
      EVEROS_MEMORIZE__MODE: 'agent',
      EVEROS_API__PORT: portFromUrl(options.baseUrl),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  }

  let child: ChildProcess
  try {
    child = (options.spawnFn ?? spawn)(executable, argumentsList, spawnOptions)
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    options.logger.warn(`everos-memory: failed to start EverOS: ${detail}`)
    return { status: 'failed', detail }
  }

  const stop = (): void => {
    try {
      child.kill()
    } catch {
      // The child may already have exited.
    }
  }
  options.onStop?.(stop)

  const lockPattern = /EngineLockHeldError|OfflineEngine instance already holds|LockException/iu
  let lockConflict = false
  const captureOutput = (chunk: Buffer | string): void => {
    const text = chunk.toString()
    if (lockPattern.test(text)) lockConflict = true
  }
  child.stdout?.on('data', captureOutput)
  child.stderr?.on('data', captureOutput)

  let childDone = false
  let exitDetail = ''
  child.on('error', (error) => {
    childDone = true
    exitDetail = `child failed: ${error.message}`
  })
  child.on('exit', (code, signal) => {
    exitDetail = `child exited (code=${String(code)}, signal=${String(signal)})`
  })
  child.on('close', () => {
    childDone = true
    if (!exitDetail) exitDetail = 'child closed before becoming healthy'
  })

  const healthy = await waitForHealthy(
    client,
    options.readinessTimeoutMs ?? 60_000,
    options.readinessIntervalMs ?? 1_000,
    () => childDone && !lockConflict,
  )
  if (healthy) {
    if (childDone) {
      options.logger.info('everos-memory: connected to the existing EverOS lock owner')
      return {
        status: 'already-running',
        detail: 'another EverOS process owns the lock and is healthy',
      }
    }
    options.logger.info(`everos-memory: started EverOS at ${options.baseUrl}`)
    return { status: 'started', detail: 'spawned and healthy', stop }
  }

  const detail = lockConflict
    ? 'another EverOS process holds the OME lock but did not become healthy'
    : exitDetail || 'readiness timeout; the child was left running'
  options.logger.warn(`everos-memory: ${detail}`)
  return {
    status: 'failed',
    detail,
    ...(childDone ? {} : { stop }),
  }
}
