/** Durable capture plus adaptive, batched extraction over the EverOS client. */

import type { Session } from '@deepseek-ai/dsh-session'

import { captureMessagesSince } from './capture.js'
import type { ResolvedConfig } from './config.js'
import type { EverosClient } from './everos-client.js'
import { projectIdOf, sessionIdOf } from './identity.js'
import type { FlushResponse, MessageItem, PluginLogger } from './types.js'

const ADD_MAX_MESSAGES = 500

type FlushReason =
  | 'idle'
  | 'threshold'
  | 'max-delay'
  | 'session-switch'
  | 'session-disposed'
  | 'shutdown'

interface SessionState {
  cursor: number
  pendingMessages: number
  pendingTokens: number
  firstPendingAt?: number
  idleTimer?: ReturnType<typeof setTimeout>
  maxTimer?: ReturnType<typeof setTimeout>
}

export interface MemoryRuntimeOptions {
  client: EverosClient
  config: ResolvedConfig
  userId: string
  logger: PluginLogger
}

function serializedMessage(message: MessageItem): string {
  const content =
    typeof message.content === 'string' ? message.content : JSON.stringify(message.content)
  const toolCalls = message.tool_calls ? JSON.stringify(message.tool_calls) : ''
  return `${content}${toolCalls}${message.tool_call_id ?? ''}`
}

/** Cheap deterministic estimate used only for deciding when to batch-flush. */
export function estimateMessageTokens(message: MessageItem): number {
  let asciiChars = 0
  let nonAsciiChars = 0
  for (const character of serializedMessage(message)) {
    if ((character.codePointAt(0) ?? 0) <= 0x7f) asciiChars += 1
    else nonAsciiChars += 1
  }
  return Math.max(1, Math.ceil(asciiChars / 4) + nonAsciiChars)
}

export class MemoryRuntime {
  private readonly states = new Map<string, SessionState>()
  private readonly sessions = new Map<string, Session>()
  private readonly queues = new Map<string, Promise<void>>()
  private readonly seals = new Map<string, Promise<void>>()
  private readonly detached = new Set<Promise<void>>()

  constructor(private readonly options: MemoryRuntimeOptions) {}

  private remember(session: Session): SessionState {
    const id = sessionIdOf(session)
    this.sessions.set(id, session)
    let state = this.states.get(id)
    if (!state) {
      state = {
        cursor: session.firstLiveSeq - 1,
        pendingMessages: 0,
        pendingTokens: 0,
      }
      this.states.set(id, state)
    }
    return state
  }

  /** A new user turn cancels the idle timer while preserving the max-age guard. */
  noteActivity(session: Session): void {
    const state = this.remember(session)
    if (state.idleTimer) clearTimeout(state.idleTimer)
    state.idleTimer = undefined
  }

  private clearTimers(state: SessionState): void {
    if (state.idleTimer) clearTimeout(state.idleTimer)
    if (state.maxTimer) clearTimeout(state.maxTimer)
    state.idleTimer = undefined
    state.maxTimer = undefined
  }

  private exclusive(session: Session, operation: () => Promise<void>): Promise<void> {
    const id = sessionIdOf(session)
    const prior = this.queues.get(id) ?? Promise.resolve()
    const current = prior.catch(() => undefined).then(operation)
    this.queues.set(id, current)
    const cleanup = (): void => {
      if (this.queues.get(id) === current) this.queues.delete(id)
    }
    void current.then(cleanup, cleanup)
    return current
  }

  private async captureNow(session: Session): Promise<SessionState> {
    const state = this.remember(session)
    const id = sessionIdOf(session)
    const slice = captureMessagesSince(
      session,
      state.cursor,
      this.options.userId,
      this.options.config.agentId,
      this.options.config.captureMaxChars,
    )
    const projectId = projectIdOf(session, this.options.config.projectId)
    for (let index = 0; index < slice.messages.length; index += ADD_MAX_MESSAGES) {
      const batch = slice.messages.slice(index, index + ADD_MAX_MESSAGES)
      await this.options.client.add(
        {
          session_id: id,
          app_id: this.options.config.appId,
          project_id: projectId,
          messages: batch.map((entry) => entry.item),
          defer_extraction: true,
        },
        { timeoutMs: this.options.config.captureTimeoutMs },
      )
      const last = batch.at(-1)
      if (last) state.cursor = last.seq
      if (batch.length > 0) {
        state.firstPendingAt ??= Date.now()
        state.pendingMessages += batch.length
        state.pendingTokens += batch.reduce(
          (total, entry) => total + estimateMessageTokens(entry.item),
          0,
        )
      }
    }
    state.cursor = Math.max(state.cursor, slice.scannedThroughSeq)
    return state
  }

  private thresholdReached(state: SessionState): boolean {
    return (
      state.pendingMessages >= this.options.config.flushMessageThreshold ||
      state.pendingTokens >= this.options.config.flushTokenThreshold
    )
  }

  private armTimers(session: Session, state: SessionState): void {
    if (state.pendingMessages === 0) return
    if (state.idleTimer) clearTimeout(state.idleTimer)
    state.idleTimer = setTimeout(() => {
      state.idleTimer = undefined
      this.track(this.flush(session, 'idle'), 'idle flush')
    }, this.options.config.flushIdleMs)
    state.idleTimer.unref()

    if (!state.maxTimer) {
      const firstPendingAt = state.firstPendingAt ?? Date.now()
      const remaining = Math.max(
        1,
        firstPendingAt + this.options.config.flushMaxDelayMs - Date.now(),
      )
      state.maxTimer = setTimeout(() => {
        state.maxTimer = undefined
        this.track(this.flush(session, 'max-delay'), 'max-delay flush')
      }, remaining)
      state.maxTimer.unref()
    }
  }

  /** Persist newly committed events, then schedule or perform a batched flush. */
  async capture(session: Session): Promise<void> {
    this.noteActivity(session)
    let state: SessionState | undefined
    await this.exclusive(session, async () => {
      state = await this.captureNow(session)
    })
    if (!state || state.pendingMessages === 0) return
    if (this.thresholdReached(state)) {
      await this.flush(session, 'threshold')
      return
    }
    this.armTimers(session, state)
  }

  /** Commit one buffered session. Calls are serialized with capture for that session. */
  async flush(session: Session, reason: FlushReason, force = false): Promise<void> {
    await this.exclusive(session, async () => {
      const state = await this.captureNow(session)
      this.clearTimers(state)
      if (!force && state.pendingMessages === 0) return

      const id = sessionIdOf(session)
      let result: FlushResponse
      try {
        result = await this.options.client.flush(
          {
            session_id: id,
            app_id: this.options.config.appId,
            project_id: projectIdOf(session, this.options.config.projectId),
          },
          { timeoutMs: this.options.config.captureTimeoutMs },
        )
      } catch (error) {
        if (reason !== 'shutdown' && reason !== 'session-disposed') {
          state.firstPendingAt = Date.now()
          this.armTimers(session, state)
        }
        throw error
      }
      state.pendingMessages = 0
      state.pendingTokens = 0
      state.firstPendingAt = undefined
      this.options.logger.info(
        `everos-memory: flushed session ${id.slice(0, 12)} reason=${reason} status=${result.status}`,
      )
    })
  }

  /** Ensure sessions being left behind are searchable before a new session recalls. */
  async flushBeforeRecall(current: Session): Promise<void> {
    this.remember(current)
    if (!this.options.config.flushOnSessionSwitch) return
    const currentId = sessionIdOf(current)
    const currentProject = projectIdOf(current, this.options.config.projectId)
    const pending = [...this.sessions.entries()].flatMap(([id, session]) => {
      const state = this.states.get(id)
      if (
        id === currentId ||
        !state ||
        state.pendingMessages === 0 ||
        projectIdOf(session, this.options.config.projectId) !== currentProject
      ) {
        return []
      }
      return [this.flush(session, 'session-switch')]
    })
    await Promise.all(pending)
  }

  /** Capture any remaining tail, force extraction, and retire local state once. */
  seal(session: Session, reason: FlushReason = 'session-disposed'): Promise<void> {
    const id = sessionIdOf(session)
    const existing = this.seals.get(id)
    if (existing) return existing
    this.clearTimers(this.remember(session))

    const job = this.flush(session, reason, true).then(() => {
      this.states.delete(id)
      this.sessions.delete(id)
      this.options.logger.info(`everos-memory: sealed session ${id.slice(0, 12)}`)
    })
    this.seals.set(id, job)
    const cleanup = (): void => {
      if (this.seals.get(id) === job) this.seals.delete(id)
    }
    void job.then(cleanup, cleanup)
    return job
  }

  /** Track fire-and-forget lifecycle work so plugin disposal can drain it. */
  track(operation: Promise<void>, label: string): void {
    const contained = operation.catch((error: unknown) => {
      this.options.logger.warn(`everos-memory: ${label} failed (ignored): ${String(error)}`)
    })
    this.detached.add(contained)
    void contained.then(() => this.detached.delete(contained))
  }

  /** Seal every observed session and wait for already-started background work. */
  async dispose(): Promise<void> {
    const sealing = [...this.sessions.values()].map((session) =>
      this.seal(session, 'shutdown').catch((error: unknown) => {
        this.options.logger.warn(`everos-memory: shutdown seal failed (ignored): ${String(error)}`)
      }),
    )
    await Promise.all([...sealing, ...this.detached])
  }
}
