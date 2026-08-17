import assert from 'node:assert/strict'
import test from 'node:test'

import type { Session } from '@deepseek-ai/dsh-session'

import { captureMessagesSince } from '../src/capture.js'
import { resolveConfig } from '../src/config.js'
import type { EverosClient } from '../src/everos-client.js'
import { estimateMessageTokens, MemoryRuntime } from '../src/memory-runtime.js'
import type { AddRequest, FlushRequest, PluginLogger } from '../src/types.js'

function fakeSession(id = 'session-1', cwd = '/work/repository'): Session & { events: unknown[] } {
  return {
    id,
    firstLiveSeq: 0,
    header: {
      version: 0,
      id,
      createdAt: 1,
      cwd,
      agentPreset: 'coding',
    },
    events: [
      {
        type: 'user/message',
        seq: 0,
        time: 10,
        data: {
          id: 'u1',
          role: 'user',
          source: { kind: 'user' },
          content: [{ type: 'text', text: 'Fix the parser' }],
        },
      },
      {
        type: 'user/message',
        seq: 1,
        time: 11,
        data: {
          id: 'memory',
          role: 'user',
          source: { kind: 'plugin', plugin: 'everos-memory' },
          content: [{ type: 'text', text: 'recalled data' }],
        },
      },
      {
        type: 'assistant/message',
        seq: 2,
        time: 12,
        data: {
          turn: 1,
          step: 1,
          message: {
            id: 'a1',
            role: 'assistant',
            source: { kind: 'model', provider: 'deepseek', model: 'v4' },
            content: [
              { type: 'reasoning', text: 'private reasoning' },
              { type: 'text', text: 'I will inspect it.' },
              { type: 'tool-call', id: 'call-1', name: 'read_file', arguments: '{"path":"x"}' },
            ],
          },
        },
      },
      {
        type: 'tool/call',
        seq: 3,
        time: 13,
        data: { turn: 1, step: 1, callId: 'call-1', name: 'read_file', arguments: '{}' },
      },
      {
        type: 'tool/result',
        seq: 4,
        time: 14,
        data: {
          turn: 1,
          step: 1,
          message: {
            id: 't1',
            role: 'user',
            source: { kind: 'tool', callId: 'call-1' },
            content: [
              {
                type: 'tool-result',
                toolCallId: 'call-1',
                content: [{ type: 'text', text: 'file contents' }],
              },
            ],
          },
        },
      },
    ],
  } as unknown as Session & { events: unknown[] }
}

test('maps direct user, assistant tool call, and tool result without reasoning echoes', () => {
  const capture = captureMessagesSince(fakeSession(), -1, 'alice', undefined, 10_000)
  assert.equal(capture.messages.length, 3)
  assert.deepEqual(
    capture.messages.map((entry) => entry.item.role),
    ['user', 'assistant', 'tool'],
  )
  assert.equal(capture.messages[1]?.item.content, 'I will inspect it.')
  assert.equal(capture.messages[1]?.item.tool_calls?.[0]?.function.name, 'read_file')
  assert.equal(capture.messages[2]?.item.tool_call_id, 'call-1')
  assert.doesNotMatch(JSON.stringify(capture.messages), /private reasoning/u)
})

test('token estimate treats CJK more conservatively than ASCII', () => {
  const base = { sender_id: 'alice', role: 'user' as const, timestamp: 1 }
  assert.equal(estimateMessageTokens({ ...base, content: 'abcdefghijkl' }), 3)
  assert.equal(estimateMessageTokens({ ...base, content: '请记住这个偏好' }), 7)
})

test('runtime captures incrementally and seals after the pending queue', async () => {
  const adds: AddRequest[] = []
  const flushes: FlushRequest[] = []
  const client: EverosClient = {
    async health() {
      return { status: 'ok' }
    },
    async add(request) {
      adds.push(request)
      return { message_count: request.messages.length, status: 'accumulated' }
    },
    async search() {
      return {
        episodes: [],
        profiles: [],
        agent_cases: [],
        agent_skills: [],
        unprocessed_messages: [],
      }
    },
    async flush(request) {
      flushes.push(request)
      return { status: 'extracted' }
    },
    resolvedApiVersion() {
      return 'v1'
    },
  }
  const logger: PluginLogger = { info() {}, warn() {} }
  const runtime = new MemoryRuntime({
    client,
    config: resolveConfig({ autoStart: false }),
    userId: 'alice',
    logger,
  })
  const session = fakeSession()

  await runtime.capture(session)
  await runtime.capture(session)
  assert.equal(adds.length, 1)
  assert.equal(adds[0]?.messages.length, 3)
  assert.equal(adds[0]?.defer_extraction, true)

  session.events.push({
    type: 'assistant/message',
    seq: 5,
    time: 15,
    data: {
      turn: 1,
      step: 2,
      message: {
        id: 'a2',
        role: 'assistant',
        source: { kind: 'model', provider: 'deepseek', model: 'v4' },
        content: [{ type: 'text', text: 'The parser is fixed.' }],
      },
    },
  })
  await runtime.seal(session)

  assert.equal(adds.length, 2)
  assert.equal(adds[1]?.messages.length, 1)
  assert.equal(flushes.length, 1)
  assert.equal(flushes[0]?.session_id, 'session-1')
})

test('runtime skips resumed seed history and splits the 500-message API limit', async () => {
  const adds: AddRequest[] = []
  const client = {
    async add(request: AddRequest) {
      adds.push(request)
      return { message_count: request.messages.length, status: 'accumulated' as const }
    },
    async flush() {
      return { status: 'extracted' as const }
    },
  } as unknown as EverosClient
  const logger: PluginLogger = { info() {}, warn() {} }
  const runtime = new MemoryRuntime({
    client,
    config: resolveConfig({
      autoStart: false,
      flushMessageThreshold: 1_000,
      flushTokenThreshold: 1_000_000,
    }),
    userId: 'alice',
    logger,
  })
  const events = Array.from({ length: 506 }, (_, seq) => ({
    type: 'user/message',
    seq,
    time: seq + 1,
    data: {
      id: `u-${seq}`,
      role: 'user',
      source: { kind: 'user' },
      content: [{ type: 'text', text: `message ${seq}` }],
    },
  }))
  const session = {
    id: 'resumed-session',
    firstLiveSeq: 5,
    header: {
      version: 0,
      id: 'resumed-session',
      createdAt: 1,
      cwd: '/work/repository',
    },
    events,
  } as unknown as Session

  await runtime.capture(session)

  assert.deepEqual(
    adds.map((request) => request.messages.length),
    [500, 1],
  )
  assert.equal(adds[0]?.messages[0]?.content, 'message 5')
  await runtime.seal(session)
})

test('runtime flushes immediately at the configured batch threshold', async () => {
  const flushes: FlushRequest[] = []
  const client = {
    async add(request: AddRequest) {
      return { message_count: request.messages.length, status: 'accumulated' as const }
    },
    async flush(request: FlushRequest) {
      flushes.push(request)
      return { status: 'extracted' as const }
    },
  } as unknown as EverosClient
  const runtime = new MemoryRuntime({
    client,
    config: resolveConfig({
      autoStart: false,
      flushMessageThreshold: 3,
      flushTokenThreshold: 1_000_000,
    }),
    userId: 'alice',
    logger: { info() {}, warn() {} },
  })

  await runtime.capture(fakeSession())

  assert.equal(flushes.length, 1)
})

test('runtime debounce-flushes after the configured idle window', async () => {
  const flushes: FlushRequest[] = []
  const client = {
    async add(request: AddRequest) {
      return { message_count: request.messages.length, status: 'accumulated' as const }
    },
    async flush(request: FlushRequest) {
      flushes.push(request)
      return { status: 'extracted' as const }
    },
  } as unknown as EverosClient
  const runtime = new MemoryRuntime({
    client,
    config: resolveConfig({
      autoStart: false,
      flushIdleMs: 10,
      flushMaxDelayMs: 1_000,
      flushMessageThreshold: 100,
      flushTokenThreshold: 1_000_000,
    }),
    userId: 'alice',
    logger: { info() {}, warn() {} },
  })

  await runtime.capture(fakeSession())
  await new Promise((resolve) => setTimeout(resolve, 40))

  assert.equal(flushes.length, 1)
})

test('failed background flush is rearmed instead of losing the pending batch', async () => {
  let attempts = 0
  const client = {
    async add(request: AddRequest) {
      return { message_count: request.messages.length, status: 'accumulated' as const }
    },
    async flush() {
      attempts += 1
      if (attempts === 1) throw new Error('temporary failure')
      return { status: 'extracted' as const }
    },
  } as unknown as EverosClient
  const runtime = new MemoryRuntime({
    client,
    config: resolveConfig({
      autoStart: false,
      flushIdleMs: 10,
      flushMaxDelayMs: 1_000,
      flushMessageThreshold: 100,
      flushTokenThreshold: 1_000_000,
    }),
    userId: 'alice',
    logger: { info() {}, warn() {} },
  })

  await runtime.capture(fakeSession())
  await new Promise((resolve) => setTimeout(resolve, 60))

  assert.equal(attempts, 2)
})

test('new session recall waits for pending sessions in the same workspace', async () => {
  const flushes: FlushRequest[] = []
  const client = {
    async add(request: AddRequest) {
      return { message_count: request.messages.length, status: 'accumulated' as const }
    },
    async flush(request: FlushRequest) {
      flushes.push(request)
      return { status: 'extracted' as const }
    },
  } as unknown as EverosClient
  const runtime = new MemoryRuntime({
    client,
    config: resolveConfig({
      autoStart: false,
      flushIdleMs: 60_000,
      flushMessageThreshold: 100,
      flushTokenThreshold: 1_000_000,
    }),
    userId: 'alice',
    logger: { info() {}, warn() {} },
  })
  const previous = fakeSession('session-previous')
  const current = fakeSession('session-current')

  await runtime.capture(previous)
  assert.equal(flushes.length, 0)

  await runtime.flushBeforeRecall(current)

  assert.deepEqual(
    flushes.map((request) => request.session_id),
    ['session-previous'],
  )
})
