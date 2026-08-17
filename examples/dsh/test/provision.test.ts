import assert from 'node:assert/strict'
import test from 'node:test'

import type { EverosClient } from '../src/everos-client.js'
import { isLoopbackUrl, portFromUrl, provision } from '../src/provision.js'
import type { PluginLogger } from '../src/types.js'

const logger: PluginLogger = { info() {}, warn() {} }

test('recognizes loopback endpoints and derives explicit or default ports', () => {
  assert.equal(isLoopbackUrl('http://127.0.0.1:8000'), true)
  assert.equal(isLoopbackUrl('http://localhost:9000'), true)
  assert.equal(isLoopbackUrl('https://memory.example.com'), false)
  assert.equal(portFromUrl('http://127.0.0.1:8123'), '8123')
  assert.equal(portFromUrl('https://memory.example.com'), '443')
})

test('never auto-starts a process for a remote EverOS URL', async () => {
  let spawnCalls = 0
  const client = {
    async health() {
      throw new Error('offline')
    },
  } as unknown as EverosClient
  const result = await provision({
    baseUrl: 'https://memory.example.com',
    logger,
    client,
    spawnFn: (() => {
      spawnCalls += 1
      throw new Error('must not run')
    }) as never,
  })

  assert.equal(result.status, 'skipped')
  assert.equal(spawnCalls, 0)
})
