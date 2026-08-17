import assert from 'node:assert/strict'
import test from 'node:test'

import type { Session } from '@deepseek-ai/dsh-session'

import { normalizeBaseUrl, resolveConfig, splitCommand } from '../src/config.js'
import { agentIdOf, projectIdOf, safeId } from '../src/identity.js'

test('normalizes URLs and splits a shell-free command line', () => {
  assert.equal(normalizeBaseUrl('localhost:9000/'), 'http://localhost:9000')
  assert.deepEqual(splitCommand('"/Users/My Name/bin/everos" server start'), [
    '/Users/My Name/bin/everos',
    'server',
    'start',
  ])
})

test('rejects invalid numeric budgets', () => {
  assert.throws(() => resolveConfig({ queryN: 0 }), /queryN/u)
  assert.throws(() => resolveConfig({ captureTimeoutMs: 1.5 }), /captureTimeoutMs/u)
  assert.throws(() => resolveConfig({ projectId: '../other' }), /projectId/u)
  assert.throws(
    () => resolveConfig({ baseUrl: 'http://user:secret@127.0.0.1:8000' }),
    /must not contain credentials/u,
  )
})

test('defaults recall to Tier 1 keyword retrieval', () => {
  const defaults = resolveConfig()
  assert.equal(defaults.recallMethod, 'keyword')
  assert.equal(defaults.flushIdleMs, 30_000)
  assert.equal(defaults.flushTokenThreshold, 12_000)
  assert.equal(defaults.flushMessageThreshold, 50)
  assert.equal(defaults.flushOnSessionSwitch, true)
  assert.equal(resolveConfig({ recallMethod: 'hybrid' }).recallMethod, 'hybrid')
})

test('derives deterministic collision-resistant workspace and agent ids', () => {
  const sessionA = {
    header: { cwd: '/work/one/project', agentPreset: 'coding/default' },
  } as unknown as Session
  const sessionB = {
    header: { cwd: '/work/two/project', agentPreset: 'coding/default' },
  } as unknown as Session

  assert.notEqual(projectIdOf(sessionA), projectIdOf(sessionB))
  assert.match(projectIdOf(sessionA), /^project-[a-f0-9]{12}$/u)
  assert.match(agentIdOf(sessionA), /^dsh-coding-default-[a-f0-9]{12}$/u)
  assert.match(safeId('../unsafe/user', 'fallback'), /^unsafe-user-[a-f0-9]{12}$/u)
})
