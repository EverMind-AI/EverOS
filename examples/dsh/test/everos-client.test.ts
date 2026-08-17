import assert from 'node:assert/strict'
import test from 'node:test'

import { createEverosClient, EverosError } from '../src/everos-client.js'
import type { SearchRequest } from '../src/types.js'

const emptySearch = {
  episodes: [],
  profiles: [],
  agent_cases: [],
  agent_skills: [],
  unprocessed_messages: [],
}

test('auto-negotiates v2 to v1 once on a 404', async () => {
  const paths: string[] = []
  const fakeFetch: typeof fetch = async (input) => {
    const path = new URL(String(input)).pathname
    paths.push(path)
    if (path.startsWith('/api/v2/')) {
      return new Response(JSON.stringify({ error: { code: 'NOT_FOUND', message: 'missing' } }), {
        status: 404,
      })
    }
    return new Response(JSON.stringify({ request_id: 'r1', data: emptySearch }), { status: 200 })
  }
  const client = createEverosClient({
    baseUrl: 'http://127.0.0.1:8000',
    apiVersion: 'auto',
    fetch: fakeFetch,
  })

  await client.search({ user_id: 'alice', query: 'first' })
  await client.search({ agent_id: 'dsh', query: 'second' })

  assert.deepEqual(paths, [
    '/api/v2/memory/search',
    '/api/v1/memory/search',
    '/api/v1/memory/search',
  ])
  assert.equal(client.resolvedApiVersion(), 'v1')
})

test('rejects invalid owner and scope before network access', async () => {
  const client = createEverosClient({
    baseUrl: 'http://127.0.0.1:8000',
    fetch: async () => new Response('{}'),
  })

  await assert.rejects(
    client.search({ user_id: 'alice', agent_id: 'dsh', query: 'x' } as unknown as SearchRequest),
    (error: unknown) => error instanceof EverosError && error.code === 'INVALID_OWNER',
  )
  await assert.rejects(
    client.add({ session_id: 's', app_id: '..', messages: [] }),
    (error: unknown) => error instanceof EverosError && error.code === 'INVALID_SCOPE_ID',
  )
})
