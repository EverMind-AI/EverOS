/** Zero-dependency HTTP client for EverOS memory routes. */

import type {
  AddRequest,
  AddResponse,
  ApiVersion,
  ErrorBody,
  FlushRequest,
  FlushResponse,
  HealthResponse,
  SearchRequest,
  SearchResponse,
} from './types.js'

const PATH_SAFE = /^[a-zA-Z0-9_.@+-]+$/u

export class EverosError extends Error {
  constructor(
    readonly status: number,
    readonly code: string | undefined,
    message: string,
    readonly requestId?: string,
    readonly path?: string,
    options?: { cause?: unknown },
  ) {
    super(message, options)
    this.name = 'EverosError'
  }
}

export interface CallOptions {
  signal?: AbortSignal
  timeoutMs?: number
}

export interface EverosClientOptions {
  baseUrl: string
  apiVersion?: ApiVersion
  timeoutMs?: number
  fetch?: typeof fetch
}

export interface EverosClient {
  health(options?: CallOptions): Promise<HealthResponse>
  add(request: AddRequest, options?: CallOptions): Promise<AddResponse>
  search(request: SearchRequest, options?: CallOptions): Promise<SearchResponse>
  flush(request: FlushRequest, options?: CallOptions): Promise<FlushResponse>
  resolvedApiVersion(): Exclude<ApiVersion, 'auto'> | undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function assertScopeId(value: string | undefined, field: string): void {
  if (value === undefined) return
  if (value === '.' || value === '..' || value.length > 128 || !PATH_SAFE.test(value)) {
    throw new EverosError(0, 'INVALID_SCOPE_ID', `invalid ${field}: ${JSON.stringify(value)}`)
  }
}

function combinedSignal(
  configuredTimeout: number | undefined,
  options: CallOptions | undefined,
): AbortSignal | undefined {
  const timeoutMs = options?.timeoutMs ?? configuredTimeout
  const timeout =
    timeoutMs !== undefined && timeoutMs > 0 ? AbortSignal.timeout(timeoutMs) : undefined
  if (timeout && options?.signal) return AbortSignal.any([timeout, options.signal])
  return timeout ?? options?.signal
}

export function createEverosClient(options: EverosClientOptions): EverosClient {
  const baseUrl = options.baseUrl.replace(/\/+$/u, '')
  const doFetch = options.fetch ?? fetch
  const configuredVersion = options.apiVersion ?? 'auto'
  let negotiatedVersion: Exclude<ApiVersion, 'auto'> | undefined =
    configuredVersion === 'auto' ? undefined : configuredVersion

  async function call(
    method: 'GET' | 'POST',
    path: string,
    body: unknown,
    callOptions?: CallOptions,
  ): Promise<{ status: number; ok: boolean; parsed: unknown }> {
    let response: Response
    try {
      response = await doFetch(`${baseUrl}${path}`, {
        method,
        headers: body === undefined ? undefined : { 'content-type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: combinedSignal(options.timeoutMs, callOptions),
      })
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause)
      throw new EverosError(
        0,
        'NETWORK_ERROR',
        `${method} ${path} failed: ${message}`,
        undefined,
        path,
        { cause },
      )
    }

    const text = await response.text()
    let parsed: unknown
    if (text) {
      try {
        parsed = JSON.parse(text)
      } catch {
        throw new EverosError(
          response.status,
          'BAD_RESPONSE',
          `${method} ${path} returned non-JSON (HTTP ${response.status})`,
          undefined,
          path,
        )
      }
    }
    return { status: response.status, ok: response.ok, parsed }
  }

  async function enveloped<T>(path: string, body: unknown, callOptions?: CallOptions): Promise<T> {
    const result = await call('POST', path, body, callOptions)
    if (result.ok && isRecord(result.parsed) && 'data' in result.parsed) {
      return result.parsed.data as T
    }
    if (isRecord(result.parsed) && isRecord(result.parsed.error)) {
      const error = result.parsed.error as ErrorBody
      const requestId =
        typeof result.parsed.request_id === 'string' ? result.parsed.request_id : undefined
      throw new EverosError(
        result.status,
        error.code,
        error.message ?? `${path} failed (HTTP ${result.status})`,
        requestId,
        error.path ?? path,
      )
    }
    throw new EverosError(
      result.status,
      undefined,
      `${path} returned an unexpected response (HTTP ${result.status})`,
      undefined,
      path,
    )
  }

  async function memoryCall<T>(
    route: 'add' | 'search' | 'flush',
    body: unknown,
    callOptions?: CallOptions,
  ): Promise<T> {
    if (negotiatedVersion) {
      return enveloped<T>(`/api/${negotiatedVersion}/memory/${route}`, body, callOptions)
    }

    try {
      const value = await enveloped<T>(`/api/v2/memory/${route}`, body, callOptions)
      negotiatedVersion = 'v2'
      return value
    } catch (error) {
      if (!(error instanceof EverosError) || error.status !== 404) throw error
    }

    const value = await enveloped<T>(`/api/v1/memory/${route}`, body, callOptions)
    negotiatedVersion = 'v1'
    return value
  }

  return {
    async health(callOptions) {
      const result = await call('GET', '/health', undefined, callOptions)
      if (result.ok && isRecord(result.parsed) && result.parsed.status === 'ok') {
        return { status: 'ok' }
      }
      throw new EverosError(
        result.status,
        undefined,
        `/health returned an unexpected response (HTTP ${result.status})`,
        undefined,
        '/health',
      )
    },

    async add(request, callOptions) {
      assertScopeId(request.app_id, 'app_id')
      assertScopeId(request.project_id, 'project_id')
      return memoryCall<AddResponse>('add', request, callOptions)
    },

    async search(request, callOptions) {
      if ((request.user_id === undefined) === (request.agent_id === undefined)) {
        throw new EverosError(
          0,
          'INVALID_OWNER',
          'exactly one of user_id / agent_id must be provided',
        )
      }
      assertScopeId(request.app_id, 'app_id')
      assertScopeId(request.project_id, 'project_id')
      return memoryCall<SearchResponse>('search', request, callOptions)
    },

    async flush(request, callOptions) {
      assertScopeId(request.app_id, 'app_id')
      assertScopeId(request.project_id, 'project_id')
      return memoryCall<FlushResponse>('flush', request, callOptions)
    },

    resolvedApiVersion() {
      return negotiatedVersion
    },
  }
}
