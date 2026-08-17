/** Wire contracts shared with the EverOS memory HTTP API. */

export type ApiVersion = 'auto' | 'v1' | 'v2'

export type SearchMethod = 'keyword' | 'vector' | 'hybrid' | 'agentic'

export type Role = 'user' | 'assistant' | 'tool'

export interface ContentItem {
  type: 'text' | 'image' | 'audio' | 'doc' | 'pdf' | 'html' | 'email'
  text?: string
  uri?: string
  base64?: string
  ext?: string
  name?: string
  extras?: Record<string, unknown>
}

export interface ToolCall {
  id: string
  type?: string
  function: {
    name: string
    arguments: string
  }
}

export interface MessageItem {
  sender_id: string
  sender_name?: string
  role: Role
  timestamp: number
  content: string | ContentItem[]
  tool_calls?: ToolCall[]
  tool_call_id?: string
}

export interface AddRequest {
  session_id: string
  app_id?: string
  project_id?: string
  messages: MessageItem[]
  /** Persist into EverOS's durable buffer without running extraction. */
  defer_extraction?: boolean
}

export interface AddResponse {
  message_count: number
  status: 'accumulated' | 'extracted'
}

export type SearchOwner =
  | { user_id: string; agent_id?: undefined }
  | { agent_id: string; user_id?: undefined }

export interface SearchRequestBase {
  app_id?: string
  project_id?: string
  query: string
  method?: SearchMethod
  top_k?: number
  radius?: number
  min_score?: number
  include_profile?: boolean
  enable_llm_rerank?: boolean
  filters?: unknown
}

export type SearchRequest = SearchOwner & SearchRequestBase

export interface SearchAtomicFact {
  id: string
  content: string
  score: number
}

export interface SearchEpisode {
  id: string
  subject: string
  summary: string
  episode: string
  score: number
  atomic_facts?: SearchAtomicFact[]
}

export interface SearchProfile {
  id: string
  profile_data: Record<string, unknown>
  score?: number | null
}

export interface SearchAgentCase {
  id: string
  task_intent: string
  approach: string
  quality_score: number
  key_insight?: string | null
  score: number
}

export interface SearchAgentSkill {
  id: string
  name: string
  description: string
  content: string
  confidence: number
  maturity_score: number
  score: number
}

export interface SearchResponse {
  episodes: SearchEpisode[]
  profiles: SearchProfile[]
  agent_cases: SearchAgentCase[]
  agent_skills: SearchAgentSkill[]
  unprocessed_messages: unknown[]
}

export interface FlushRequest {
  session_id: string
  app_id?: string
  project_id?: string
}

export interface FlushResponse {
  status: 'extracted' | 'no_extraction'
}

export interface HealthResponse {
  status: 'ok'
}

export interface ErrorBody {
  code?: string
  message?: string
  path?: string
}

export interface PluginLogger {
  info(message: string): void
  warn(message: string): void
  error?(message: string): void
}
