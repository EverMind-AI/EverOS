/** Lossless-enough DSH event mapping for EverOS user and agent extraction. */

import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { Session } from '@deepseek-ai/dsh-session'

import { agentIdOf } from './identity.js'
import { blocksToText, isDirectUserMessage } from './recall.js'
import type { MessageItem, ToolCall } from './types.js'

export interface CapturedMessage {
  seq: number
  item: MessageItem
}

export interface CaptureSlice {
  messages: CapturedMessage[]
  scannedThroughSeq: number
}

function clipText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text
  const marker = '\n[truncated by everos-memory]'
  return `${text.slice(0, Math.max(0, maxChars - marker.length))}${marker}`
}

function boundedArguments(argumentsText: string, maxChars: number): string {
  if (argumentsText.length <= maxChars) return argumentsText
  return JSON.stringify({
    everos_truncated: true,
    preview: argumentsText.slice(0, Math.max(0, maxChars - 100)),
  })
}

function toolCalls(blocks: readonly ContentBlock[], maxChars: number): ToolCall[] | undefined {
  const calls = blocks.flatMap((block) => {
    if (block.type !== 'tool-call') return []
    return [
      {
        id: String(block.id),
        type: 'function',
        function: {
          name: block.name,
          arguments: boundedArguments(block.arguments, maxChars),
        },
      },
    ]
  })
  return calls.length === 0 ? undefined : calls
}

function toolNameIndex(session: Session): Map<string, string> {
  const names = new Map<string, string>()
  for (const event of session.events) {
    if (event.type === 'tool/call') names.set(String(event.data.callId), event.data.name)
  }
  return names
}

/**
 * Convert new model-visible events after `afterSeq`.
 *
 * Raw reasoning chunks and plugin-injected context are intentionally excluded.
 * Assistant tool calls and tool results are retained because EverOS uses them to
 * extract reusable agent cases and skills.
 */
export function captureMessagesSince(
  session: Session,
  afterSeq: number,
  userId: string,
  configuredAgentId: string | undefined,
  maxChars: number,
): CaptureSlice {
  const agentId = agentIdOf(session, configuredAgentId)
  const names = toolNameIndex(session)
  const output: CapturedMessage[] = []
  let scannedThroughSeq = afterSeq

  for (const event of session.events) {
    if (event.seq <= afterSeq) continue
    scannedThroughSeq = Math.max(scannedThroughSeq, event.seq)
    const timestamp = Number.isSafeInteger(event.time) && event.time > 0 ? event.time : Date.now()

    switch (event.type) {
      case 'user/message': {
        if (!isDirectUserMessage(event.data)) break
        const content = clipText(blocksToText(event.data.content), maxChars)
        if (!content) break
        output.push({
          seq: event.seq,
          item: {
            sender_id: userId,
            role: 'user',
            timestamp,
            content,
          },
        })
        break
      }

      case 'assistant/message': {
        const message = event.data.message
        const calls = toolCalls(message.content, maxChars)
        const content = clipText(blocksToText(message.content), maxChars)
        if (!content && !calls) break
        output.push({
          seq: event.seq,
          item: {
            sender_id: agentId,
            sender_name: `${message.source.provider}/${message.source.model}`,
            role: 'assistant',
            timestamp,
            content,
            ...(calls ? { tool_calls: calls } : {}),
          },
        })
        break
      }

      case 'tool/result': {
        const message = event.data.message
        const block = message.content[0]
        const callId = String(block.toolCallId)
        const resultText = blocksToText(block.content)
        const status = block.isError || event.data.error ? '[tool error]\n' : ''
        const content = clipText(`${status}${resultText}`, maxChars).trim() || '[empty tool result]'
        output.push({
          seq: event.seq,
          item: {
            sender_id: agentId,
            sender_name: names.get(callId) ?? 'tool',
            role: 'tool',
            timestamp,
            content,
            tool_call_id: callId,
          },
        })
        break
      }

      default:
        break
    }
  }

  return { messages: output, scannedThroughSeq }
}
