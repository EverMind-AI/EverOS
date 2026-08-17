/** EverOS automatic long-term memory for DeepSeek Harness. */

import type { Context } from '@deepseek-ai/cordis'
import type { PreStepDecision } from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-session'

import { Config, type Config as PluginConfig, resolveConfig } from './config.js'
import { createEverosClient } from './everos-client.js'
import { resolveUserId } from './identity.js'
import { MemoryRuntime } from './memory-runtime.js'
import { type ProvisionResult, provision } from './provision.js'
import { PLUGIN_NAME, recallMessage } from './recall.js'
import type { PluginLogger } from './types.js'

export const name = PLUGIN_NAME
export const inject = ['agents']
export { Config }
export type { PluginConfig as EverosMemoryConfig }

export function apply(ctx: Context, input: PluginConfig): void {
  const config = resolveConfig(input)
  const logger: PluginLogger = {
    info: (message) => ctx.logger.info(message),
    warn: (message) => ctx.logger.warn(message),
    error: (message) => ctx.logger.error(message),
  }
  const userId = resolveUserId(config.userId)
  const client = createEverosClient({
    baseUrl: config.baseUrl,
    apiVersion: config.apiVersion,
  })
  const runtime = new MemoryRuntime({ client, config, userId, logger })

  ctx.on(
    'agent/pre-step',
    async ({ agent, signal, step }, next): Promise<PreStepDecision> => {
      const decision = await next()
      if (decision.kind === 'reject' || signal.aborted || step !== 1) return decision
      runtime.noteActivity(agent.session)
      try {
        await runtime.flushBeforeRecall(agent.session)
      } catch (error) {
        logger.warn(`everos-memory: session-switch flush failed open: ${String(error)}`)
      }
      try {
        const recalled = await recallMessage({
          client,
          config,
          userId,
          session: agent.session,
          messages: decision.messages,
          signal,
          logger,
        })
        if (!recalled) return decision
        return { kind: 'enter', messages: [...decision.messages, recalled] }
      } catch (error) {
        logger.warn(`everos-memory: recall failed open: ${String(error)}`)
        return decision
      }
    },
    { prepend: true },
  )

  ctx.on('agent/turn-stopping', async ({ agent }): Promise<void> => {
    try {
      await runtime.capture(agent.session)
    } catch (error) {
      logger.warn(`everos-memory: capture failed open: ${String(error)}`)
    }
  })

  ctx.on('session/disposed', (session) => {
    runtime.track(runtime.seal(session), 'session seal')
  })

  let stopping = false
  let inFlightStop: (() => void) | undefined
  const provisioned: Promise<ProvisionResult | undefined> = config.autoStart
    ? provision({
        baseUrl: config.baseUrl,
        apiVersion: config.apiVersion,
        startCommand: config.startCommand,
        cwd: config.everosDir,
        readinessTimeoutMs: config.readinessTimeoutMs,
        readinessIntervalMs: config.readinessIntervalMs,
        logger,
        client,
        onStop: (stop) => {
          inFlightStop = stop
          if (stopping) stop()
        },
      }).catch((error: unknown) => {
        logger.warn(`everos-memory: provisioning failed open: ${String(error)}`)
        return undefined
      })
    : Promise.resolve(undefined)

  ctx.effect(
    () => async () => {
      await runtime.dispose()
      stopping = true
      const result = await provisioned
      const stop = result?.stop ?? inFlightStop
      stop?.()
    },
    'everos-memory: drain capture and stop owned EverOS process',
  )

  logger.info(`everos-memory: active at ${config.baseUrl}, api=${config.apiVersion}`)
}
