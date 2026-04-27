import { Router, Request, Response } from 'express';
import { IMemoryService } from '../services/IMemoryService.js';
import { OpenAIService } from '../services/OpenAIService.js';

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  backend: 'ok';
  openai: 'ok' | 'error';
  memory: 'ok' | 'error';
  timestamp: string;
}

export function createHealthRouter(
  memoryService: IMemoryService,
  openaiService: OpenAIService
): Router {
  const router = Router();

  router.get('/health', async (_req: Request, res: Response) => {
    const health: HealthStatus = {
      status: 'healthy',
      backend: 'ok',
      openai: 'ok',
      memory: 'ok',
      timestamp: new Date().toISOString(),
    };

    try {
      // Check OpenAI
      const openaiAvailable = await openaiService.isAvailable();
      if (!openaiAvailable) {
        health.openai = 'error';
        health.status = 'degraded';
      }
    } catch {
      health.openai = 'error';
      health.status = 'degraded';
    }

    try {
      // Check Memory Service
      const memoryAvailable = await memoryService.isAvailable();
      if (!memoryAvailable) {
        health.memory = 'error';
        health.status = 'degraded';
      }
    } catch {
      health.memory = 'error';
      health.status = 'degraded';
    }

    // If both critical services are down, status is unhealthy
    if (health.openai === 'error' && health.memory === 'error') {
      health.status = 'unhealthy';
    }

    const statusCode = health.status === 'healthy' ? 200 : health.status === 'degraded' ? 200 : 503;
    res.status(statusCode).json(health);
  });

  return router;
}
