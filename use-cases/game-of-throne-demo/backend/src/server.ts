import express from 'express';
import cors from 'cors';
import { MockMemoryService } from './services/MockMemoryService.js';
import { EverCoreService } from './services/EverMemOSService.js';
import { OpenAIService } from './services/OpenAIService.js';
import { createChatRouter } from './routes/chat.js';
import { createHealthRouter } from './routes/health.js';
import { logger } from './utils/logger.js';

// Environment variables
const PORT = process.env.PORT || 3001;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'openai/gpt-5.2';
const FRONTEND_URL = process.env.FRONTEND_URL || 'http://localhost:3000';
const USE_EVERMEMOS = process.env.USE_EVERMEMOS === 'true';
const EVERMEMOS_URL = process.env.EVERMEMOS_URL || 'http://localhost:1995';
const EVERMEMOS_API_KEY = process.env.EVERMEMOS_API_KEY || '';
const EVERMEMOS_GROUP_ID = process.env.EVERMEMOS_GROUP_ID || 'asoiaf';

if (!OPENAI_API_KEY) {
  logger.error('Server', 'OPENAI_API_KEY environment variable is not set (use OpenRouter API key)');
  process.exit(1);
}

// Initialize services
const memoryService = USE_EVERMEMOS
  ? new EverCoreService({
      baseUrl: EVERMEMOS_URL,
      apiKey: EVERMEMOS_API_KEY || undefined,
      groupId: EVERMEMOS_GROUP_ID,
    })
  : new MockMemoryService();
const openaiService = new OpenAIService(OPENAI_API_KEY, OPENAI_MODEL);

const isCloudMode = USE_EVERMEMOS && !!EVERMEMOS_API_KEY;
logger.info('Server', `Memory service: ${USE_EVERMEMOS ? (isCloudMode ? 'EverMind Cloud' : 'EverCore (local)') : 'Mock'}`);
if (USE_EVERMEMOS) {
  logger.info('Server', `EverCore URL: ${EVERMEMOS_URL}`);
  if (isCloudMode) {
    logger.info('Server', `EverMind Cloud API Key: ${EVERMEMOS_API_KEY.slice(0, 8)}...`);
  }
}

// Create Express app
const app = express();

// Middleware
app.use(cors({
  origin: FRONTEND_URL === '*' ? true : FRONTEND_URL,
}));
app.use(express.json());

// Request logging middleware
app.use((req, _res, next) => {
  logger.info('Server', `${req.method} ${req.path}`);
  next();
});

// Routes
app.use('/api', createChatRouter(memoryService, openaiService));
app.use('/api', createHealthRouter(memoryService, openaiService));

// Error handling middleware
app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  logger.error('Server', 'Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, () => {
  logger.info('Server', `Backend server running on http://localhost:${PORT}`);
  logger.info('Server', `CORS enabled for: ${FRONTEND_URL}`);
  logger.info('Server', `Using OpenRouter with model: ${OPENAI_MODEL}`);
});
