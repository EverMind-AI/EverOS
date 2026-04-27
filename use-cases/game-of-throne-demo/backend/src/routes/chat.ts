import { Router, Request, Response } from 'express';
import { IMemoryService } from '../services/IMemoryService.js';
import { OpenAIService, Message } from '../services/OpenAIService.js';
import { logger } from '../utils/logger.js';

interface ChatRequest {
  message: string;
  conversationHistory?: Message[];
}

export function createChatRouter(
  memoryService: IMemoryService,
  openaiService: OpenAIService
): Router {
  const router = Router();

  router.post('/chat', async (req: Request, res: Response) => {
    const startTime = Date.now();

    try {
      const { message, conversationHistory = [] } = req.body as ChatRequest;

      if (!message || typeof message !== 'string') {
        res.status(400).json({ error: 'Invalid request: message is required' });
        return;
      }

      logger.info('ChatRoute', `User query received: "${message}"`);

      // Set up SSE
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // Retrieve memories
      const memoryStartTime = Date.now();
      const memories = await memoryService.retrieveMemories(message, 5);
      const memoryTime = Date.now() - memoryStartTime;

      logger.info('MemoryService', `Retrieved ${memories.length} memories in ${memoryTime}ms`);

      // Send memories to client
      res.write(`data: ${JSON.stringify({ type: 'memories', memories })}\n\n`);

      // Stream LLM response
      logger.info('OpenAIService', 'Streaming response started');
      let tokenCount = 0;

      try {
        let fullResponse = '';
        for await (const token of openaiService.streamChatCompletion(
          message,
          memories,
          conversationHistory
        )) {
          res.write(`data: ${JSON.stringify({ type: 'token', token })}\n\n`);
          fullResponse += token;
          tokenCount++;
        }

        // Send done event
        res.write(`data: ${JSON.stringify({ type: 'done' })}\n\n`);

        const totalTime = Date.now() - startTime;
        logger.info('OpenAIService', `Response complete: ${tokenCount} tokens in ${totalTime}ms`);

        // Generate follow-up questions asynchronously
        try {
          const followUps = await openaiService.generateFollowUps(message, fullResponse, memories);
          if (followUps.length > 0) {
            res.write(`data: ${JSON.stringify({ type: 'followups', followUps })}\n\n`);
            logger.info('OpenAIService', `Generated ${followUps.length} follow-up questions`);
          }
        } catch (followUpError) {
          logger.error('OpenAIService', 'Error generating follow-ups:', followUpError);
          // Don't fail the response if follow-ups fail
        }

        res.end();
      } catch (streamError) {
        logger.error('OpenAIService', 'Streaming error:', streamError);
        res.write(
          `data: ${JSON.stringify({
            type: 'error',
            message: 'Unable to generate response. Please try again.'
          })}\n\n`
        );
        res.end();
      }
    } catch (error) {
      logger.error('ChatRoute', 'Error processing chat request:', error);

      // Try to send error if headers not sent
      if (!res.headersSent) {
        res.status(500).json({ error: 'Internal server error' });
      } else {
        res.write(
          `data: ${JSON.stringify({
            type: 'error',
            message: 'An error occurred processing your request.'
          })}\n\n`
        );
        res.end();
      }
    }
  });

  // Comparison endpoint - runs two parallel streams (with and without memory)
  router.post('/chat/compare', async (req: Request, res: Response) => {
    const startTime = Date.now();

    try {
      const { message, conversationHistory = [] } = req.body as ChatRequest;

      if (!message || typeof message !== 'string') {
        res.status(400).json({ error: 'Invalid request: message is required' });
        return;
      }

      logger.info('ChatRoute', `Comparison query received: "${message}"`);

      // Set up SSE
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // 1. Retrieve memories once
      const memoryStartTime = Date.now();
      const memories = await memoryService.retrieveMemories(message, 5);
      const memoryTime = Date.now() - memoryStartTime;

      logger.info('MemoryService', `Retrieved ${memories.length} memories in ${memoryTime}ms`);

      // Send memories to client
      res.write(`data: ${JSON.stringify({ type: 'memories', memories })}\n\n`);

      // 2. Run both streams in parallel
      let withMemoryResponse = '';

      const processStream = async (
        stream: AsyncIterable<string>,
        streamName: 'withMemory' | 'withoutMemory'
      ): Promise<string> => {
        let fullResponse = '';
        let tokenCount = 0;

        for await (const token of stream) {
          res.write(`data: ${JSON.stringify({ type: 'token', token, stream: streamName })}\n\n`);
          fullResponse += token;
          tokenCount++;
        }

        // Send stream-specific done event
        res.write(`data: ${JSON.stringify({ type: 'done', stream: streamName })}\n\n`);
        logger.info('OpenAIService', `${streamName} stream complete: ${tokenCount} tokens`);

        return fullResponse;
      };

      try {
        // Create both streams
        const withMemoryStream = openaiService.streamChatCompletion(message, memories, conversationHistory);
        const withoutMemoryStream = openaiService.streamChatCompletion(message, [], conversationHistory);

        // 3. Process concurrently with Promise.all
        logger.info('OpenAIService', 'Starting parallel streaming for comparison');
        const [withMemoryResult] = await Promise.all([
          processStream(withMemoryStream, 'withMemory'),
          processStream(withoutMemoryStream, 'withoutMemory')
        ]);

        withMemoryResponse = withMemoryResult;

        const totalTime = Date.now() - startTime;
        logger.info('OpenAIService', `Comparison complete in ${totalTime}ms`);

        // 4. Generate follow-ups for "with memory" response
        try {
          const followUps = await openaiService.generateFollowUps(message, withMemoryResponse, memories);
          if (followUps.length > 0) {
            res.write(`data: ${JSON.stringify({ type: 'followups', followUps })}\n\n`);
            logger.info('OpenAIService', `Generated ${followUps.length} follow-up questions`);
          }
        } catch (followUpError) {
          logger.error('OpenAIService', 'Error generating follow-ups:', followUpError);
        }

        // 5. Send complete event
        res.write(`data: ${JSON.stringify({ type: 'complete' })}\n\n`);
        res.end();
      } catch (streamError) {
        logger.error('OpenAIService', 'Comparison streaming error:', streamError);
        res.write(
          `data: ${JSON.stringify({
            type: 'error',
            message: 'Unable to generate comparison response. Please try again.'
          })}\n\n`
        );
        res.end();
      }
    } catch (error) {
      logger.error('ChatRoute', 'Error processing comparison request:', error);

      if (!res.headersSent) {
        res.status(500).json({ error: 'Internal server error' });
      } else {
        res.write(
          `data: ${JSON.stringify({
            type: 'error',
            message: 'An error occurred processing your request.'
          })}\n\n`
        );
        res.end();
      }
    }
  });

  return router;
}
