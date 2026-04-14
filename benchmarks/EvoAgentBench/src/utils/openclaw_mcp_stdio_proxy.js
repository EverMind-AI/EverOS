#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) {
      continue;
    }
    const key = token.slice(2);
    const value = argv[i + 1];
    if (value === undefined || value.startsWith('--')) {
      out[key] = 'true';
      continue;
    }
    out[key] = value;
    i += 1;
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const remoteUrl = args.url || process.env.MCP_REMOTE_URL;
const transportType = args.transport || process.env.MCP_REMOTE_TRANSPORT || 'auto';
const nodeModulesRoot = args['node-modules'] || process.env.OPENCLAW_MCP_PROXY_NODE_MODULES;

function fail(message) {
  process.stderr.write(`[openclaw-mcp-proxy] ${message}\n`);
  process.exit(1);
}

if (!remoteUrl) {
  fail('missing --url');
}
if (!nodeModulesRoot) {
  fail('missing --node-modules');
}

const sdkRoot = path.join(nodeModulesRoot, '@modelcontextprotocol', 'sdk');
const zodRoot = path.join(nodeModulesRoot, 'zod');
if (!fs.existsSync(sdkRoot)) {
  fail(`sdk root not found: ${sdkRoot}`);
}
if (!fs.existsSync(zodRoot)) {
  fail(`zod root not found: ${zodRoot}`);
}

const { Client } = require(path.join(sdkRoot, 'dist', 'cjs', 'client', 'index.js'));
const { SSEClientTransport } = require(path.join(sdkRoot, 'dist', 'cjs', 'client', 'sse.js'));
const { StreamableHTTPClientTransport } = require(path.join(sdkRoot, 'dist', 'cjs', 'client', 'streamableHttp.js'));
const { McpServer } = require(path.join(sdkRoot, 'dist', 'cjs', 'server', 'mcp.js'));
const { StdioServerTransport } = require(path.join(sdkRoot, 'dist', 'cjs', 'server', 'stdio.js'));
const z = require(path.join(zodRoot, 'v4'));

function log(message) {
  process.stderr.write(`[openclaw-mcp-proxy] ${message}\n`);
}

function applyDescription(schema, description) {
  return description ? schema.describe(description) : schema;
}

function jsonSchemaToZod(schema, depth = 0) {
  if (!schema || typeof schema !== 'object' || depth > 20) {
    return z.any();
  }

  if (Array.isArray(schema.oneOf) && schema.oneOf.length > 0) {
    return z.union(schema.oneOf.map((entry) => jsonSchemaToZod(entry, depth + 1)));
  }

  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    const nonNull = schema.anyOf.filter((entry) => entry && entry.type !== 'null');
    const hasNull = nonNull.length !== schema.anyOf.length;
    let base;
    if (nonNull.length === 0) {
      base = z.any();
    } else if (nonNull.length === 1) {
      base = jsonSchemaToZod(nonNull[0], depth + 1);
    } else {
      base = z.union(nonNull.map((entry) => jsonSchemaToZod(entry, depth + 1)));
    }
    return hasNull ? base.nullable() : base;
  }

  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    const literals = schema.enum.map((value) => z.literal(value));
    const union = literals.length === 1 ? literals[0] : z.union(literals);
    return applyDescription(union, schema.description);
  }

  if (Array.isArray(schema.type)) {
    const nonNull = schema.type.filter((entry) => entry !== 'null');
    const hasNull = nonNull.length !== schema.type.length;
    let base;
    if (nonNull.length === 0) {
      base = z.any();
    } else if (nonNull.length === 1) {
      base = jsonSchemaToZod({ ...schema, type: nonNull[0] }, depth + 1);
    } else {
      base = z.union(nonNull.map((entry) => jsonSchemaToZod({ ...schema, type: entry }, depth + 1)));
    }
    return hasNull ? base.nullable() : base;
  }

  let converted;
  switch (schema.type) {
    case 'string':
      converted = z.string();
      break;
    case 'number':
      converted = z.number();
      break;
    case 'integer':
      converted = z.number().int();
      break;
    case 'boolean':
      converted = z.boolean();
      break;
    case 'array':
      converted = z.array(jsonSchemaToZod(schema.items || {}, depth + 1));
      break;
    case 'object':
    default: {
      const properties = schema.properties && typeof schema.properties === 'object' ? schema.properties : {};
      const required = new Set(Array.isArray(schema.required) ? schema.required : []);
      const shape = {};
      for (const [key, value] of Object.entries(properties)) {
        const child = jsonSchemaToZod(value, depth + 1);
        shape[key] = required.has(key) ? child : child.optional();
      }
      converted = z.object(shape);
      if (schema.additionalProperties === false) {
        converted = converted.strict();
      } else {
        converted = converted.passthrough();
      }
      break;
    }
  }

  return applyDescription(converted, schema.description);
}

async function connectClient(url, kind) {
  const client = new Client({ name: 'openclaw-mcp-proxy', version: '1.0.0' }, { capabilities: {} });
  const makeTransport = (transportKind) => {
    if (transportKind === 'sse') {
      return new SSEClientTransport(new URL(url));
    }
    if (transportKind === 'streamable-http') {
      return new StreamableHTTPClientTransport(new URL(url));
    }
    throw new Error(`unsupported transport: ${transportKind}`);
  };

  const connectOnce = async (transportKind) => {
    const transport = makeTransport(transportKind);
    client.onerror = (error) => log(`client error: ${error.stack || error.message}`);
    transport.onerror = (error) => log(`transport error: ${error.stack || error.message}`);
    transport.onclose = () => log(`transport closed: ${transportKind}`);
    await client.connect(transport);
    return transport;
  };

  if (kind === 'sse' || kind === 'streamable-http') {
    return { client, transport: await connectOnce(kind), transportKind: kind };
  }

  try {
    return { client, transport: await connectOnce('streamable-http'), transportKind: 'streamable-http' };
  } catch (error) {
    log(`streamable-http connect failed, falling back to sse: ${error.message || error}`);
    return { client, transport: await connectOnce('sse'), transportKind: 'sse' };
  }
}

async function main() {
  const { client, transport: remoteTransport, transportKind } = await connectClient(remoteUrl, transportType);
  const toolsResult = await client.listTools();
  const tools = Array.isArray(toolsResult.tools) ? toolsResult.tools : [];
  log(`connected via ${transportKind}; exporting ${tools.length} tool(s) from ${remoteUrl}`);

  const server = new McpServer({ name: 'openclaw-mcp-stdio-proxy', version: '1.0.0' });
  for (const tool of tools) {
    const inputSchema = jsonSchemaToZod(tool.inputSchema || { type: 'object' });
    server.registerTool(
      tool.name,
      {
        title: tool.title,
        description: tool.description,
        inputSchema,
        annotations: tool.annotations,
        _meta: tool._meta,
      },
      async (args) => client.callTool({ name: tool.name, arguments: args || {} }),
    );
  }

  const stdioTransport = new StdioServerTransport();
  stdioTransport.onerror = (error) => log(`stdio error: ${error.stack || error.message}`);
  stdioTransport.onclose = () => log('stdio closed');
  await server.connect(stdioTransport);
  log('stdio proxy ready');

  const shutdown = async (signal) => {
    log(`received ${signal}, shutting down`);
    try {
      await server.close();
    } catch (error) {
      log(`server close error: ${error.stack || error.message}`);
    }
    try {
      await remoteTransport.close();
    } catch (error) {
      log(`remote close error: ${error.stack || error.message}`);
    }
    process.exit(0);
  };

  process.on('SIGINT', () => void shutdown('SIGINT'));
  process.on('SIGTERM', () => void shutdown('SIGTERM'));
}

main().catch((error) => {
  fail(error.stack || error.message);
});
