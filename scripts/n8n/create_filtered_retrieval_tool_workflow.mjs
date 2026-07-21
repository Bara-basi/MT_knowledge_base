import fs from 'node:fs';
import crypto from 'node:crypto';

const outputPath = process.argv[2];
if (!outputPath) {
  throw new Error('Usage: node create_filtered_retrieval_tool_workflow.mjs <output.json>');
}

const id = () => crypto.randomUUID();

const workflow = {
  id: 'FilterRetToolP01',
  name: '按文档过滤二次检索-prod',
  active: false,
  nodes: [
    {
      parameters: {
        workflowInputs: {
          values: [
            { name: 'query' },
            { name: 'file_path' },
            { name: 'chunk_type' },
            { name: 'path_prefix' },
            { name: 'allowed_paths' },
          ],
        },
      },
      type: 'n8n-nodes-base.executeWorkflowTrigger',
      typeVersion: 1.1,
      position: [0, 0],
      id: id(),
      name: 'When Executed by Another Workflow',
    },
    {
      parameters: {
        jsCode: `const input = $input.first().json || {};
const requested = String(input.file_path || '').trim();
let allowed = input.allowed_paths;
if (typeof allowed === 'string') {
  try { allowed = JSON.parse(allowed); } catch { allowed = []; }
}
allowed = Array.isArray(allowed) ? allowed.map((value) => String(value || '').trim()) : [];
if (!requested || !allowed.includes(requested)) {
  throw new Error('Requested file_path is not present in the current retrieval context');
}

const query = String(input.query || '').trim();
if (!query) throw new Error('Missing focused retrieval query');

const body = {
  query,
  file_path: requested,
  limit: 8,
  rerank: true,
};
const chunkType = String(input.chunk_type || '').trim();
const pathPrefix = String(input.path_prefix || '').trim();
if (chunkType) body.chunk_type = chunkType;
if (pathPrefix) body.path_prefix = pathPrefix;

return [{ json: { body } }];`,
      },
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [240, 0],
      id: id(),
      name: 'validate_and_build_filter',
    },
    {
      parameters: {
        method: 'POST',
        url: 'http://api:8000/prod/api/v1/retrieval/filtered',
        sendHeaders: true,
        specifyHeaders: 'json',
        jsonHeaders: '{\n  "Content-Type": "application/json"\n}',
        sendBody: true,
        specifyBody: 'json',
        jsonBody: '={{ $json.body }}',
        options: { timeout: 300000 },
      },
      type: 'n8n-nodes-base.httpRequest',
      typeVersion: 4.4,
      position: [480, 0],
      id: id(),
      name: 'filtered_retrieval_request',
    },
    {
      parameters: {
        jsCode: `const response = $input.first().json || {};
const chunks = Array.isArray(response.chunks) ? response.chunks : [];
return [{
  json: {
    query: String(response.query || ''),
    count: chunks.length,
    chunks: chunks.map((chunk) => ({
      chunk_id: String(chunk.chunk_id || ''),
      content: String(chunk.content || ''),
      chunk_index: chunk.chunk_index ?? null,
      chunk_type: String(chunk.chunk_type || ''),
      file_name: String(chunk.file_name || ''),
      file_path: String(chunk.file_path || ''),
      path: String(chunk.path || ''),
      imgs: Array.isArray(chunk.imgs) ? chunk.imgs : [],
      links: Array.isArray(chunk.links) ? chunk.links : [],
      citation_allowed: Boolean(chunk.file_path),
    })),
  },
}];`,
      },
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [720, 0],
      id: id(),
      name: 'normalize_filtered_results',
    },
  ],
  connections: {
    'When Executed by Another Workflow': {
      main: [[{ node: 'validate_and_build_filter', type: 'main', index: 0 }]],
    },
    validate_and_build_filter: {
      main: [[{ node: 'filtered_retrieval_request', type: 'main', index: 0 }]],
    },
    filtered_retrieval_request: {
      main: [[{ node: 'normalize_filtered_results', type: 'main', index: 0 }]],
    },
    normalize_filtered_results: { main: [[]] },
  },
  settings: {
    executionOrder: 'v1',
    binaryMode: 'separate',
  },
};

fs.writeFileSync(outputPath, JSON.stringify([workflow]), 'utf8');
console.log(outputPath);
