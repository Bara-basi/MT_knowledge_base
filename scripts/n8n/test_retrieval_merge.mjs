import assert from 'node:assert/strict';
import fs from 'node:fs';

const [, , kgWorkflowPath, mainWorkflowPath] = process.argv;
if (!kgWorkflowPath || !mainWorkflowPath) {
  throw new Error('Pass the normalized KG workflow and merged main workflow paths');
}

function loadWorkflow(filePath) {
  const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  return Array.isArray(parsed) ? parsed[0] : parsed;
}

function executeCode(workflow, nodeName, rows) {
  const node = workflow.nodes.find((item) => item.name === nodeName);
  if (!node) throw new Error(`Node not found: ${nodeName}`);
  const execute = new Function('$input', node.parameters.jsCode);
  return execute({ all: () => rows.map((json) => ({ json })) });
}

const kgWorkflow = loadWorkflow(kgWorkflowPath);
const mainWorkflow = loadWorkflow(mainWorkflowPath);

const productOutput = executeCode(kgWorkflow, '规范化图谱召回结果', [
  {
    keyword: '无缝管',
    match_mode: 'exact',
    matched_products: [
      {
        id: 'product:seamless_pipe',
        name: 'Seamless Pipe',
        labels: ['GraphNode', 'Product'],
        properties: { chinese_name: '无缝管（Pipe）', source_file: 'ignored.docx' },
      },
    ],
    standards: ['SA-312/SA-312M', 'SA-213/SA-213M'],
  },
]);

assert.equal(productOutput.length, 2);
assert.deepEqual(
  productOutput.map((item) => item.json.graph.name),
  ['SA-312/SA-312M', 'SA-213/SA-213M'],
);
assert.ok(productOutput.every((item) => item.json.chunk_type === 'kg_standard'));
assert.ok(productOutput.every((item) => item.json.file_path === ''));
assert.ok(productOutput.every((item) => item.json.citation_allowed === false));
assert.ok(productOutput.every((item) => !item.json.content_text.includes('Seamless Pipe')));

const standardOutput = executeCode(kgWorkflow, '规范化图谱召回结果', [
  {
    keyword: 'A213',
    match_mode: 'exact',
    matched_standards: [
      {
        id: 'standard:sa_213',
        name: 'SA-213/SA-213M',
        labels: ['GraphNode', 'Standard'],
        properties: {
          code: 'SA-213/SA-213M',
          standard_title: 'Tube standard',
          standard_version: 'ASME2023',
          aliases: ['A213'],
          graph_name: 'ignored',
        },
      },
    ],
    referenced_standards: [],
    products: [
      {
        id: 'product:seamless_tube',
        name: 'Seamless Tube',
        labels: ['GraphNode', 'Product'],
        properties: { chinese_name: '无缝管（Tube）', product_type: 'tube' },
      },
    ],
    documents: [
      {
        id: 'document:sa_213',
        name: 'SA-213.pdf',
        labels: ['GraphNode', 'Document'],
        properties: {
          file_path: 'minio://knowledge-raw-docs/SA-213.pdf',
          document_level: 'sub_document',
          create_at: 'ignored',
        },
      },
    ],
    sections: [],
    tables: [
      {
        id: 'table:chemical',
        name: 'table_chemical',
        labels: ['GraphNode', 'Table'],
        properties: {
          title: 'Chemical composition',
          file_path: 'minio://knowledge-standard-assets/table.png',
        },
      },
    ],
    versions: [{
      id: 'standardversion:asme2023',
      name: 'ASME2023',
      labels: ['GraphNode', 'StandardVersion'],
      properties: {},
    }],
  },
]);

assert.deepEqual(
  standardOutput.map((item) => item.json.chunk_type),
  ['kg_standard', 'kg_product', 'kg_document', 'kg_table', 'kg_standardversion'],
);
assert.equal(standardOutput[0].json.citation_allowed, false);
assert.equal(standardOutput[1].json.citation_allowed, false);
assert.equal(standardOutput[2].json.citation_allowed, true);
assert.equal(standardOutput[3].json.citation_allowed, true);
assert.equal(standardOutput[3].json.imgs.length, 1);
assert.ok(!standardOutput[0].json.content_text.includes('aliases'));
assert.ok(!standardOutput[0].json.content_text.includes('graph_name'));

const hybridRow = {
  chunk_id: 'hybrid:1',
  content_text: 'Hybrid document content',
  chunk_type: 'text',
  file_name: 'guide.docx',
  file_path: 'data/guide.docx',
  imgs: [],
  links: [],
};

const mainOutput = executeCode(
  mainWorkflow,
  '合并item并去重',
  [hybridRow, ...productOutput.map((item) => item.json), ...standardOutput.map((item) => item.json)],
);

assert.equal(mainOutput.length, 1);
assert.equal(mainOutput[0].json.retrieval_summary.hybrid_count, 1);
assert.equal(mainOutput[0].json.retrieval_summary.kg_count, 7);
assert.equal(mainOutput[0].json.retrieval_summary.citeable_count, 3);
assert.ok(mainOutput[0].json.prompt_text.includes('文件路径为空或“可引用=否”时不要引用'));
assert.ok(mainOutput[0].json.prompt_text.includes('Hybrid document content'));
assert.ok(mainOutput[0].json.prompt_text.includes('SA-312/SA-312M'));

const executeKgNode = mainWorkflow.nodes.find((node) => node.name === '前往图谱检索');
assert.equal(executeKgNode.parameters.workflowInputs.value.keyword, '={{ $json.keyword }}');
assert.equal(executeKgNode.parameters.workflowInputs.value.port, '={{ $json.method.params.port }}');

console.log(JSON.stringify({
  product_standard_items: productOutput.length,
  standard_context_items: standardOutput.length,
  merged_summary: mainOutput[0].json.retrieval_summary,
}, null, 2));

if (process.argv.includes('--live')) {
  const [liveProductResponse, liveStandardResponse] = await Promise.all([
    fetch('http://localhost:8000/prod/api/v1/graph/product-standards', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ keyword: '无缝管', limit: 5 }),
    }),
    fetch('http://localhost:8000/prod/api/v1/graph/standard-context', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ keyword: 'A213', limit: 5 }),
    }),
  ]);

  assert.equal(liveProductResponse.ok, true);
  assert.equal(liveStandardResponse.ok, true);

  const liveProduct = await liveProductResponse.json();
  const liveStandard = await liveStandardResponse.json();
  const normalizedLiveProduct = executeCode(
    kgWorkflow,
    '规范化图谱召回结果',
    [liveProduct],
  );
  const normalizedLiveStandard = executeCode(
    kgWorkflow,
    '规范化图谱召回结果',
    [liveStandard],
  );

  assert.ok(normalizedLiveProduct.length > 0);
  assert.ok(normalizedLiveProduct.every((item) => item.json.chunk_type === 'kg_standard'));
  assert.ok(normalizedLiveStandard.some((item) => item.json.chunk_type === 'kg_document'));
  assert.ok(normalizedLiveStandard.some((item) => item.json.chunk_type === 'kg_table'));

  console.log(JSON.stringify({
    live_product_items: normalizedLiveProduct.length,
    live_standard_context_items: normalizedLiveStandard.length,
    live_standard_context_types: [...new Set(
      normalizedLiveStandard.map((item) => item.json.chunk_type),
    )],
  }, null, 2));
}
