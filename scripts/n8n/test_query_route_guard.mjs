import assert from 'node:assert/strict';
import fs from 'node:fs';

const [, , workflowPath] = process.argv;
if (!workflowPath) throw new Error('Pass the hardened workflow JSON path');

const [workflow] = JSON.parse(fs.readFileSync(workflowPath, 'utf8'));
const guard = workflow.nodes.find((node) => node.name === 'KG路由校验与关键词清洗');
if (!guard) throw new Error('Route guard node was not found');

const execute = new Function('$input', guard.parameters.jsCode);
const input = [
  {
    json: {
      keyword: 'A213标准的使用产品',
      entity_type: 'product',
      method: { type: 'kg_search', params: { port: 'product-standards' } },
    },
  },
  {
    json: {
      keyword: '无缝管标准',
      entity_type: 'product',
      method: { type: 'kg_search', params: { port: 'standard-context' } },
    },
  },
  {
    json: {
      keyword: '法兰使用场景',
      entity_type: 'product',
      method: { type: 'kg_search', params: { port: 'standard-context' } },
    },
  },
  {
    json: {
      keyword: 'ASTM A312适用产品',
      entity_type: 'standard',
      method: { type: 'kg_search', params: { port: 'product-standards' } },
    },
  },
  {
    json: {
      keyword: '未知对象的关联',
      entity_type: 'none',
      method: { type: 'kg_search', params: { port: '' } },
    },
  },
  {
    json: {
      keyword: 'A213',
      entity_type: 'standard',
      method: { type: 'kg_search', params: { port: 'standard-context' } },
    },
  },
  {
    json: {
      keyword: '员工报销流程',
      entity_type: 'product',
      method: { type: 'hybrid_search', params: { port: 'product-standards' } },
    },
  },
];

const output = execute({ all: () => input }).map((item) => item.json);

assert.equal(output.length, 6, 'duplicate A213 route should be removed');
assert.deepEqual(
  output.slice(0, 5).map((item) => [
    item.keyword,
    item.entity_type,
    item.method.type,
    item.method.params.port,
  ]),
  [
    ['A213', 'standard', 'kg_search', 'standard-context'],
    ['无缝管', 'product', 'kg_search', 'product-standards'],
    ['法兰', 'product', 'kg_search', 'product-standards'],
    ['A312', 'standard', 'kg_search', 'standard-context'],
    ['未知对象的关联', 'none', 'hybid_search', ''],
  ],
);
assert.deepEqual(
  [
    output[5].keyword,
    output[5].entity_type,
    output[5].method.type,
    output[5].method.params.port,
  ],
  ['员工报销流程', 'none', 'hybid_search', ''],
);

console.log(JSON.stringify(output, null, 2));
