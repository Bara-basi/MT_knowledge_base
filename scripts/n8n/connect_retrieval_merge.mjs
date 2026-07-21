import fs from 'node:fs';
import path from 'node:path';

const [, , kgInputArg, mainInputArg, kgOutputArg, mainOutputArg] = process.argv;

if (!kgInputArg || !mainInputArg || !kgOutputArg || !mainOutputArg) {
  throw new Error(
    'Usage: node scripts/n8n/connect_retrieval_merge.mjs ' +
      '<kg-input.json> <main-input.json> <kg-output.json> <main-output.json>',
  );
}

function loadWorkflow(filePath, expectedId) {
  const parsed = JSON.parse(fs.readFileSync(path.resolve(filePath), 'utf8'));
  const workflows = Array.isArray(parsed) ? parsed : [parsed];
  const workflow = workflows.find((item) => item.id === expectedId);
  if (!workflow) throw new Error(`Workflow not found: ${expectedId}`);
  return { workflows, workflow };
}

const kgData = loadWorkflow(kgInputArg, '3SGEWkY3IRTD72F5');
const mainData = loadWorkflow(mainInputArg, 'KZKRj0Y1QW2xTS0J');

const kgNormalizerCode = String.raw`const inputItems = $input.all();

function cleanString(value) {
  if (value === null || value === undefined) return '';
  return String(value).trim();
}

function compactObject(value) {
  const result = {};
  for (const [key, item] of Object.entries(value || {})) {
    if (item === null || item === undefined || item === '') continue;
    if (Array.isArray(item) && item.length === 0) continue;
    result[key] = item;
  }
  return result;
}

function stableKey(value) {
  return cleanString(value)
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 160) || 'unknown';
}

function fileNameFromPath(filePath) {
  const rawName = cleanString(filePath).split('/').pop() || '';
  if (!rawName) return '';
  try {
    return decodeURIComponent(rawName);
  } catch (error) {
    return rawName;
  }
}

function getEntityType(node, fallback) {
  const labels = Array.isArray(node && node.labels) ? node.labels : [];
  const label = labels.find((item) => item !== 'GraphNode');
  return cleanString(label || (node && node.properties && node.properties.label) || fallback)
    .toLowerCase();
}

function minimalProperties(node, entityType) {
  const properties = node && node.properties && typeof node.properties === 'object'
    ? node.properties
    : {};

  if (entityType === 'standard') {
    return compactObject({
      code: properties.code,
      title: properties.standard_title,
      version: properties.standard_version,
    });
  }

  if (entityType === 'product') {
    return compactObject({
      chinese_name: properties.chinese_name,
      product_type: properties.product_type,
      category: properties.category,
    });
  }

  if (entityType === 'document') {
    return compactObject({
      document_level: properties.document_level,
      source_volume: properties.source_volume,
    });
  }

  if (entityType === 'section' || entityType === 'table') {
    return compactObject({ title: properties.title });
  }

  return {};
}

function makeGraphItem(options) {
  const node = options.node || {};
  const properties = node.properties && typeof node.properties === 'object'
    ? node.properties
    : {};
  const entityType = options.entityType || getEntityType(node, 'entity');
  const entityId = cleanString(node.id || properties.id);
  const entityName = cleanString(node.name || properties.name);
  const filePath = cleanString(properties.file_path);
  const citationAllowed = Boolean(filePath);
  const details = minimalProperties(node, entityType);

  const graphData = compactObject({
    entity_type: entityType,
    relation: options.relation,
    entity_id: entityId,
    name: entityName,
    ...details,
  });

  const chunkId = [
    'kg',
    options.port,
    options.relation,
    entityId || stableKey(entityName),
  ].join(':');

  const tableImage = entityType === 'table' && filePath
    ? [{
        index: 0,
        img_name: cleanString(details.title) || entityName || fileNameFromPath(filePath),
        img_path: filePath,
      }]
    : [];

  return {
    json: {
      chunk_id: chunkId,
      content_text: JSON.stringify(graphData),
      chunk_index: null,
      chunk_type: 'kg_' + entityType,
      retrieval_type: 'kg_search',
      source_type: 'knowledge_graph',
      kg_port: options.port,
      query_keyword: options.keyword,
      match_mode: options.matchMode,
      file_name: citationAllowed ? fileNameFromPath(filePath) : '',
      file_path: filePath,
      path: 'KG:' + options.relation,
      imgs: tableImage,
      links: [],
      citation_allowed: citationAllowed,
      graph: graphData,
    },
  };
}

function normalizeProductStandards(raw) {
  const keyword = cleanString(raw.keyword);
  const matchMode = cleanString(raw.match_mode);
  const standards = Array.isArray(raw.standards) ? raw.standards : [];
  const seen = new Set();
  const output = [];

  for (const value of standards) {
    const standard = cleanString(value);
    if (!standard || seen.has(standard.toLowerCase())) continue;
    seen.add(standard.toLowerCase());

    const graphData = {
      entity_type: 'standard',
      relation: 'product_applies_to_standard',
      name: standard,
    };

    output.push({
      json: {
        chunk_id: 'kg:product-standards:standard:' + stableKey(standard),
        content_text: JSON.stringify(graphData),
        chunk_index: null,
        chunk_type: 'kg_standard',
        retrieval_type: 'kg_search',
        source_type: 'knowledge_graph',
        kg_port: 'product-standards',
        query_keyword: keyword,
        match_mode: matchMode,
        file_name: '',
        file_path: '',
        path: 'KG:product_applies_to_standard',
        imgs: [],
        links: [],
        citation_allowed: false,
        graph: graphData,
      },
    });
  }

  return output;
}

function normalizeStandardContext(raw) {
  const keyword = cleanString(raw.keyword);
  const matchMode = cleanString(raw.match_mode);
  const groups = [
    ['matched_standards', 'matched_standard', 'standard'],
    ['referenced_standards', 'referenced_standard', 'standard'],
    ['products', 'applicable_product', 'product'],
    ['documents', 'standard_document', 'document'],
    ['sections', 'document_section', 'section'],
    ['tables', 'document_table', 'table'],
    ['versions', 'standard_version', 'standardversion'],
  ];
  const output = [];

  for (const [field, relation, fallbackType] of groups) {
    const nodes = Array.isArray(raw[field]) ? raw[field] : [];
    for (const node of nodes) {
      if (!node || typeof node !== 'object') continue;
      output.push(makeGraphItem({
        node,
        relation,
        entityType: getEntityType(node, fallbackType),
        keyword,
        matchMode,
        port: 'standard-context',
      }));
    }
  }

  return output;
}

const output = [];

for (const item of inputItems) {
  const raw = item.json || {};
  const isProductStandards = Array.isArray(raw.standards) && !Array.isArray(raw.matched_standards);
  const normalized = isProductStandards
    ? normalizeProductStandards(raw)
    : normalizeStandardContext(raw);
  output.push(...normalized);
}

if (output.length > 0) return output;

return [{
  json: {
    chunk_id: 'kg:no-result',
    content_text: JSON.stringify({ status: 'no_result', source: 'knowledge_graph' }),
    chunk_index: null,
    chunk_type: 'kg_no_result',
    retrieval_type: 'kg_search',
    source_type: 'knowledge_graph',
    kg_port: '',
    query_keyword: '',
    match_mode: 'none',
    file_name: '',
    file_path: '',
    path: 'KG:no_result',
    imgs: [],
    links: [],
    citation_allowed: false,
    graph: { status: 'no_result' },
  },
}];`;

const mainMergeCode = String.raw`const inputItems = $input.all();

function cleanString(value) {
  if (value === null || value === undefined) return '';
  return String(value).trim();
}

function cleanNullableNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function cleanAssetArray(value, nameKey, pathKey) {
  if (!Array.isArray(value)) return [];

  return value
    .map((item, fallbackIndex) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return null;
      const assetPath = cleanString(item[pathKey]);
      if (!assetPath) return null;
      const indexNumber = Number(item.index);
      return {
        index: Number.isFinite(indexNumber) ? indexNumber : fallbackIndex,
        [nameKey]: cleanString(item[nameKey]) || assetPath,
        [pathKey]: assetPath,
      };
    })
    .filter(Boolean);
}

function dedupeAssets(items, pathKey, nameKey) {
  const seen = new Set();
  const result = [];

  for (const item of items) {
    const assetPath = cleanString(item[pathKey]);
    if (!assetPath || seen.has(assetPath)) continue;
    seen.add(assetPath);
    result.push({
      ...item,
      [nameKey]: cleanString(item[nameKey]) || assetPath,
      [pathKey]: assetPath,
    });
  }

  return result.sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
}

function inferRetrievalType(raw) {
  const explicit = cleanString(raw.retrieval_type || raw.retrieval_method);
  if (explicit) return explicit === 'hybrid_search' ? 'hybid_search' : explicit;
  return cleanString(raw.chunk_type).startsWith('kg_') ? 'kg_search' : 'hybid_search';
}

function normalizeRecord(raw, fallbackIndex) {
  const content = cleanString(raw.content_text) || cleanString(raw.content);
  const retrievalType = inferRetrievalType(raw);
  const filePath = cleanString(raw.file_path);
  const citationAllowed = raw.citation_allowed === undefined
    ? Boolean(filePath)
    : raw.citation_allowed === true && Boolean(filePath);
  const fallbackId = [
    retrievalType,
    cleanString(raw.chunk_type) || 'item',
    cleanString(raw.query_keyword),
    String(fallbackIndex),
  ].join(':');

  return {
    chunk_id: cleanString(raw.chunk_id) || fallbackId,
    content_text: content,
    chunk_index: cleanNullableNumber(raw.chunk_index),
    chunk_type: cleanString(raw.chunk_type),
    retrieval_type: retrievalType,
    source_type: cleanString(raw.source_type) || (retrievalType === 'kg_search' ? 'knowledge_graph' : 'knowledge_base'),
    kg_port: cleanString(raw.kg_port),
    query_keyword: cleanString(raw.query_keyword),
    match_mode: cleanString(raw.match_mode),
    file_name: citationAllowed ? cleanString(raw.file_name) : '',
    file_path: citationAllowed ? filePath : '',
    path: cleanString(raw.path),
    imgs: cleanAssetArray(raw.imgs, 'img_name', 'img_path'),
    links: cleanAssetArray(raw.links, 'link_name', 'link_path'),
    citation_allowed: citationAllowed,
    graph: raw.graph && typeof raw.graph === 'object' && !Array.isArray(raw.graph)
      ? raw.graph
      : null,
  };
}

function mergeRecord(existing, incoming) {
  return {
    ...existing,
    content_text: existing.content_text || incoming.content_text,
    chunk_index: existing.chunk_index ?? incoming.chunk_index,
    chunk_type: existing.chunk_type || incoming.chunk_type,
    retrieval_type: existing.retrieval_type || incoming.retrieval_type,
    source_type: existing.source_type || incoming.source_type,
    kg_port: existing.kg_port || incoming.kg_port,
    query_keyword: existing.query_keyword || incoming.query_keyword,
    match_mode: existing.match_mode || incoming.match_mode,
    file_name: existing.file_name || incoming.file_name,
    file_path: existing.file_path || incoming.file_path,
    path: existing.path || incoming.path,
    citation_allowed: existing.citation_allowed || incoming.citation_allowed,
    graph: existing.graph || incoming.graph,
    imgs: dedupeAssets([...existing.imgs, ...incoming.imgs], 'img_path', 'img_name'),
    links: dedupeAssets([...existing.links, ...incoming.links], 'link_path', 'link_name'),
  };
}

function formatLinks(links) {
  if (!links.length) return '';
  return '相关链接:\n' + links
    .map((item) => '- index=' + item.index + ': ' + item.link_name + ': ' + item.link_path)
    .join('\n');
}

function formatImages(imgs) {
  if (!imgs.length) return '';
  return '相关图片路径:\n' + imgs
    .map((item) => '- index=' + item.index + ': ' + item.img_name + ': ' + item.img_path)
    .join('\n');
}

function formatRecord(record, index) {
  const parts = [];
  parts.push('【召回项 ' + (index + 1) + '｜' + record.retrieval_type + '｜' + (record.chunk_type || 'unknown') + '】');
  if (record.query_keyword) parts.push('检索实体/检索词:\n' + record.query_keyword);
  if (record.kg_port) parts.push('图谱接口:\n' + record.kg_port);
  if (record.match_mode) parts.push('匹配方式:\n' + record.match_mode);
  if (record.path) parts.push('知识路径:\n' + record.path);
  parts.push('可引用:\n' + (record.citation_allowed ? '是' : '否'));
  if (record.file_name) parts.push('文件名:\n' + record.file_name);
  if (record.file_path) parts.push('文件路径:\n' + record.file_path);
  parts.push('数据/正文:\n' + record.content_text);

  const links = formatLinks(record.links);
  if (links) parts.push(links);
  const images = formatImages(record.imgs);
  if (images) parts.push(images);
  return parts.join('\n\n');
}

const byId = new Map();
let fallbackIndex = 0;

for (const item of inputItems) {
  const raw = item.json || {};
  const rows = Array.isArray(raw.chunks) ? raw.chunks : [raw];
  for (const row of rows) {
    const record = normalizeRecord(row || {}, fallbackIndex++);
    if (!record.content_text) continue;
    if (byId.has(record.chunk_id)) {
      byId.set(record.chunk_id, mergeRecord(byId.get(record.chunk_id), record));
    } else {
      byId.set(record.chunk_id, record);
    }
  }
}

const records = Array.from(byId.values()).sort((a, b) => {
  if (a.retrieval_type !== b.retrieval_type) {
    return a.retrieval_type.localeCompare(b.retrieval_type);
  }
  return a.chunk_id.localeCompare(b.chunk_id);
});

const retrievalContext = records.map(formatRecord).join('\n\n---\n\n');
const promptText = [
  '以下是 hybrid_search 与 kg_search 的统一召回数据。hybrid_search 项包含正文片段；kg_search 项是经过精简的图谱实体干数据。',
  '引用硬规则：只有“可引用=是”且文件路径非空的召回项才允许输出 <reference>file_path</Reference>。文件路径为空或“可引用=否”时不要引用，不得用实体名、标准名、file_name 或知识路径代替 file_path。',
  '表格节点的图片路径仅表示可读取的表格资产；当前阶段只按已提供的数据回答，不主动调用额外工具。',
  '',
  retrievalContext || '没有可用的召回结果。',
].join('\n');

return [{
  json: {
    prompt_text: promptText,
    retrieval_items: records,
    retrieval_summary: {
      total: records.length,
      hybrid_count: records.filter((item) => item.retrieval_type === 'hybid_search').length,
      kg_count: records.filter((item) => item.retrieval_type === 'kg_search').length,
      citeable_count: records.filter((item) => item.citation_allowed).length,
    },
  },
}];`;

const kgWorkflow = kgData.workflow;
const normalizerName = '规范化图谱召回结果';
let normalizer = kgWorkflow.nodes.find((node) => node.name === normalizerName);
if (!normalizer) {
  normalizer = {
    parameters: { jsCode: kgNormalizerCode },
    type: 'n8n-nodes-base.code',
    typeVersion: 2,
    position: [720, 0],
    id: 'dbad8099-3d3e-49ed-9e4e-8ae266e35eb0',
    name: normalizerName,
  };
  kgWorkflow.nodes.push(normalizer);
} else {
  normalizer.parameters.jsCode = kgNormalizerCode;
}

for (const sourceName of ['产品查标准', '标准查上下文']) {
  kgWorkflow.connections[sourceName] = {
    main: [[{ node: normalizerName, type: 'main', index: 0 }]],
  };
}
kgWorkflow.connections[normalizerName] = { main: [[]] };

const mainWorkflow = mainData.workflow;
const kgExecuteNode = mainWorkflow.nodes.find((node) => node.name === '前往图谱检索');
if (!kgExecuteNode) throw new Error('Main workflow node not found: 前往图谱检索');
kgExecuteNode.parameters.workflowInputs.mappingMode = 'defineBelow';
kgExecuteNode.parameters.workflowInputs.value = {
  keyword: '={{ $json.keyword }}',
  port: '={{ $json.method.params.port }}',
};
kgExecuteNode.parameters.workflowInputs.matchingColumns = ['keyword', 'port'];
kgExecuteNode.parameters.mode = 'each';

const mergeCodeNode = mainWorkflow.nodes.find((node) => node.name === '合并item并去重');
if (!mergeCodeNode) throw new Error('Main workflow node not found: 合并item并去重');
mergeCodeNode.parameters.jsCode = mainMergeCode;

const answerNode = mainWorkflow.nodes.find((node) => node.name === '检索问答');
if (!answerNode) throw new Error('Main workflow node not found: 检索问答');
let answerPrompt = answerNode.parameters.messages.messageValues[0].message;
answerPrompt = answerPrompt.replace(
  '5. 必须正确引用 chunk 来源',
  '5. 仅对 citation_allowed=true 且 file_path 非空的召回项生成引用',
);
answerPrompt = answerPrompt.replace(
  '2. 引用标签内容使用该 chunk 的 文件名（file_name）或文件路径（file_path），优先使用路径，没有找到时退回到文件名。',
  '2. 仅当召回项 citation_allowed=true 且 file_path 非空时才允许引用；引用标签中只使用该项的 file_path。',
);
answerPrompt = answerPrompt.replace(
  '5. 如果 文件路径（file_path）/文件名（file_name）都缺失，应仍然写出来源，并注明“来源路径缺失，无法跳转”。',
  '5. 如果 file_path 为空或 citation_allowed=false，不要输出 reference 标签；不得使用实体名、标准名、file_name 或知识路径冒充引用。',
);
answerPrompt = answerPrompt.replace(
  '2. 可追溯（必须有引用）',
  '2. 可追溯（存在可引用 file_path 时必须引用；file_path 为空时不引用）',
);
answerNode.parameters.messages.messageValues[0].message = answerPrompt;

fs.writeFileSync(path.resolve(kgOutputArg), JSON.stringify(kgData.workflows) + '\n', 'utf8');
fs.writeFileSync(path.resolve(mainOutputArg), JSON.stringify(mainData.workflows) + '\n', 'utf8');

console.log(JSON.stringify({
  kg: {
    id: kgWorkflow.id,
    nodes: kgWorkflow.nodes.length,
    output: path.resolve(kgOutputArg),
  },
  main: {
    id: mainWorkflow.id,
    nodes: mainWorkflow.nodes.length,
    output: path.resolve(mainOutputArg),
  },
}, null, 2));
