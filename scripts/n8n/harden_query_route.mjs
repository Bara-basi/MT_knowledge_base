import fs from 'node:fs';
import path from 'node:path';

const [, , inputArg, outputArg] = process.argv;

if (!inputArg || !outputArg) {
  throw new Error('Usage: node scripts/n8n/harden_query_route.mjs <input.json> <output.json>');
}

const inputPath = path.resolve(inputArg);
const outputPath = path.resolve(outputArg);
const workflows = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const workflowList = Array.isArray(workflows) ? workflows : [workflows];
const workflow = workflowList.find((item) => item.id === 'KZKRj0Y1QW2xTS0J');

if (!workflow) {
  throw new Error('Target workflow KZKRj0Y1QW2xTS0J was not found');
}

const systemPrompt = `=你是企业内部知识库的 Query 转写与检索路由 Agent。你不回答问题，只在一次调用中完成：补全多轮语境、生成检索词、选择检索方式。

优先级（从高到低）：
1. 保持用户原意，不编造实体；
2. kg_search 的 keyword 与实体类型、port 必须一致；
3. 在不重复同一检索目标的前提下提高召回。

仅可输出 JSON，禁止 Markdown 和解释。结构：
{
  "rewritten_queries": [
    {
      "keyword": "检索词",
      "entity_type": "product | standard | none",
      "method": {
        "type": "hybid_search | kg_search | daily_chat | out_of_scope",
        "params": { "port": "product-standards | standard-context | 空字符串" }
      },
      "confidence": 0.0,
      "reason": "简短依据"
    }
  ]
}

路由规则：
- kg_search 目前只支持两类确定的实体关联：
  1) 产品 -> 适用标准：entity_type=product，port=product-standards；
  2) 标准 -> 该标准的上下文、文档、产品等：entity_type=standard，port=standard-context。
- port 只由 keyword 的实体类型决定，不由用户句子里“想查什么”决定。只要 keyword 是 A213、A312、SA-213、B16.5 等标准实体，就必须走 standard-context，即使用户问“这个标准用于哪些产品”。
- 无法提取出一个明确产品或标准实体时，不得使用 kg_search，改用 hybid_search，entity_type=none，port=""。
- 产品/标准之外的问题使用 hybid_search。沿用现有拼写 hybid_search，不要输出 hybrid_search。
- daily_chat 或 out_of_scope 出现时，列表中不得再有其他类型。

kg_search 的 keyword 是实体键，不是查询句：
- 产品只保留产品名或图谱别名，例如：无缝管、法兰、焊管、板材、棒材、管件、线材。
- 标准只保留标准代号/名称，例如：A213、A312、SA-312、B16.5；删除“标准、规范、适用产品、使用场景、相关文档、上下文、是什么、有哪些”等意图词。
- 禁止输出：无缝管标准、法兰使用场景、A213标准、ASTM A312适用产品。
- 同一个 keyword + method + port 不得重复。

例子：
- “无缝管执行哪些标准？” -> keyword=无缝管，entity_type=product，kg_search，product-standards。
- “A213标准使用的产品是什么？” -> keyword=A213，entity_type=standard，kg_search，standard-context。
- “ASTM A312的适用范围和相关文档” -> keyword=A312，entity_type=standard，kg_search，standard-context。
- “法兰的使用场景是什么？” -> 该关系不在图谱现有能力内，使用 hybid_search，keyword=法兰的使用场景。
- “无缝管和法兰分别执行哪些标准？” -> 拆成两个 kg_search 项，keyword 分别为无缝管、法兰。

多轮与拆分：
- 当前问题省略主语时，用主题和历史对话补全为可独立检索的表达；上下文不确定时不要猜。
- 只有多个独立信息目标才拆分；不要为了召回而制造同义重复项。
- hybid_search 的 keyword 应保留回答问题所需的限定词，写成简洁、完整的检索语句。
- 纠正明显错别字，去掉寒暄和语气词，但不扩大或缩小问题范围。`;

const schemaExample = `{
  "rewritten_queries": [
    {
      "keyword": "无缝管",
      "entity_type": "product",
      "method": {
        "type": "kg_search",
        "params": {
          "port": "product-standards"
        }
      },
      "confidence": 0.95,
      "reason": "产品实体查询适用标准"
    }
  ]
}`;

const routeGuardCode = `const items = $input.all();

const PRODUCT_GROUPS = [
  { keyword: '对焊管件', aliases: ['锻制对焊管件', '对焊管件', '对焊件', 'butt weld fitting'] },
  { keyword: '高压管件', aliases: ['高压管件', '锻制管件', '承插管件', 'high pressure fitting'] },
  { keyword: '低压管件', aliases: ['低压管件', '铸造管件', '铸件管件', 'low pressure fitting'] },
  { keyword: '无缝管', aliases: ['无缝换热管', '无缝仪表管', '无缝钢管', '无缝管道', 'seamless pipe', 'seamless tube', 'smls pipe', 'smls tube', '无缝管'] },
  { keyword: '焊管', aliases: ['焊接换热管', '焊接仪表管', '焊接钢管', '焊接管道', 'welded pipe', 'welded tube', '焊管'] },
  { keyword: '不锈钢板', aliases: ['不锈钢板', '板卷', '卷板', '钢板', '板材', 'plate'] },
  { keyword: '不锈钢棒材', aliases: ['不锈钢棒材', '棒材', '圆钢', 'bar'] },
  { keyword: '线材', aliases: ['不锈钢线材', 'wire rod', '线材', '盘条', 'wire'] },
  { keyword: '法兰', aliases: ['法兰盘', 'flange', '法兰'] },
  { keyword: '管件', aliases: ['管件', 'fitting'] },
];

function text(value) {
  return String(value ?? '')
    .normalize('NFKC')
    .trim()
    .replace(/^[\\s“”‘’'\"\`]+|[\\s“”‘’'\"\`，。；;！？!?]+$/g, '');
}

function extractStandard(value) {
  const source = text(value).toUpperCase();
  const patterns = [
    /\\b(?:ASTM|ASME|ANSI)\\s+((?:S[AB]|A|B)\\s*[-_ ]?\\s*\\d{2,4}(?:M)?(?:\\s*[\\/_]\\s*(?:S[AB]|A|B)?\\s*[- ]?\\s*\\d{2,4}M?)?(?:\\.\\d+)?)\\b/i,
    /\\b((?:S[AB]|A|B)\\s*[-_ ]?\\s*\\d{2,4}(?:M)?(?:\\s*[\\/_]\\s*(?:S[AB]|A|B)?\\s*[- ]?\\s*\\d{2,4}M?)?(?:\\.\\d+)?)\\b/i,
    /\\b((?:ISO|EN|DIN|JIS|API|GB(?:\\/T)?|HG(?:\\/T)?|SH(?:\\/T)?|NB(?:\\/T)?)\\s*[- ]?\\s*\\d{2,6}(?:[.\\-]\\d+)*(?:\\s*[-:]\\s*\\d{2,4})?)\\b/i,
  ];

  for (const pattern of patterns) {
    const match = source.match(pattern);
    if (match) {
      return match[1]
        .replace(/\\s+/g, '')
        .replace(/_/g, '/')
        .replace(/([A-Z])-(?=\\d)/g, '$1-');
    }
  }
  return '';
}

function findKnownProduct(value) {
  const source = text(value).toLowerCase();
  let best = null;
  for (const group of PRODUCT_GROUPS) {
    for (const alias of group.aliases) {
      const normalizedAlias = alias.toLowerCase();
      if (!source.includes(normalizedAlias)) continue;
      if (!best || normalizedAlias.length > best.alias.length) {
        best = { alias: normalizedAlias, keyword: group.keyword };
      }
    }
  }
  return best?.keyword ?? '';
}

function cleanProduct(value) {
  let result = text(value)
    .replace(/^(?:关于|有关|查询|查找|请查询|请查找|了解)\\s*/u, '')
    .replace(/(?:的)?(?:相关)?(?:执行|采用|适用|对应|产品)?标准(?:是什么|有哪些|有哪几种|列表)?$/u, '')
    .replace(/(?:的)?(?:使用场景|应用场景|适用场景|适用范围|适用产品|相关产品|使用产品|上下文|相关文档|文档|资料)(?:是什么|有哪些|吗)?$/u, '')
    .replace(/(?:是什么|有哪些|有哪几种|吗|呢)$/u, '')
    .trim();
  return result || text(value);
}

function cleanStandardFallback(value) {
  return text(value)
    .replace(/^(?:关于|有关|查询|查找|请查询|请查找|ASTM\\s*)/iu, '')
    .replace(/(?:标准|规范)(?:的)?/gu, '')
    .replace(/(?:适用产品|相关产品|使用产品|使用场景|应用场景|适用范围|上下文|相关文档|文档|资料)(?:是什么|有哪些|吗)?$/u, '')
    .replace(/(?:是什么|有哪些|有哪几种|吗|呢)$/u, '')
    .trim();
}

const output = [];
const seen = new Set();

for (let index = 0; index < items.length; index += 1) {
  const original = items[index].json ?? {};
  const item = structuredClone(original);
  item.method = item.method && typeof item.method === 'object' ? item.method : {};
  item.method.params = item.method.params && typeof item.method.params === 'object'
    ? item.method.params
    : {};

  if (item.method.type === 'hybrid_search') item.method.type = 'hybid_search';

  const originalKeyword = text(item.keyword);
  let applied = 'none';

  if (item.method.type === 'kg_search') {
    const standard = extractStandard(originalKeyword);
    const product = findKnownProduct(originalKeyword);
    const hint = String(item.entity_type ?? '').toLowerCase();
    const currentPort = String(item.method.params.port ?? '');

    if (standard) {
      item.keyword = standard;
      item.entity_type = 'standard';
      item.method.params.port = 'standard-context';
      applied = 'standard-pattern';
    } else if (product) {
      item.keyword = product;
      item.entity_type = 'product';
      item.method.params.port = 'product-standards';
      applied = 'product-alias';
    } else if (hint === 'standard' || currentPort === 'standard-context') {
      item.keyword = cleanStandardFallback(originalKeyword);
      item.entity_type = 'standard';
      item.method.params.port = 'standard-context';
      applied = 'standard-hint';
    } else if (hint === 'product' || currentPort === 'product-standards') {
      item.keyword = cleanProduct(originalKeyword);
      item.entity_type = 'product';
      item.method.params.port = 'product-standards';
      applied = 'product-hint';
    } else {
      item.keyword = originalKeyword;
      item.entity_type = 'none';
      item.method.type = 'hybid_search';
      item.method.params.port = '';
      applied = 'ambiguous-to-hybrid';
    }
  } else {
    item.keyword = originalKeyword;
    item.entity_type = 'none';
    item.method.params.port = '';
  }

  if (!item.keyword) continue;

  const dedupeKey = [
    item.method.type,
    item.method.params.port,
    String(item.keyword).toLowerCase(),
  ].join('|');
  if (seen.has(dedupeKey)) continue;
  seen.add(dedupeKey);

  item.route_guard = {
    applied,
    original_keyword: originalKeyword,
    keyword_changed: originalKeyword !== item.keyword,
    port_forced: item.method.type === 'kg_search',
  };

  output.push({ json: item, pairedItem: { item: index } });
}

return output;`;

for (const nodeName of ['query转写', 'query转写（失败兜底）']) {
  const node = workflow.nodes.find((item) => item.name === nodeName);
  if (!node) throw new Error(`Node not found: ${nodeName}`);
  node.parameters.messages.messageValues[0].message = systemPrompt;
}

for (const parserName of ['Structured Output Parser', 'Structured Output Parser3']) {
  const node = workflow.nodes.find((item) => item.name === parserName);
  if (!node) throw new Error(`Node not found: ${parserName}`);
  node.parameters.jsonSchemaExample = schemaExample;
}

const guardName = 'KG路由校验与关键词清洗';
let guardNode = workflow.nodes.find((item) => item.name === guardName);
if (!guardNode) {
  guardNode = {
    parameters: { jsCode: routeGuardCode },
    type: 'n8n-nodes-base.code',
    typeVersion: 2,
    position: [704, -176],
    id: '59ca0b2b-8441-4a8c-8348-27e26d215ca7',
    name: guardName,
  };
  workflow.nodes.push(guardNode);
} else {
  guardNode.parameters.jsCode = routeGuardCode;
}

workflow.connections['Split Out'] = {
  main: [[{ node: guardName, type: 'main', index: 0 }]],
};
workflow.connections[guardName] = {
  main: [[{ node: 'Switch', type: 'main', index: 0 }]],
};

fs.writeFileSync(outputPath, `${JSON.stringify(workflowList)}\n`, 'utf8');

console.log(JSON.stringify({
  workflowId: workflow.id,
  workflowName: workflow.name,
  nodes: workflow.nodes.length,
  outputPath,
}, null, 2));
