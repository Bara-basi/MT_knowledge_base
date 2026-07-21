import fs from 'node:fs';

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error('Usage: node fix_agent_ascii_name.mjs <input.json> <output.json>');
}

const exported = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const workflow = Array.isArray(exported) ? exported[0] : exported;
const agents = workflow.nodes.filter(
  (node) => node.type === '@n8n/n8n-nodes-langchain.agent',
);

if (agents.length !== 1) {
  throw new Error(`Expected exactly one Agent node, found ${agents.length}`);
}

const agent = agents[0];
const oldName = agent.name;
const newName = 'final_qa_agent';

if (oldName !== newName && workflow.nodes.some((node) => node.name === newName)) {
  throw new Error(`A node named ${newName} already exists`);
}

agent.name = newName;
agent.parameters.options = {
  ...(agent.parameters.options || {}),
  autoSaveHighlightedData: false,
};

if (oldName !== newName && Object.hasOwn(workflow.connections, oldName)) {
  workflow.connections[newName] = workflow.connections[oldName];
  delete workflow.connections[oldName];
}

for (const connection of Object.values(workflow.connections)) {
  for (const outputGroups of Object.values(connection)) {
    for (const group of outputGroups) {
      if (!Array.isArray(group)) continue;
      for (const target of group) {
        if (target.node === oldName) target.node = newName;
      }
    }
  }
}

const oldExpression = `$('检索问答')`;
const newExpression = `$('final_qa_agent')`;

const rewriteStrings = (value) => {
  if (typeof value === 'string') {
    return value.split(oldExpression).join(newExpression);
  }
  if (Array.isArray(value)) return value.map(rewriteStrings);
  if (value && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) {
      value[key] = rewriteStrings(child);
    }
  }
  return value;
};

for (const node of workflow.nodes) rewriteStrings(node.parameters);

const serialized = JSON.stringify(workflow);
if (serialized.includes(`\"node\":\"${oldName}\"`)) {
  throw new Error(`A connection still targets the old Agent name: ${oldName}`);
}
if (serialized.includes(oldExpression)) {
  throw new Error('An expression still references the old Agent name');
}
if (!/^[A-Za-z0-9_]+$/.test(agent.name)) {
  throw new Error(`Agent name is not metadata-safe: ${agent.name}`);
}

delete workflow.pinData;
fs.writeFileSync(outputPath, JSON.stringify([workflow]), 'utf8');
console.log(JSON.stringify({ oldName, newName, outputPath }));
