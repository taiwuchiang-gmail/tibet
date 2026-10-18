const fs = require('fs');

const inPath = process.argv[2];
const outPath = process.argv[3];

const raw = fs.readFileSync(inPath, 'utf8');
const arr = JSON.parse(raw);
const full = arr.map(x => x.text).join('');
console.log('total chars across all chunks:', full.length);

const delimIdx = full.indexOf('|||SPLIT|||');
console.log('delimiter found at index:', delimIdx);
const firstCopy = delimIdx > 0 ? full.slice(0, delimIdx) : full.slice(0, Math.round(full.length / 120));
console.log('extracted single-copy length:', firstCopy.length);
fs.writeFileSync(outPath, firstCopy, 'utf8');
console.log('wrote', outPath);
