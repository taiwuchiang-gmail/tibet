const fs = require('fs');
const inPath = process.argv[2];
const outPath = process.argv[3];

const raw = fs.readFileSync(inPath, 'utf8');
const arr = JSON.parse(raw);
const chunk = arr[0].text; // JSON-encoded string literal, possibly with trailing junk after it

// Manually scan for the matching closing quote of the JSON string literal that
// starts at index 0, respecting backslash escapes, so trailing junk (e.g. a
// tool-added "Tab Context" footer) doesn't get included or break JSON.parse.
let i = 1; // skip opening quote
let end = -1;
while (i < chunk.length) {
    if (chunk[i] === '\\') { i += 2; continue; }
    if (chunk[i] === '"') { end = i; break; }
    i += 1;
}
if (end === -1) throw new Error('could not find closing quote');

const jsonStringLiteral = chunk.slice(0, end + 1);
const decoded = JSON.parse(jsonStringLiteral);
console.log('decoded length:', decoded.length);

const delimIdx = decoded.indexOf('|||SPLIT|||');
console.log('delimiter at:', delimIdx);
const firstCopy = delimIdx > 0 ? decoded.slice(0, delimIdx) : decoded;

console.log('first char codepoint:', firstCopy.codePointAt(0).toString(16));
console.log('first copy length:', firstCopy.length);
console.log('preview (JSON-escaped):', JSON.stringify(firstCopy.slice(0, 30)));

fs.writeFileSync(outPath, firstCopy, 'utf8');
console.log('wrote', outPath);
