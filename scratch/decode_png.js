const fs = require('fs');
const path = require('path');

const inPath = process.argv[2];
const outPath = process.argv[3];

const raw = fs.readFileSync(inPath, 'utf8');
const arr = JSON.parse(raw);
const b64 = arr[0].text.trim();
console.log('base64 length:', b64.length);
fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
console.log('wrote', outPath);
