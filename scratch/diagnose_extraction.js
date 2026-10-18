const fs = require('fs');
const inPath = process.argv[2];
const raw = fs.readFileSync(inPath, 'utf8');
const arr = JSON.parse(raw);
console.log('arr length:', arr.length);
console.log('arr[0].text first 5 chars (JSON):', JSON.stringify(arr[0].text.slice(0, 5)));
console.log('arr[0].text last 5 chars (JSON):', JSON.stringify(arr[0].text.slice(-5)));

try {
    const doubleParsed = JSON.parse(arr[0].text);
    console.log('DOUBLE-PARSE SUCCEEDED. First 20 chars:', JSON.stringify(doubleParsed.slice(0, 20)));
    console.log('length after double-parse:', doubleParsed.length);
} catch (e) {
    console.log('double-parse failed:', e.message);
}
