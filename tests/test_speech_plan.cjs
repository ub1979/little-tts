const assert = require('node:assert/strict');
const {planSpeech} = require('../web/speech-plan.js');
const phones = text => ({ids: [0, ...text.split(/\s+/).map((_, i) => i + 1), 2], stress: []});

const paragraph = 'Hello! I hope you are having a lovely day. Of course you are welcome.';
assert.deepEqual(planSpeech(paragraph, phones, 100).map(p => p.inputText), [paragraph]);

const long = 'The garden was quiet, and the air was filled with roses. The road curved through the hills.';
const chunks = planSpeech(long, phones, 10);
assert(chunks.length > 1);
assert(chunks.every(c => c.ids.length <= 10));
assert.equal(chunks.map(c => c.inputText).join(' '), long);

const number = 'The meeting begins at 3:30 pm. It costs $12.99.';
assert.equal(planSpeech(number, phones, 100)[0].inputText, number);
const title = 'We asked Dr. Jones to explain the story. The answer was clear and very helpful.';
assert.equal(planSpeech(title, phones, 100)[0].inputText, title);
assert.throws(() => planSpeech('...', phones, 100), /words/);
assert.throws(() => planSpeech('word', phones, 2), /limit/);
assert.throws(() => planSpeech('extraordinarily', () => ({ids: new Array(25)}), 10), /too long/);
console.log('Speech planning checks passed');
