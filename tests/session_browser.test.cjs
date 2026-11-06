const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../ui/session.js'), 'utf8');

function storage() {
  const data = new Map();
  return {getItem: k => data.get(k) || null, setItem: (k,v) => data.set(k,v)};
}
function locks() {
  let queue = Promise.resolve();
  return {request: (_, fn) => { const next = queue.then(fn); queue = next.catch(() => {}); return next; }};
}
let issued = 0;
function page(localStorage = storage(), sharedLocks = locks()) {
  const window = {localStorage, navigator: {locks: sharedLocks}, calls: 0};
  window.fetch = async (url, options) => {
    assert.equal(url, 'https://api.example/sessions');
    assert.equal(options.method, 'POST');
    window.calls++;
    const token = String(++issued).padStart(43,'a');
    return {ok: true, json: async () => ({token})};
  };
  vm.runInNewContext(source, {window, Error});
  return {window, get: window.createHydroSession('https://api.example')};
}

test('parallel startup requests use one persisted session', async () => {
  const a = page();
  const tokens = await Promise.all(Array.from({length: 10}, () => a.get()));
  assert.equal(new Set(tokens).size, 1);
  assert.equal(a.window.calls, 1);
  assert.equal(a.window.localStorage.getItem('hb-session-v1'), tokens[0]);
});
test('two tabs in the same browser share the first session', async () => {
  const store = storage(), lock = locks();
  const a = page(store, lock), b = page(store, lock);
  const [ta, tb] = await Promise.all([a.get(), b.get()]);
  assert.equal(ta, tb);
  assert.equal(a.window.calls + b.window.calls, 1);
});
test('reload restores the session without issuing a replacement', async () => {
  const store = storage();
  const original = await page(store).get();
  const restored = page(store);
  assert.equal(await restored.get(), original);
  assert.equal(restored.window.calls, 0);
});
test('separate browsers receive separate credentials', async () => {
  const a = page(), b = page();
  assert.notEqual(await a.get(), await b.get());
});
test('blocked storage fails explicitly, without a temporary inaccessible session', async () => {
  const a = page({getItem() {throw Error('blocked');}});
  await assert.rejects(a.get(), /Дозвольте збереження/);
  assert.equal(a.window.calls, 0);
});
test('storage write failure does not pretend the session was saved', async () => {
  const a = page({getItem: () => null, setItem() {throw Error('quota');}});
  await assert.rejects(a.get(), /Не вдалося зберегти доступ/);
});
test('network failure permits retry without creating parallel sessions', async () => {
  const a = page();
  const fetch = a.window.fetch;
  a.window.fetch = async () => {throw Error('offline');};
  await assert.rejects(a.get(), /Не вдалося відкрити сесію/);
  a.window.fetch = fetch;
  assert.equal((await a.get()).length, 43);
  assert.equal(a.window.calls, 1);
});
