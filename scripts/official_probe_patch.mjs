// Source patches applied to the pinned third_party probe (never edited on disk).
//
// profile: GFRL_PROFILE_DIR -> persistent browser profile (ranked run).
// memory:  GFRL_MEMORY_ONLY=1 -> no raw receive data, capture metadata or screenshot
//          ever touches disk. Stream bytes stay in memory buffers and travel to the
//          parent inline (base64) over the stdout pipe; the output directory is not
//          created. The browser itself only gets its own profile directory: HTTP/media
//          caches are disabled, and no logging/crash-dump switch is enabled.
// Every replaced fragment must match exactly once, otherwise the pinned probe changed
// and the run refuses to start (nothing is guessed).

const once = (source, from, to, code) => {
  if (source.split(from).length !== 2) throw new Error(code);
  return source.replace(from, () => to);
};

const LAUNCH = "browser = await chromium.launch({ headless: true });\n  context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, locale: 'ja-JP' });";
const DIRECTORY = `const directory = path.resolve(options['--out-dir']);
await fs.mkdir(directory, { recursive: true, mode: 0o700 });
const directoryInfo = await fs.lstat(directory);
if (!directoryInfo.isDirectory() || directoryInfo.isSymbolicLink()) fail('invalid_private_directory');
await fs.chmod(directory, 0o700);`;
const EMIT = "const emit = value => process.stdout.write(JSON.stringify(value) + '\\n');";
const PRIVATE_FILE = /const privateFile = async \(suffix, bytes\) => \{[\s\S]*?\n\};\n/;
const STREAM_FILE = `      const file = path.join(directory, \`\${run}-stream-\${String(record.sequence).padStart(6, '0')}.part\`);
      record.file = file;
      record.handle = await fs.open(file, constants.O_RDWR | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);`;
const NETWORK_ENABLE = "await cdp.send('Network.enable');";

// Applied to the patched source: nothing here may open, create or write a file.
export const DISK_API = /\bfs\.(open|mkdir|lstat|chmod|writeFile|appendFile|createWriteStream|mkdtemp|rename|copyFile)\b|createWriteStream|\.screenshot\(\s*\{[^}]*\bpath\b|recordHar|recordVideo|tracesDir|downloadsPath/;

const MEMORY_RUNTIME = `
// ---- memory-only (ranked): replaces the on-disk stream files ----
const memoryHandle = () => {
  const chunks = []; let merged = null;
  return {
    writeFile: async bytes => { chunks.push(Buffer.from(bytes)); merged = null; },
    read: async (target, offset, length, position) => {
      merged ??= Buffer.concat(chunks);
      return { bytesRead: merged.copy(target, offset, position, position + length) };
    },
    close: async () => { chunks.length = 0; merged = null; },
  };
};
const memoryCaptures = new Map();
const memoryName = suffix => 'memory-' + String(++fileSequence).padStart(6, '0') + '-' + suffix;
`;

export function patchProbe(source, { profile = '', memory = false } = {}) {
  if (profile) {
    source = once(source, LAUNCH, `context = await chromium.launchPersistentContext(${JSON.stringify(profile)}, { headless: true, viewport: { width: 1440, height: 1000 }, locale: 'ja-JP'${memory ? ", args: ['--disk-cache-size=1', '--media-cache-size=1', '--disable-logging', '--disable-breakpad']" : ''} });
  browser = context;`, 'probe_launch_contract_changed');
  }
  if (!memory) return source;
  // 1. No output directory: the value of --out-dir is never used.
  source = once(source, DIRECTORY, "const directory = '';", 'probe_memory_directory_contract_changed');
  // 2. Screenshots and capture.json are refused/never produced; prefix bytes are
  //    kept in a map until emit() attaches them to the capture reply.
  if (!PRIVATE_FILE.test(source)) throw new Error('probe_memory_file_contract_changed');
  source = source.replace(PRIVATE_FILE, () => `${MEMORY_RUNTIME}
const privateFile = async (suffix, bytes) => {
  if (suffix === 'screenshot.png') fail('screenshot_disabled');
  if (suffix === 'capture.json') return '';
  const name = memoryName(suffix);
  memoryCaptures.set(name, Buffer.from(bytes));
  return name;
};
`);
  // 3. The live stream record writes into a memory buffer, not a .part file.
  source = once(source, STREAM_FILE, "      record.file = 'memory-stream-' + String(record.sequence).padStart(6, '0');\n      record.handle = memoryHandle();", 'probe_memory_stream_contract_changed');
  // 4. emit(): attach prefix bytes inline and release them; one JSON line per reply.
  source = once(source, EMIT, `const emit = value => {
  const reply = value && value.capture ? { ...value, capture: { ...value.capture,
    captures: (value.capture.captures || []).map(c => {
      const data = memoryCaptures.get(c.file);
      memoryCaptures.delete(c.file);
      return { ...c, data: data ? data.toString('base64') : null };
    }) } } : value;
  memoryCaptures.clear();
  process.stdout.write(JSON.stringify(reply) + '\\n');
};`, 'probe_memory_emit_contract_changed');
  // 5. Disable the HTTP cache so no response body is retained by the browser.
  source = once(source, NETWORK_ENABLE, `${NETWORK_ENABLE}
  await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });`, 'probe_memory_network_contract_changed');
  if (DISK_API.test(source)) throw new Error('probe_memory_disk_api_remains');
  return source;
}

// The pinned CLI insists on --out-dir; memory mode supplies a placeholder that is
// never opened (see patch 1).
export function ensureOutDir(argv) {
  if (!argv.includes('--out-dir')) argv.push('--out-dir', 'memory-only');
}
