"""Ranked privacy: raw receive data never reaches the disk (memory buffers + stdout pipe).

A fake `playwright-core` drives the real scripts/official_probe.mjs wrapper and the
pinned third_party probe, with a preloaded hook that reports every file-writing fs
call. Synthetic data only; no browser, no network.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from gfrl.official import OfficialStreams, ROOT, capture_bytes
from gfrl.official_bot import Probe, checked_capture
from gfrl.official_ui import UIUnsupported

NODE = shutil.which('node')
pytestmark = pytest.mark.skipif(NODE is None, reason='node is not installed')
PROBE = ROOT / 'scripts/official_probe.mjs'
SECRET_BUFFERED, SECRET_CHUNK = b'SyntheticRawBuffered', b'SyntheticRawChunk'

FAKE_PLAYWRIGHT = r"""
const b64 = text => Buffer.from(text).toString('base64');
const loc = () => ({ evaluateAll: async () => [], evaluate: async () => [], innerText: async () => 'lobby',
  filter: () => loc(), count: async () => 0, first: () => loc(), nth: () => loc(),
  click: async () => {}, fill: async () => {}, waitFor: async () => {} });
const handlers = {};
const cdp = { on: (name, fn) => { handlers[name] = fn; },
  send: async (method) => method === 'Network.streamResourceContent'
    ? { bufferedData: b64(process.env.FAKE_BUFFERED) } : {} };
const page = { setDefaultTimeout() {}, locator: () => loc(), getByText: () => loc(),
  screenshot: async () => Buffer.alloc(0), mouse: { click: async () => {} },
  goto: async () => {
    handlers['Network.requestWillBeSent']({ requestId: 'r1', request: { method: 'GET',
      url: 'https://firestore.googleapis.com/google.firestore.v1.Firestore/Listen/channel?x=1' } });
    handlers['Network.responseReceived']({ requestId: 'r1' });
    handlers['Network.dataReceived']({ requestId: 'r1', data: b64(process.env.FAKE_CHUNK) });
    handlers['Network.loadingFinished']({ requestId: 'r1' });
  } };
const context = { newPage: async () => page, newCDPSession: async () => cdp, close: async () => {} };
const note = (kind, value) => process.stderr.write('FAKE ' + kind + ' ' + JSON.stringify(value) + '\n');
module.exports = { chromium: {
  launch: async () => { note('launch', true); return { newContext: async () => context, close: async () => {} }; },
  launchPersistentContext: async (dir, options) => { note('persistent', options); return context; },
} };
"""

HOOK = r"""
const fs = require('node:fs');
const write = flags => typeof flags === 'string' ? /[wa+]/.test(flags)
  : typeof flags === 'number' && (flags & (fs.constants.O_WRONLY | fs.constants.O_RDWR | fs.constants.O_CREAT | fs.constants.O_APPEND)) !== 0;
const report = name => process.stderr.write('DISKWRITE ' + name + '\n');
const guard = (target, name, test) => {
  const original = target[name];
  target[name] = function (...args) { if (test(args)) report(name); return original.apply(this, args); };
};
for (const target of [fs, fs.promises]) {
  guard(target, 'open', a => write(a[1]));
  for (const name of ['mkdir', 'writeFile', 'appendFile', 'copyFile', 'rename', 'mkdtemp', 'truncate'])
    guard(target, name, () => true);
}
for (const name of ['openSync', 'writeFileSync', 'appendFileSync', 'mkdirSync', 'mkdtempSync', 'createWriteStream', 'copyFileSync', 'renameSync'])
  guard(fs, name, a => name === 'openSync' ? write(a[1]) : true);
"""


@pytest.fixture
def sandbox(tmp_path):
    pw = tmp_path / 'pw'; pw.mkdir()
    (pw / 'package.json').write_text('{"name":"fake-playwright-core","main":"index.js"}')
    (pw / 'index.js').write_text(FAKE_PLAYWRIGHT)
    hook = tmp_path / 'hook.cjs'; hook.write_text(HOOK)
    cwd = tmp_path / 'cwd'; cwd.mkdir()
    tmp = tmp_path / 'tmp'; tmp.mkdir()
    return dict(pw=pw, hook=hook, cwd=cwd, tmp=tmp, root=tmp_path)


def launch(sandbox, extra_args, **env):
    environ = {k: v for k, v in os.environ.items() if not k.startswith(('GFRL_', 'NODE_'))}
    environ.update(TMPDIR=str(sandbox['tmp']), FAKE_BUFFERED=SECRET_BUFFERED.decode(),
                   FAKE_CHUNK=SECRET_CHUNK.decode(), **env)
    process = subprocess.Popen([NODE, '--require', str(sandbox['hook']), str(PROBE), '--playwright',
                                str(sandbox['pw']), *extra_args], cwd=sandbox['cwd'], env=environ,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process


def talk(process, *commands):
    replies = []
    for command in commands:
        process.stdin.write(json.dumps(command) + '\n'); process.stdin.flush()
        replies.append(json.loads(process.stdout.readline()))
    return replies


def finish(process):
    process.stdin.close()
    process.wait(timeout=20)
    return process.stderr.read()


def files_under(*roots):
    return sorted(str(p) for root in roots for p in Path(root).rglob('*'))


def test_memory_probe_returns_stream_bytes_inline_and_writes_nothing(sandbox):
    process = launch(sandbox, [], GFRL_MEMORY_ONLY='1', GFRL_PROFILE_DIR=str(sandbox['root'] / 'profile'))
    ready = json.loads(process.stdout.readline())
    assert ready['ready'] is True
    capture, shot, closing = talk(process, dict(kind='capture'), dict(kind='screenshot'), dict(kind='close'))
    assert json.loads(process.stdout.readline()) == {'closed': True}
    stderr = finish(process)
    assert 'DISKWRITE' not in stderr
    c, = capture['capture']['captures']
    raw = SECRET_BUFFERED + SECRET_CHUNK
    assert base64.b64decode(c['data']) == raw and c['bytes'] == len(raw)
    assert c['sha256'] == hashlib.sha256(raw).hexdigest() and c['capture_error'] is False
    assert capture['capture']['metadata'] == ''
    # The stream key still parses, so the existing boundary logic keeps working.
    assert OfficialStreams.stream_key(c['file']) == ('memory', '000001')
    assert shot == {'error': 'screenshot_disabled'}
    # A capture reply on close also carries bytes inline (never a path).
    assert 'data' in closing['capture']['captures'][0]
    # Nothing was created anywhere the probe could have written.
    assert files_under(sandbox['cwd'], sandbox['tmp']) == []
    assert not (sandbox['cwd'] / 'memory-only').exists() and not (sandbox['root'] / 'profile').exists()
    for text in (stderr, capture['capture']['metadata']):
        assert SECRET_CHUNK.decode() not in text


def test_memory_probe_uses_persistent_profile_with_disk_cache_off_and_no_logging(sandbox):
    process = launch(sandbox, [], GFRL_MEMORY_ONLY='1', GFRL_PROFILE_DIR=str(sandbox['root'] / 'profile'))
    process.stdout.readline()
    talk(process, dict(kind='close'))
    stderr = finish(process)
    notes = [json.loads(l.split(' ', 2)[2]) for l in stderr.splitlines() if l.startswith('FAKE persistent')]
    assert len(notes) == 1 and 'FAKE launch' not in stderr
    args = notes[0]['args']
    assert '--disk-cache-size=1' in args and '--media-cache-size=1' in args
    assert '--disable-logging' in args and '--disable-breakpad' in args
    assert not any(a.startswith(('--enable-logging', '--log-file', '--log-net-log', '--trace')) for a in args)
    assert not {'recordHar', 'recordVideo', 'tracesDir', 'downloadsPath'} & set(notes[0])


def test_control_unpatched_probe_does_write_files_so_the_hook_is_meaningful(sandbox):
    out = sandbox['root'] / 'out'
    process = launch(sandbox, ['--out-dir', str(out)])
    process.stdout.readline()
    talk(process, dict(kind='capture'), dict(kind='close'))
    stderr = finish(process)
    assert 'DISKWRITE' in stderr and any(p.name.endswith('-prefix.bin') for p in out.iterdir())


def test_memory_mode_ignores_an_explicit_out_dir(sandbox):
    out = sandbox['root'] / 'should-not-exist'
    process = launch(sandbox, ['--out-dir', str(out)], GFRL_MEMORY_ONLY='1')
    process.stdout.readline()
    talk(process, dict(kind='capture'), dict(kind='close'))
    assert 'DISKWRITE' not in finish(process) and not out.exists()


def test_patch_contract_failures_refuse_to_start(tmp_path):
    script = tmp_path / 'check.mjs'
    script.write_text(f"""
import {{ patchProbe, DISK_API }} from {json.dumps((ROOT / 'scripts/official_probe_patch.mjs').as_uri())};
import fs from 'node:fs/promises';
const source = await fs.readFile({json.dumps(str(ROOT / 'third_party/goodfield/scripts/official_headless_probe.mjs'))}, 'utf8');
const out = patchProbe(source, {{ profile: '/p', memory: true }});
const result = {{ clean: !DISK_API.test(out), original_has_disk: DISK_API.test(source) }};
for (const cut of ["const privateFile", "record.file = file;", "await cdp.send('Network.enable');", "const directory = path.resolve"]) {{
  try {{ patchProbe(source.replace(cut, 'changed'), {{ memory: true }}); result[cut] = 'accepted'; }}
  catch (error) {{ result[cut] = error.message; }}
}}
console.log(JSON.stringify(result));
""")
    out = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=30)
    result = json.loads(out.stdout)
    assert result['clean'] is True and result['original_has_disk'] is True
    assert all(v.startswith('probe_memory_') for k, v in result.items() if k not in ('clean', 'original_has_disk'))


# ---- Python Probe(memory=True) over the real wrapper ------------------------------

def test_python_probe_memory_end_to_end_with_streams_integrity(sandbox, monkeypatch):
    for key, value in dict(TMPDIR=str(sandbox['tmp']), FAKE_BUFFERED=SECRET_BUFFERED.decode(),
                           FAKE_CHUNK=SECRET_CHUNK.decode(), GFRL_MEMORY_ONLY='1',
                           GFRL_PROFILE_DIR=str(sandbox['root'] / 'profile')).items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(sandbox['cwd'])
    probe = Probe([NODE, str(PROBE), '--playwright', str(sandbox['pw'])], memory=True)
    reply = probe.send(dict(kind='capture'))
    c, = reply['capture']['captures']
    assert isinstance(c['data'], bytes) and checked_capture(c) == SECRET_BUFFERED + SECRET_CHUNK
    assert capture_bytes(c) == c['data']
    with pytest.raises(UIUnsupported) as refused:
        probe.send(dict(kind='screenshot'))
    assert str(refused.value) == 'screenshot_disabled'
    probe.close()
    assert files_under(sandbox['cwd'], sandbox['tmp']) == []


def fake_node(script):
    return [sys.executable, '-c', script]


def test_python_probe_memory_rejects_captures_without_inline_data():
    script = ("import sys,json\nprint(json.dumps({'ready':True}),flush=True)\n"
              "for l in sys.stdin:\n"
              " print(json.dumps({'capture':{'captures':[{'file':'/tmp/x-1-stream-1-prefix.bin','bytes':1}]}}),flush=True)\n")
    probe = Probe(fake_node(script), memory=True)
    with pytest.raises(UIUnsupported) as error:
        probe.send(dict(kind='capture'))
    assert str(error.value) == 'capture_not_inline' and probe.broken


def test_python_probe_memory_rejects_malformed_base64():
    script = ("import sys,json\nprint(json.dumps({'ready':True}),flush=True)\n"
              "for l in sys.stdin:\n"
              " print(json.dumps({'capture':{'captures':[{'file':'a','bytes':1,'data':'!!not base64!!'}]}}),flush=True)\n")
    probe = Probe(fake_node(script), memory=True)
    with pytest.raises(UIUnsupported) as error:
        probe.send(dict(kind='capture'))
    assert str(error.value) == 'capture_not_inline'


def test_capture_bytes_prefers_inline_data_and_never_reads_the_named_path(tmp_path):
    missing = tmp_path / 'absent-1-stream-1-prefix.bin'
    assert capture_bytes(dict(file=str(missing), data=b'abc')) == b'abc'
    with pytest.raises(FileNotFoundError):
        capture_bytes(dict(file=str(missing)))
    with pytest.raises(ValueError):
        capture_bytes(dict(file=str(missing), data='text'))
