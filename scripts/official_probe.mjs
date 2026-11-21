// Extend the pinned probe's visible state without changing goodfield.
import fs from 'node:fs/promises';
import { patchProbe, ensureOutDir } from './official_probe_patch.mjs';
const probe = new URL('../third_party/goodfield/scripts/official_headless_probe.mjs', import.meta.url);
const pixels = new URL('./official_probe_pixels.mjs', import.meta.url).href;
const controls = new URL('./official_probe_controls.mjs', import.meta.url).href;
let source = await fs.readFile(probe, 'utf8');
// Ranked run only (environment, never stdin or logs): GFRL_PROFILE_DIR selects a
// dedicated persistent browser profile; GFRL_MEMORY_ONLY=1 keeps raw receive bytes,
// capture metadata and screenshots off the disk (memory buffers + the stdout pipe).
// Without them the pinned ephemeral, out-dir based probe is used unchanged.
const memory = process.env.GFRL_MEMORY_ONLY === '1';
if (memory) ensureOutDir(process.argv);
source = patchProbe(source, { profile: process.env.GFRL_PROFILE_DIR || '', memory });
const marker = "return { body: await page.locator('body').innerText(), inputs, images, texts };";
if (source.split(marker).length !== 2) throw new Error('probe_state_contract_changed');
const replacement = `
  const body = await page.locator('body').innerText();
  const { visibleOverlays } = await import(${JSON.stringify(pixels)});
  const { commandPanes } = await import(${JSON.stringify(controls)});
  return { body, inputs, images, texts, ...await visibleOverlays(page, body, texts),
           commandPanes: await commandPanes(page) };
`;
await import('data:text/javascript;base64,' + Buffer.from(source.replace(marker, replacement)).toString('base64'));
