// Extend the pinned probe's visible state without changing goodfield.
import fs from 'node:fs/promises';
const probe = new URL('../third_party/goodfield/scripts/official_headless_probe.mjs', import.meta.url);
const pixels = new URL('./official_probe_pixels.mjs', import.meta.url).href;
const source = await fs.readFile(probe, 'utf8');
const marker = "return { body: await page.locator('body').innerText(), inputs, images, texts };";
if (source.split(marker).length !== 2) throw new Error('probe_state_contract_changed');
const replacement = `
  const body = await page.locator('body').innerText();
  const { visibleOverlays } = await import(${JSON.stringify(pixels)});
  return { body, inputs, images, texts, ...await visibleOverlays(page, body, texts) };
`;
await import('data:text/javascript;base64,' + Buffer.from(source.replace(marker, replacement)).toString('base64'));
