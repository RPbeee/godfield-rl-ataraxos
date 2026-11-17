// Visible DOM geometry/styles only. Never read handlers or application objects.
export async function commandPanes(page) {
  return page.locator('div:visible').evaluateAll(elements => {
    const panes = [];
    for (const e of elements) {
      const r = e.getBoundingClientRect(), s = getComputedStyle(e);
      // Pinned client ut: the command's transparent pointer overlay, created
      // only after the event animation queue drains and removed on submission.
      if (![195, 535].includes(r.x) || r.y !== 93 || r.width !== 310 || r.height !== 300 ||
          s.zIndex !== '1' || s.cursor !== 'pointer' || s.pointerEvents === 'none' ||
          s.visibility !== 'visible' || Number(s.opacity) !== 1) continue;
      // ut has a 10px rounded corner. A hit-test at (+2,+2) is outside
      // its pointer region even though it lies in the bounding rectangle.
      const hit = document.elementFromPoint(r.x+15, r.y+15);
      if (hit && e.contains(hit)) panes.push(r.x+155);
    }
    return panes.sort((a, b) => a-b);
  });
}
