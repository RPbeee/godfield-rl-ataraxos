// Inspect only the small visible status overlay; no application state is read.
import { inflateSync } from 'node:zlib';
function readPixels(png) {
  let width, height, channels, offset=8;
  const compressed=[];
  while (offset+12 <= png.length) {
    const size=png.readUInt32BE(offset), kind=png.toString('ascii',offset+4,offset+8);
    const data=png.subarray(offset+8,offset+8+size);
    if (kind==='IHDR') {
      width=data.readUInt32BE(0);height=data.readUInt32BE(4);
      channels=data[9]===2?3:data[9]===6?4:0;
      if (data[8]!==8 || data[12]!==0) return false;
    }
    if (kind==='IDAT') compressed.push(data);
    offset+=size+12;
  }
  if (!channels || !width || !height) return null;
  const rows=[];
  const bytes=inflateSync(Buffer.concat(compressed)), stride=width*channels;
  let prior=Buffer.alloc(stride), pos=0;
  for (let y=0;y<height;y++) {
    const filter=bytes[pos++], row=Buffer.from(bytes.subarray(pos,pos+stride));pos+=stride;
    if (row.length!==stride || filter>4) return false;
    for (let i=0;i<stride;i++) {
      const left=i>=channels?row[i-channels]:0, up=prior[i], ul=i>=channels?prior[i-channels]:0;
      const p=left+up-ul, a=Math.abs(p-left), b=Math.abs(p-up), c=Math.abs(p-ul);
      const paeth=a<=b && a<=c?left:b<=c?up:ul;
      row[i]=(row[i]+[0,left,up,Math.floor((left+up)/2),paeth][filter])&255;
    }
    rows.push(row);
    prior=row;
  }
  return {width,height,channels,rows};
}

export function purpleFog(png) {
  const p=readPixels(png);
  if (!p || p.width!==340 || p.height!==40) return false;
  for (let y=5;y<35;y++) for (let x=10;x<335;x++) {
    const row=p.rows[y], i=x*p.channels;
    // The translucent overlay shows the moving fog underneath it.
    const r=row[i], g=row[i+1], b=row[i+2];
    if (r<118 || r>142 || g<128 || g>146 || b<244 || b>254 || Math.abs(r-g)>20 ||
        (p.channels===4 && row[i+3]!==255)) return false;
  }
  return true;
}

// The visible pink arrow between the top bands, witnessed in roomB9.
// Require both the shaft/head and clear corners; a pink rectangle is insufficient.
export function targetArrow(png) {
  const p=readPixels(png);
  if (!p || p.width!==40 || p.height!==40) return null;
  const pink=(x,y) => {
    const row=p.rows[y], i=x*p.channels;
    return row[i]>210 && row[i+1]>50 && row[i+1]<160 && row[i+2]>90 && row[i+2]<220
      && (p.channels===3 || row[i+3]===255);
  };
  for (const direction of ['right','left']) {
    const test=(x,y) => pink(direction==='right'?x:39-x,y);
    if ([[5,19],[15,19],[25,19],[25,12],[32,19],[25,27],[22,6],[22,32]].every(([x,y])=>test(x,y))
        && [[5,5],[5,34],[35,5],[35,34],[5,10],[5,29],[17,6],[17,32]].every(([x,y])=>!test(x,y)))
      return direction;
  }
  return null;
}

export async function visibleOverlays(page, body, texts) {
  const fogBands = [];
  const groups = [...body.matchAll(/HP\s*\d+\s*MP\s*\d+\s*[¥￥]\s*\d+/g)].length;
  if (groups === 1 || (groups === 2 && body.includes('+10'))) {
    for (const y of [131, 264]) {
      const png = await page.screenshot({clip:{x:860,y,width:340,height:40}, scale:'css'});
      if (purpleFog(png)) fogBands.push({x:860,y,w:340,h:40});
    }
  }
  let arrow = null;
  if (texts.filter(r => r.y >= 55 && r.y <= 95 && r.x >= 195 && r.x < 840).length === 2) {
    const png = await page.screenshot({clip:{x:500,y:48,width:40,height:40}, scale:'css'});
    arrow = targetArrow(png);
  }
  return {fogBands, targetArrow:arrow};
}
