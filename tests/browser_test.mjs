// Drive headless Chrome over CDP: open the page, start the (fake) camera, switch filters, take screenshots.
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// screenshots + fake camera live in tests/out (run tests/make_fake_cam.py first, and serve the project on :8765)
const TMP = fileURLToPath(new URL("./out", import.meta.url)).split("\\").join("/");
mkdirSync(TMP, { recursive: true });
const chrome = spawn("C:/Program Files/Google/Chrome/Application/chrome.exe", [
  "--headless=new", "--remote-debugging-port=9333", `--user-data-dir=${TMP}/chrome-prof`,
  "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
  `--use-file-for-fake-video-capture=${TMP}/cam.y4m`, "--window-size=1280,760",
  "--autoplay-policy=no-user-gesture-required", "about:blank",
], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 40; i++) {
  try { targets = await (await fetch("http://127.0.0.1:9333/json")).json(); break; } catch { await sleep(250); }
}
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map(); const logs = [];
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  if (msg.method === "Runtime.consoleAPICalled") logs.push(`[${msg.params.type}] ` + msg.params.args.map((a) => a.value ?? a.description).join(" "));
  if (msg.method === "Runtime.exceptionThrown") logs.push("[EXCEPTION] " + JSON.stringify(msg.params.exceptionDetails).slice(0, 600));
};
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
const shot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(`${TMP}/${name}.png`, Buffer.from(r.result.data, "base64"));
};

await send("Runtime.enable"); await send("Page.enable");
// optional arg: page URL, e.g. the live site https://sesili-a.github.io/hand-portal/web/
await send("Page.navigate", { url: process.argv[2] || "http://127.0.0.1:8765/web/" });
for (let i = 0; i < 120; i++) { if (await ev("!document.getElementById('start').disabled")) break; await sleep(500); }
console.log("status:", await ev("document.getElementById('status').textContent"));
await shot("01_intro");
await ev("document.getElementById('start').click()");
await sleep(6000);
console.log("intro hidden:", await ev("document.getElementById('intro').hidden"));
await shot("02_running");
const key = (k) => ev(`dispatchEvent(new KeyboardEvent('keydown',{key:'${k}'}))`);
await key("5"); await sleep(7000); await shot("03_shades");
await key("4"); await sleep(3000); await shot("04_mesh");
await key("1"); await sleep(3000); await shot("05_neon");
await key("p"); await sleep(1500); await shot("06_snap");
console.log("--- console ---\n" + logs.join("\n"));
chrome.kill();
process.exit(0);
