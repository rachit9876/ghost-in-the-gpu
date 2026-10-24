// Ghost in the GPU — Extension Background Service Worker
// Connects directly to http://127.0.0.1:8770 and passes the user's selected domain_id.

const DEFAULT_SERVER = "http://127.0.0.1:8770";
const IMAGE_TIMEOUT_MS = 25000;
const API_TIMEOUT_MS = 60000;
const MAX_IMAGE_BYTES = 35 * 1024 * 1024;

// Portable fetch-with-timeout. Does not rely on AbortSignal.timeout
// (Chrome 103+), because MV3 extensions install on Chrome 88+.
async function fetchWithTimeout(url, opts = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function getServerBase() {
  try {
    const res = await chrome.storage.local.get(["serverBase"]);
    const val = ((res && res.serverBase) || "").trim().replace(/\/+$/, "");
    if (/^https?:\/\/.+/.test(val)) return val;
  } catch (e) {
    console.warn("[Ghost] getServerBase failed:", e);
  }
  return DEFAULT_SERVER;
}

async function getActiveDomainId() {
  try {
    const res = await chrome.storage.local.get(["activeDomainId"]);
    if (res && res.activeDomainId) {
      const n = parseInt(res.activeDomainId, 10);
      if (Number.isFinite(n) && n > 0) return n;
    }
  } catch (e) {
    console.warn("[Ghost] getActiveDomainId failed:", e);
  }
  return 1;
}

function toDataUrl(buffer, contentType) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const CHUNK = 8192;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return `data:${contentType};base64,${btoa(binary)}`;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (!request || !request.action) return;

  if (request.action === "fetch_image_base64") {
    if (!request.url) {
      sendResponse({ success: false, error: "No URL provided" });
      return false;
    }
    (async () => {
      try {
        let res;
        try {
          res = await fetchWithTimeout(request.url, {
            method: "GET",
            credentials: "include",
            cache: "force-cache",
          }, IMAGE_TIMEOUT_MS);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
        } catch (e1) {
          res = await fetchWithTimeout(request.url, {
            method: "GET",
            credentials: "omit",
          }, IMAGE_TIMEOUT_MS);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
        }

        const buffer = await res.arrayBuffer();
        if (buffer.byteLength > MAX_IMAGE_BYTES) throw new Error("Image too large");
        const contentType = res.headers.get("content-type") || "image/jpeg";
        sendResponse({ success: true, base64: toDataUrl(buffer, contentType) });
      } catch (err) {
        console.warn("[Ghost] fetch_image_base64 failed:", err);
        sendResponse({ success: false, error: String(err) });
      }
    })();
    return true;
  }

  if (request.action === "log_image" || request.action === "test_image") {
    if (!request.payload || typeof request.payload !== "object") {
      sendResponse({ ok: false, error: "Missing payload" });
      return false;
    }
    (async () => {
      try {
        const base = await getServerBase();
        const domainId = await getActiveDomainId();
        const endpoint = request.action === "test_image" ? "/api/ext/test" : "/api/ext/log";
        const payload = Object.assign({}, request.payload, { domain_id: domainId });
        const res = await fetchWithTimeout(`${base}${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }, API_TIMEOUT_MS);
        const data = await res.json().catch(() => ({}));
        sendResponse({ ok: res.ok, status: res.status, data });
      } catch (err) {
        console.warn("[Ghost] API call failed:", err);
        sendResponse({ ok: false, error: String(err) });
      }
    })();
    return true;
  }

  if (request.action === "check_server_status") {
    (async () => {
      try {
        const base = await getServerBase();
        const domainId = await getActiveDomainId();
        const res = await fetchWithTimeout(
          `${base}/api/ext/health?domain_id=${domainId}`,
          { method: "GET" },
          4000
        );
        const data = await res.json().catch(() => ({}));
        sendResponse({ live: res.ok, data, activeDomainId: domainId });
      } catch (err) {
        sendResponse({ live: false, error: String(err) });
      }
    })();
    return true;
  }
});

if (chrome.commands && chrome.commands.onCommand) {
  chrome.commands.onCommand.addListener(async (command) => {
    if (command !== "log_hovered_image") return;
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && tab.id) {
        chrome.tabs.sendMessage(tab.id, { action: "trigger_log_hovered" }).catch(() => {});
      }
    } catch (e) {
      console.warn("[Ghost] command forwarding failed:", e);
    }
  });
}