/**
 * Ghost in the GPU Web Extension - Content Script
 * Ultra-lightweight image logger with instant PE-Core scoring.
 */
(() => {
  const origin = window.location.origin;
  const GHOST_HOSTS = [
    "127.0.0.1:8770", "localhost:8770", "[::1]:8770",
    "127.0.0.1:8771", "localhost:8771",
    "127.0.0.1:8765", "localhost:8765",
  ];
  if (GHOST_HOSTS.some(h => origin.includes(h))) return;

  let hoverEl = null;
  let hoverTimer = null;
  let hideTimer = null;
  let shownToolbar = null;
  const toolbars = new WeakMap();

  // Session-level dedupe: same Instagram media rendered in multiple DOM nodes
  // (e.g. _aagu carousel items) shares the same ig_cache_key. We only send
  // one log per key per page lifetime; the server also has its own dedupe.
  const loggedIds = new Set();

  const layer = document.createElement("div");
  layer.className = "ghost-gpu-layer";
  layer.setAttribute("aria-hidden", "true");
  document.documentElement.appendChild(layer);

  // Isolate all extension UI inside an airtight Shadow DOM
  const shadowRoot = layer.attachShadow({ mode: "open" });
  const styleEl = document.createElement("link");
  styleEl.rel = "stylesheet";
  styleEl.href = chrome.runtime.getURL("styles.css");
  shadowRoot.appendChild(styleEl);

  // Import / Download Tray icon for Log action
  const ICON_IMPORT = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`;
  const ICON_TEST = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>`;
  const SVG_STAR = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>`;
  const SVG_CHECK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
  const SVG_CROSS = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
  const SVG_INFO = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;

  // Wrap chrome.runtime.sendMessage in a promise with a hard timeout
  // so the UI never hangs if the service worker dies.
  function sendMessageWithTimeout(msg, timeoutMs = 70000) {
    return new Promise((resolve) => {
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        resolve({ ok: false, error: "timeout" });
      }, timeoutMs);
      try {
        chrome.runtime.sendMessage(msg, (res) => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          if (chrome.runtime.lastError) {
            resolve({ ok: false, error: chrome.runtime.lastError.message });
          } else {
            resolve(res || { ok: false, error: "empty response" });
          }
        });
      } catch (e) {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve({ ok: false, error: String(e) });
        }
      }
    });
  }

  function createToolbar(el) {
    const tb = document.createElement("div");
    tb.className = "ghost-gpu-overlay";
    tb.innerHTML = `
      <div class="ghost-gpu-actions">
        <button class="ghost-gpu-btn ghost-gpu-test" title="Test score only (does NOT save to disk or DB)">
          ${ICON_TEST}<span>Test</span>
        </button>
        <button class="ghost-gpu-btn ghost-gpu-log" title="Log and save image to Ghost in the GPU library">
          ${ICON_IMPORT}<span>Log</span>
        </button>
      </div>
    `;
    tb._target = el;

    tb.querySelector(".ghost-gpu-test").addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      sendTest(tb);
    });

    tb.querySelector(".ghost-gpu-log").addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      sendLog(tb);
    });

    tb.addEventListener("mouseenter", () => {
      if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
    });
    tb.addEventListener("mouseleave", () => scheduleHide());

    shadowRoot.appendChild(tb);
    return tb;
  }

  function toolbarFor(el) {
    let tb = toolbars.get(el);
    if (!tb || !tb.isConnected) {
      tb = createToolbar(el);
      toolbars.set(el, tb);
    }
    return tb;
  }

  function showToolbar(el) {
    const tb = toolbarFor(el);
    const rect = el.getBoundingClientRect();
    if (rect.width < 90 || rect.height < 90) return;

    tb.style.visibility = "hidden";
    tb.classList.add("visible");

    // Position vertically at 3/4 (75%) of the image height
    let top = rect.top + window.scrollY + (rect.height * 0.75) - (tb.offsetHeight / 2);
    let left = rect.left + window.scrollX + (rect.width / 2) - (tb.offsetWidth / 2);

    const viewTop = window.scrollY + 10;
    if (top < viewTop) top = viewTop;
    const maxTop = window.scrollY + window.innerHeight - tb.offsetHeight - 10;
    if (top > maxTop) top = Math.max(viewTop, maxTop);
    left = Math.max(window.scrollX + 8, Math.min(window.scrollX + window.innerWidth - tb.offsetWidth - 8, left));

    tb.style.top = `${Math.round(top)}px`;
    tb.style.left = `${Math.round(left)}px`;
    tb.style.visibility = "";

    if (shownToolbar && shownToolbar !== tb) shownToolbar.classList.remove("visible");
    shownToolbar = tb;
  }

  function hideToolbar() {
    if (shownToolbar) {
      shownToolbar.classList.remove("visible");
      shownToolbar = null;
    }
    if (hoverTimer) { clearTimeout(hoverTimer); hoverTimer = null; }
    hoverEl = null;
  }

  function scheduleHide() {
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      hideTimer = null;
      hideToolbar();
    }, 300);
  }

  let activeToast = null;
  let activeToastTimer = null;

  function showToast(htmlContent, type = "normal", duration = 3800) {
    if (activeToast) {
      clearTimeout(activeToastTimer);
      activeToast.remove();
      activeToast = null;
    }

    const toast = document.createElement("div");
    toast.className = `ghost-gpu-toast ${type}`;
    toast.innerHTML = htmlContent;
    shadowRoot.appendChild(toast);
    activeToast = toast;

    activeToastTimer = setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      setTimeout(() => {
        if (activeToast === toast) activeToast = null;
        toast.remove();
      }, 300);
    }, duration);
  }

  // Unified notification: Routes everything cleanly to the bottom-right toast
  function showStatus(tb, htmlContent, type = "normal", duration = 3800) {
    showToast(htmlContent, type, duration);
  }

  // ------------ Scraper Helpers ------------

  // Parse a srcset and return the URL of the largest candidate. Used to avoid
  // sending a 320–640px preview to the server when a 1080px asset is available.
  function pickLargestFromSrcset(img) {
    const srcset = img.getAttribute && img.getAttribute("srcset");
    if (!srcset) return null;
    let bestUrl = null;
    let bestScore = -1;
    for (const entry of srcset.split(",")) {
      const parts = entry.trim().split(/\s+/);
      if (!parts[0]) continue;
      const url = parts[0];
      const d = parts[1] || "";
      let score = 1;
      if (d.endsWith("w")) {
        score = parseFloat(d) || 0;
      } else if (d.endsWith("x")) {
        // Density descriptors: approximate by rendered width
        score = (parseFloat(d) || 1) * (img.naturalWidth || img.width || 0) * 10;
      }
      if (score > bestScore) {
        bestScore = score;
        bestUrl = url;
      }
    }
    return bestUrl;
  }

  // Return the largest width advertised by a srcset (for deciding whether to
  // bother fetching a higher-res variant than what the canvas would produce).
  function parseSrcsetMaxWidth(img) {
    const srcset = img.getAttribute && img.getAttribute("srcset");
    if (!srcset) return 0;
    let best = 0;
    for (const entry of srcset.split(",")) {
      const parts = entry.trim().split(/\s+/);
      const d = parts[1] || "";
      if (d.endsWith("w")) {
        const w = parseFloat(d) || 0;
        if (w > best) best = w;
      } else if (d.endsWith("x")) {
        const rw = img.naturalWidth || img.width || 0;
        const w = (parseFloat(d) || 1) * rw;
        if (w > best) best = w;
      }
    }
    return best;
  }

  function getBase64FromVideoFrame(video) {
    try {
      if (!video) return null;
      let w = video.videoWidth || video.clientWidth;
      let h = video.videoHeight || video.clientHeight;
      if (!w || !h) return null;

      const maxDim = 1024;
      if (w > maxDim || h > maxDim) {
        if (w >= h) {
          h = Math.round((h * maxDim) / w);
          w = maxDim;
        } else {
          w = Math.round((w * maxDim) / h);
          h = maxDim;
        }
      }

      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(video, 0, 0, w, h);
      return canvas.toDataURL("image/jpeg", 0.90);
    } catch (e) {
      console.warn("[Ghost] Video frame capture error:", e);
      return null;
    }
  }

  function getBase64FromImage(img) {
    try {
      if (!img.complete) return null;
      let w = img.naturalWidth || img.width;
      let h = img.naturalHeight || img.height;
      if (!w || !h) return null;

      const maxDim = 1024;
      if (w > maxDim || h > maxDim) {
        if (w >= h) {
          h = Math.round((h * maxDim) / w);
          w = maxDim;
        } else {
          w = Math.round((w * maxDim) / h);
          h = maxDim;
        }
      }

      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      return canvas.toDataURL("image/jpeg", 0.90);
    } catch (e) {
      return null;
    }
  }

  function resolveImageSource(el) {
    if (!el) return null;
    if (el.tagName === "IMG") {
      return pickLargestFromSrcset(el)
          || el.currentSrc
          || el.src
          || el.getAttribute("data-src")
          || el.getAttribute("data-original");
    }
    if (el.tagName === "VIDEO" || el.hasAttribute("poster")) {
      const poster = el.getAttribute("poster") || el.poster;
      if (poster) return poster;
    }
    const bg = window.getComputedStyle(el).backgroundImage;
    if (bg && bg.includes("url(")) {
      const m = bg.match(/url\((['"]?)([^'")]+)\1\)/);
      if (m) return m[2];
    }
    const innerImg = el.querySelector && el.querySelector("img");
    if (innerImg) {
      return pickLargestFromSrcset(innerImg)
          || innerImg.currentSrc
          || innerImg.src;
    }
    const innerVideo = el.querySelector && el.querySelector("video[poster]");
    if (innerVideo) return innerVideo.getAttribute("poster") || innerVideo.poster;
    return null;
  }

  async function fetchViaBackground(url) {
    if (!url || typeof chrome === "undefined" || !chrome.runtime || !chrome.runtime.sendMessage) return null;
    const res = await sendMessageWithTimeout({ action: "fetch_image_base64", url }, 30000);
    if (res && res.success && res.base64) return res.base64;
    return null;
  }

  async function getImageBase64(tb) {
    const el = tb._target;
    if (!el) return null;

    // Resolve the underlying <img>/<video> even when the hover target is a
    // container (e.g. Instagram's _aagu carousel wrapper).
    const imgEl = el.tagName === "IMG" ? el
                : (el.querySelector && el.querySelector("img"));
    const videoEl = el.tagName === "VIDEO" ? el
                  : (el.querySelector && el.querySelector("video"));

    // 1. Image: always try to fetch the original URL first via background
    //    fetch. Background fetch bypasses CORS and returns the original
    //    bytes at full resolution. Previously the code only did this when
    //    srcset advertised >2x renderedW, so a 4K wallpaper (single-URL,
    //    no srcset) was silently downscaled to 1024px / JPEG-0.90 by the
    //    canvas fallback. That is fixed here.
    if (imgEl) {
      const bestUrl = pickLargestFromSrcset(imgEl)
                    || imgEl.currentSrc
                    || imgEl.src;
      if (bestUrl) {
        const hi = await fetchViaBackground(bestUrl);
        if (hi) return hi;
        // Background fetch failed (403, timeout, offline) — fall through
        // to canvas, which will downscale if the image is huge.
      }
      const direct = getBase64FromImage(imgEl);
      if (direct) return direct;
    }

    // 2. Video: capture paused/current frame; fallback to poster.
    if (videoEl) {
      const frame = getBase64FromVideoFrame(videoEl);
      if (frame) return frame;
      const poster = videoEl.getAttribute("poster") || videoEl.poster;
      if (poster) return fetchViaBackground(poster);
    }

    // 3. Final fallback: background fetch of the resolved source URL.
    return fetchViaBackground(resolveImageSource(el));
  }

  // Extract platform metadata (Instagram-agnostic: only fills what it finds).
  function decodeEfg(efg) {
    try {
      let b64 = String(efg).replace(/-/g, "+").replace(/_/g, "/");
      while (b64.length % 4) b64 += "=";
      return JSON.parse(atob(b64));
    } catch (_) {
      return null;
    }
  }

  function extractMeta(imgUrl, el) {
    const meta = {};
    try {
      if (imgUrl) {
        const u = new URL(imgUrl);
        const ck = u.searchParams.get("ig_cache_key");
        if (ck) {
          // Drop the ".3-ccb7-5" style version suffix for a stable id.
          meta.ig_cache_key = ck.split(".")[0];
        }
        const efg = u.searchParams.get("efg");
        if (efg) {
          const json = decodeEfg(efg);
          if (json && json.vencode_tag) meta.vencode_tag = json.vencode_tag;
        }
      }
    } catch (_) {}

    const imgEl = el && (el.tagName === "IMG" ? el : (el.querySelector && el.querySelector("img")));
    const alt = (imgEl && imgEl.alt) || (el && el.alt);
    if (alt) meta.alt_text = alt;

    return meta;
  }

  function formatScoreSuffix(score) {
    if (score === null || score === undefined) return "";
    const prefix = score > 0 ? "+" : "";
    return ` (${prefix}${Number(score).toFixed(2)})`;
  }

  // ------------ Send Test Action (No saving) ------------
  async function sendTest(tb) {
    showStatus(tb, "Testing...", "normal", 5000);
    const base64 = await getImageBase64(tb);
    const imgUrl = resolveImageSource(tb._target);

    if (!base64 && !imgUrl) {
      showStatus(tb, "No image source", "error", 2500);
      return;
    }

    const payload = {
      image_url: imgUrl,
      image_base64: base64,
      page_url: window.location.href,
      ...extractMeta(imgUrl, tb._target),
    };

    const res = await sendMessageWithTimeout({ action: "test_image", payload });
    if (!res.ok) {
      showStatus(tb, "Server offline (Run START.bat)", "error", 3500);
      return;
    }

    const d = res.data || {};
    const score = d.score;
    const badge = d.badge || (score >= -0.05 ? "Approved" : "Below Threshold");
    const tier = d.tier || (score >= -0.05 ? "approved" : "reject");

    let icon = SVG_CHECK;
    if (tier === "elite" || tier === "favorite") icon = SVG_STAR;
    else if (tier === "reject") icon = SVG_CROSS;

    showStatus(tb, `${icon}<span>${badge}${formatScoreSuffix(score)}</span>`, tier, 4500);
  }

  // ------------ Send Log Action ------------
  async function sendLog(tb) {
    const imgUrl = resolveImageSource(tb._target);
    const meta = extractMeta(imgUrl, tb._target);

    // Session dedupe: same ig_cache_key in a different DOM node → skip.
    if (meta.ig_cache_key && loggedIds.has(meta.ig_cache_key)) {
      showStatus(tb, `${SVG_INFO}<span>Already logged this session</span>`, "exists", 2500);
      return;
    }

    showStatus(tb, "Logging...", "normal", 5000);
    const base64 = await getImageBase64(tb);

    if (!base64 && !imgUrl) {
      showStatus(tb, "No image source", "error", 2500);
      return;
    }

    const payload = {
      image_url: imgUrl,
      image_base64: base64,
      page_url: window.location.href,
      ...meta,
    };

    const res = await sendMessageWithTimeout({ action: "log_image", payload }, 90000);
    if (!res.ok) {
      showStatus(tb, "Server offline (Run START.bat)", "error", 3500);
      return;
    }

    const d = res.data || {};
    if (d.status === "exists") {
      if (meta.ig_cache_key) loggedIds.add(meta.ig_cache_key);
      const badge = d.badge ? ` - ${d.badge}` : "";
      showStatus(tb, `${SVG_INFO}<span>In Library${badge}${formatScoreSuffix(d.score)}</span>`, "exists", 3500);
      return;
    }

    const score = d.score;
    const badge = d.badge || (score >= -0.05 ? "Approved" : "Below Threshold");
    const tier = d.tier || (score >= -0.05 ? "approved" : "reject");

    let icon = SVG_CHECK;
    if (tier === "elite" || tier === "favorite") icon = SVG_STAR;
    else if (tier === "reject") icon = SVG_CROSS;

    if (meta.ig_cache_key) loggedIds.add(meta.ig_cache_key);
    showStatus(tb, `${icon}<span>Saved - ${badge}${formatScoreSuffix(score)}</span>`, tier, 4000);
  }

  // ------------ Hover & Shortcut Listeners ------------
  function triggerLogOnCurrentHover() {
    const target = hoverEl || (shownToolbar && shownToolbar._target);
    if (!target || !target.isConnected) return;
    showToolbar(target);
    const tb = toolbarFor(target);
    sendLog(tb);
  }

  if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.onMessage) {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg && msg.action === "trigger_log_hovered") {
        triggerLogOnCurrentHover();
      }
    });
  }

  document.addEventListener("keydown", (e) => {
    if (!e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (e.key !== "l" && e.key !== "L" && e.code !== "KeyL") return;
    const target = hoverEl || (shownToolbar && shownToolbar._target);
    if (target && target.isConnected) {
      e.preventDefault();
      triggerLogOnCurrentHover();
    }
  });

  const HOVER_SEL = "img, video[poster], video, [style*='background-image'], shreddit-player";

  document.addEventListener("mouseover", (e) => {
    let target = e.target.closest && e.target.closest(HOVER_SEL);

    // Some sites (Instagram's _aagu carousel items, Pinterest, etc.) place a
    // transparent overlay <div> on top of the actual <img>. In that case
    // e.target is the overlay and closest() finds nothing, so the toolbar
    // never appears. Fall back to elementsFromPoint to find the image beneath.
    if (!target && e.clientX !== undefined && e.clientY !== undefined) {
      const stack = document.elementsFromPoint(e.clientX, e.clientY);
      for (const n of stack) {
        if (n && n.matches && n.matches(HOVER_SEL)) {
          target = n;
          break;
        }
      }
    }

    if (!target) return;
    if (target.closest(".ghost-gpu-layer")) return;

    if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
    if (hoverEl === target) return;

    hoverEl = target;
    if (hoverTimer) clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => {
      if (hoverEl === target && target.isConnected) {
        showToolbar(target);
      }
    }, 400);
  }, { passive: true });

  document.addEventListener("mouseout", (e) => {
    const related = e.relatedTarget;
    if (related && related.nodeType === 1) {
      if (layer.contains(related)) return;
      if (related.closest && (related.closest(".ghost-gpu-overlay") || related.closest(".ghost-gpu-layer"))) return;
    }
    scheduleHide();
  }, { passive: true });

})();