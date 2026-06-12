const cameraStatus = document.querySelector("#camera-status");
const askStatus = document.querySelector("#ask-status");
const askForm = document.querySelector("#ask-form");
const askButton = document.querySelector("#ask-button");
const questionInput = document.querySelector("#question");
const conversationEl = document.querySelector("#conversation");
const conversationEmpty = document.querySelector("#conversation-empty");

const statInferMs = document.querySelector("#stat-infer-ms");
const statFps = document.querySelector("#stat-fps");
const cameraStream = document.querySelector("#camera-stream");
const cameraOverlay = document.querySelector("#camera-overlay");
const streamFrame = cameraStream ? cameraStream.closest(".stream-frame") : null;

const historyList = document.querySelector("#history-list");
const historyEmpty = document.querySelector("#history-empty");
const historyMenu = document.querySelector("#history-menu");
const newChatBtn = document.querySelector("#new-chat-btn");
const sidebarToggleBtn = document.querySelector("#sidebar-toggle");
const pageEl = document.querySelector(".page");
const cvModelButton = document.querySelector("#cv-model-button");
const cvModelLabel = document.querySelector("#cv-model-label");
const cvModelMenu = document.querySelector("#cv-model-menu");

const SIDEBAR_KEY = "ssdet.sidebar.collapsed";
const CV_MODEL_KEY = "ssdet.cv_model.v3";
const CV_MODELS = {
  none: "默认",
  ssg_a_robot_toy: "robot_toy",
};

function applySidebarState(collapsed) {
  pageEl.classList.toggle("sidebar-collapsed", collapsed);
  sidebarToggleBtn.dataset.tooltip = collapsed ? "展开菜单" : "收起菜单";
  sidebarToggleBtn.setAttribute("aria-label", collapsed ? "展开菜单" : "收起菜单");
}

function setSidebarCollapsed(collapsed) {
  applySidebarState(collapsed);
  try {
    localStorage.setItem(SIDEBAR_KEY, collapsed ? "1" : "0");
  } catch (_) {
    /* noop */
  }
}

function loadCvModel() {
  try {
    const stored = localStorage.getItem(CV_MODEL_KEY);
    if (stored && Object.prototype.hasOwnProperty.call(CV_MODELS, stored)) {
      return stored;
    }
  } catch (_) {
    /* noop */
  }
  return "none";
}

function saveCvModel(model) {
  try {
    localStorage.setItem(CV_MODEL_KEY, model);
  } catch (_) {
    /* noop */
  }
}

function renderCvModel() {
  cvModelLabel.textContent = CV_MODELS[activeCvModel] || activeCvModel;
  for (const button of cvModelMenu.querySelectorAll("button[data-cv-model]")) {
    button.classList.toggle("active", button.dataset.cvModel === activeCvModel);
  }
}

function openCvModelMenu() {
  closeMenu();
  const rect = cvModelButton.getBoundingClientRect();
  cvModelMenu.hidden = false;
  cvModelButton.setAttribute("aria-expanded", "true");
  cvModelMenu.style.top = `${rect.top}px`;
  cvModelMenu.style.left = `${Math.min(rect.right + 12, window.innerWidth - 236)}px`;
}

function closeCvModelMenu() {
  cvModelMenu.hidden = true;
  cvModelButton.setAttribute("aria-expanded", "false");
}

async function syncCvModel(model) {
  await fetch("/cv_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
}

applySidebarState(localStorage.getItem(SIDEBAR_KEY) === "1");
sidebarToggleBtn.addEventListener("click", () => {
  setSidebarCollapsed(!pageEl.classList.contains("sidebar-collapsed"));
});

cvModelButton.addEventListener("click", (event) => {
  event.stopPropagation();
  if (cvModelMenu.hidden) {
    openCvModelMenu();
  } else {
    closeCvModelMenu();
  }
});

cvModelMenu.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-cv-model]");
  if (!button) return;
  const nextModel = button.dataset.cvModel;
  activeCvModel = nextModel;
  saveCvModel(nextModel);
  renderCvModel();
  closeCvModelMenu();
  try {
    await syncCvModel(nextModel);
    refreshStats();
  } catch (error) {
    setCameraStatus("切换失败", "error");
  }
});

const STORAGE_KEY = "ssdet.conversations.v2";
const MAX_ROUNDS_PER_CONV = 10;
const MAX_ASSISTANT_CHARS = 600;

let conversations = loadConversations();
let activeId = null;
let menuTargetId = null;
let activeRequest = null;
let activeCvModel = loadCvModel();

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((c) => c && typeof c.id === "string" && Array.isArray(c.messages))
      .map((c) => {
        applyConversationLimits(c);
        return c;
      });
  } catch (error) {
    return [];
  }
}

function saveConversations() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch (error) {
    /* localStorage 不可用时静默忽略 */
  }
}

function makeId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function truncate(text, max) {
  const trimmed = text.replace(/\s+/g, " ").trim();
  if (trimmed.length <= max) return trimmed;
  return trimmed.slice(0, max) + "…";
}

function trimToMaxRounds(messages, maxRounds) {
  if (!messages || !messages.length || maxRounds <= 0) return messages;
  const userIdx = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === "user") userIdx.push(i);
  }
  if (userIdx.length <= maxRounds) return messages;
  const start = userIdx[userIdx.length - maxRounds];
  return messages.slice(start);
}

function clampAssistantContent(text) {
  const s = String(text);
  /* 仅用长度裁剪，不改变换行——避免流式时每条一段、落库再被压成一行 */
  if (s.length <= MAX_ASSISTANT_CHARS) {
    return s;
  }
  return s.slice(0, MAX_ASSISTANT_CHARS) + "…";
}

function applyConversationLimits(conv) {
  if (!conv || !Array.isArray(conv.messages)) return;
  conv.messages = trimToMaxRounds(conv.messages, MAX_ROUNDS_PER_CONV);
  conv.messages = conv.messages.map((m) =>
    m.role === "assistant" ? { ...m, content: clampAssistantContent(m.content) } : m,
  );
}

function getActiveConversation() {
  return activeId ? conversations.find((c) => c.id === activeId) : null;
}

function autoResizeTextarea() {
  questionInput.style.height = "auto";
  const next = Math.min(questionInput.scrollHeight, 180);
  questionInput.style.height = `${next}px`;
}

/* ---------- 侧边栏历史 ---------- */

function renderHistory() {
  historyList.innerHTML = "";
  if (conversations.length === 0) {
    historyEmpty.hidden = false;
    return;
  }
  historyEmpty.hidden = true;
  for (const item of conversations) {
    const row = document.createElement("div");
    row.className = "history-item" + (item.id === activeId ? " active" : "");
    row.dataset.id = item.id;

    const open = document.createElement("button");
    open.type = "button";
    open.className = "history-open";
    open.textContent = item.title;
    open.title = item.title;
    open.addEventListener("click", () => openConversation(item.id));

    const more = document.createElement("button");
    more.type = "button";
    more.className = "history-more";
    more.setAttribute("aria-label", "更多");
    more.textContent = "⋮";
    more.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleMenu(item.id, more);
    });

    row.append(open, more);
    historyList.append(row);
  }
}

function toggleMenu(id, anchor) {
  if (!historyMenu.hidden && menuTargetId === id) {
    closeMenu();
    return;
  }
  menuTargetId = id;
  historyMenu.hidden = false;
  const rect = anchor.getBoundingClientRect();
  const menuWidth = 176;
  historyMenu.style.top = `${rect.bottom + 6}px`;
  historyMenu.style.left = `${Math.min(rect.left, window.innerWidth - menuWidth - 8)}px`;
}

function closeMenu() {
  historyMenu.hidden = true;
  menuTargetId = null;
}

function deleteConversation(id) {
  if (activeRequest && activeRequest.conv && activeRequest.conv.id === id) {
    cancelActiveRequest();
  }
  conversations = conversations.filter((c) => c.id !== id);
  if (activeId === id) activeId = null;
  saveConversations();
  renderHistory();
  renderConversation();
}

function renameConversation(id) {
  const item = conversations.find((c) => c.id === id);
  if (!item) return;
  const next = window.prompt("重命名对话", item.title);
  if (next === null) return;
  const cleaned = next.trim();
  if (!cleaned) return;
  item.title = truncate(cleaned, 60);
  saveConversations();
  renderHistory();
}

historyMenu.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button || menuTargetId === null) return;
  const id = menuTargetId;
  closeMenu();
  if (button.dataset.action === "delete") {
    deleteConversation(id);
  } else if (button.dataset.action === "rename") {
    renameConversation(id);
  }
});

document.addEventListener("click", (event) => {
  if (!cvModelMenu.hidden && !event.target.closest("#cv-model-menu") && !event.target.closest("#cv-model-button")) {
    closeCvModelMenu();
  }
  if (historyMenu.hidden) return;
  if (event.target.closest("#history-menu")) return;
  if (event.target.closest(".history-more")) return;
  closeMenu();
});

window.addEventListener("resize", () => {
  closeMenu();
  closeCvModelMenu();
});
window.addEventListener("scroll", () => {
  closeMenu();
  closeCvModelMenu();
}, true);

/* ---------- 对话渲染 ---------- */

function appendMessageNode(role, content, options = {}) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  if (role === "assistant") {
    const avatar = document.createElement("img");
    avatar.className = "avatar";
    avatar.src = "/ui/logo.png";
    avatar.alt = "";
    wrap.append(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (options.loading ? " loading" : "");
  bubble.textContent = content;
  wrap.append(bubble);

  conversationEl.append(wrap);
  conversationEl.scrollTop = conversationEl.scrollHeight;
  return bubble;
}

function renderConversation() {
  conversationEl.innerHTML = "";
  const conv = getActiveConversation();
  if (!conv || conv.messages.length === 0) {
    conversationEl.append(conversationEmpty);
    conversationEmpty.hidden = false;
    return;
  }
  conversationEmpty.hidden = true;
  for (const msg of conv.messages) {
    appendMessageNode(msg.role, msg.content);
  }
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

function openConversation(id) {
  const item = conversations.find((c) => c.id === id);
  if (!item) return;
  if (activeRequest && activeRequest.conv && activeRequest.conv.id !== id) {
    cancelActiveRequest();
  }
  activeId = id;
  askStatus.textContent = "已加载";
  renderHistory();
  renderConversation();
}

function startNewChat() {
  cancelActiveRequest();
  activeId = null;
  questionInput.value = "";
  autoResizeTextarea();
  askStatus.textContent = "等待提问";
  renderHistory();
  renderConversation();
  questionInput.focus();
}

newChatBtn.addEventListener("click", startNewChat);

/* ---------- 摄像头状态 ---------- */

function setCameraStatus(text, state) {
  cameraStatus.textContent = text;
  cameraStatus.className = `status-pill ${state || ""}`.trim();
}

let lastCvPayload = null;

function drawDetectionOverlay() {
  if (!cameraOverlay || !streamFrame) return;
  const ctx = cameraOverlay.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const cw = streamFrame.clientWidth;
  const ch = streamFrame.clientHeight;
  cameraOverlay.width = Math.round(cw * dpr);
  cameraOverlay.height = Math.round(ch * dpr);
  cameraOverlay.style.width = `${cw}px`;
  cameraOverlay.style.height = `${ch}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  if (!lastCvPayload) return;
  const iw = Number(lastCvPayload.image_width) || 0;
  const ih = Number(lastCvPayload.image_height) || 0;
  const raw = lastCvPayload.objects || lastCvPayload.detections;
  const items = Array.isArray(raw) ? raw : [];
  if (iw <= 0 || ih <= 0 || items.length === 0) return;

  const scale = Math.min(cw / iw, ch / ih);
  if (!Number.isFinite(scale) || scale <= 0) return;
  const dispW = iw * scale;
  const dispH = ih * scale;
  const offX = (cw - dispW) / 2;
  const offY = (ch - dispH) / 2;

  ctx.strokeStyle = "rgba(255, 60, 60, 0.95)";
  ctx.lineWidth = 2;
  ctx.font = "12px system-ui, -apple-system, sans-serif";

  for (const obj of items) {
    const b = obj && obj.bbox_xyxy;
    if (!Array.isArray(b) || b.length < 4) continue;
    const x1 = offX + Number(b[0]) * scale;
    const y1 = offY + Number(b[1]) * scale;
    const x2 = offX + Number(b[2]) * scale;
    const y2 = offY + Number(b[3]) * scale;
    const w = x2 - x1;
    const h = y2 - y1;
    if (w <= 0 || h <= 0) continue;
    ctx.strokeRect(x1, y1, w, h);
    const label = String(obj.label != null ? obj.label : "?");
    const scoreText = obj.score != null && Number.isFinite(Number(obj.score)) ? ` ${Number(obj.score).toFixed(2)}` : "";
    const t = `${label}${scoreText}`;
    const pad = 4;
    const tw = ctx.measureText(t).width;
    const th = 16;
    const ty = Math.max(0, y1 - th - 2);
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fillRect(x1, ty, tw + pad * 2, th);
    ctx.fillStyle = "#b7ffb0";
    ctx.fillText(t, x1 + pad, ty + th - 4);
  }
}

if (streamFrame && typeof ResizeObserver !== "undefined") {
  new ResizeObserver(() => {
    drawDetectionOverlay();
  }).observe(streamFrame);
}

async function refreshStats() {
  try {
    const [statsRes, cvRes] = await Promise.all([
      fetch("/stats", { cache: "no-store" }),
      fetch("/cv_result.json", { cache: "no-store" }),
    ]);
    if (!statsRes.ok) {
      throw new Error(`HTTP ${statsRes.status}`);
    }
    const stats = await statsRes.json();
    statInferMs.textContent = Number.isFinite(stats.infer_ms) ? `${stats.infer_ms} ms` : "-";
    statFps.textContent = Number.isFinite(stats.infer_fps) ? stats.infer_fps : "-";
    setCameraStatus("运行中", "ok");
    if (cvRes.ok) {
      lastCvPayload = await cvRes.json();
    } else {
      lastCvPayload = null;
    }
    drawDetectionOverlay();
  } catch (error) {
    lastCvPayload = null;
    drawDetectionOverlay();
    setCameraStatus("未就绪", "error");
  }
}

/* ---------- 提问 ---------- */

questionInput.addEventListener("input", autoResizeTextarea);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    askForm.requestSubmit();
  }
});

function setAskButtonMode(mode) {
  if (mode === "stop") {
    askButton.dataset.mode = "stop";
    askButton.dataset.tooltip = "停止回答";
    askButton.setAttribute("aria-label", "停止回答");
    askButton.type = "button";
    askButton.disabled = false;
  } else {
    askButton.dataset.mode = "send";
    delete askButton.dataset.tooltip;
    askButton.setAttribute("aria-label", "发送");
    askButton.type = "submit";
    askButton.disabled = false;
  }
}

function cancelActiveRequest(label = "（已中断）") {
  if (!activeRequest) return;
  const prev = activeRequest;
  activeRequest = null;
  try {
    prev.controller.abort();
  } catch (_) {
    /* noop */
  }
  if (prev.conv) {
    prev.conv.messages.push({ role: "assistant", content: clampAssistantContent(label) });
    applyConversationLimits(prev.conv);
    prev.conv.updatedAt = Date.now();
    saveConversations();
    if (activeId === prev.conv.id) {
      renderConversation();
    }
  }
  setAskButtonMode("send");
  askStatus.textContent = "已中断";
}

askButton.addEventListener("click", (event) => {
  if (askButton.dataset.mode === "stop") {
    event.preventDefault();
    cancelActiveRequest();
  }
});

async function readAskStream(response, pendingBubble) {
  if (!response.body) {
    const data = await response.json();
    return data.answer || "";
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answer = "";
  let started = false;

  const handleEvent = (rawEvent) => {
    const lines = rawEvent.split(/\r?\n/);
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (dataLines.length === 0) return;

    const payload = JSON.parse(dataLines.join("\n"));
    if (payload.error) {
      throw new Error(payload.error);
    }
    if (payload.delta) {
      if (answer.length >= MAX_ASSISTANT_CHARS) {
        return;
      }
      const remaining = MAX_ASSISTANT_CHARS - answer.length;
      let piece = payload.delta;
      if (piece.length > remaining) {
        piece = piece.slice(0, remaining);
      }
      if (!started) {
        pendingBubble.classList.remove("loading");
        pendingBubble.textContent = "";
        started = true;
      }
      answer += piece;
      pendingBubble.textContent = answer;
      conversationEl.scrollTop = conversationEl.scrollHeight;
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let splitAt;
    while ((splitAt = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, splitAt).trim();
      buffer = buffer.slice(splitAt + 2);
      if (rawEvent) handleEvent(rawEvent);
    }

    if (done) break;
  }

  const trailing = buffer.trim();
  if (trailing) handleEvent(trailing);
  return answer;
}

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  cancelActiveRequest();

  let conv = getActiveConversation();
  if (!conv) {
    conv = {
      id: makeId(),
      title: truncate(question, 32),
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    conversations.unshift(conv);
    activeId = conv.id;
  }

  conv.messages.push({ role: "user", content: question });
  applyConversationLimits(conv);
  conv.updatedAt = Date.now();
  saveConversations();

  conversationEmpty.hidden = true;
  renderConversation();
  const pendingBubble = appendMessageNode("assistant", "正在思考…", { loading: true });

  questionInput.value = "";
  autoResizeTextarea();
  askStatus.textContent = "思考中…";
  renderHistory();

  const controller = new AbortController();
  const myRequest = { controller, bubble: pendingBubble, conv };
  activeRequest = myRequest;
  setAskButtonMode("stop");

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        stream: true,
        cv_model: activeCvModel,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `HTTP ${response.status}`);
    }

    const contentType = response.headers.get("content-type") || "";
    let answer;
    if (contentType.includes("text/event-stream")) {
      answer = await readAskStream(response, pendingBubble);
    } else {
      const data = await response.json();
      answer = data.answer || "";
    }
    answer = answer || "模型没有返回文本。";
    answer = clampAssistantContent(answer);

    conv.messages.push({ role: "assistant", content: answer });
    applyConversationLimits(conv);
    conv.updatedAt = Date.now();
    saveConversations();
    renderConversation();
    askStatus.textContent = "完成";
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    const errText = clampAssistantContent(`调用失败：${error.message}`);
    conv.messages.push({
      role: "assistant",
      content: errText,
    });
    applyConversationLimits(conv);
    conv.updatedAt = Date.now();
    saveConversations();
    renderConversation();
    askStatus.textContent = "失败";
  } finally {
    if (activeRequest === myRequest) {
      activeRequest = null;
      setAskButtonMode("send");
    }
    conversationEl.scrollTop = conversationEl.scrollHeight;
    questionInput.focus();
  }
});

renderHistory();
renderConversation();
renderCvModel();
syncCvModel(activeCvModel).catch(() => setCameraStatus("切换失败", "error"));
autoResizeTextarea();
refreshStats();
setInterval(refreshStats, 2000);
