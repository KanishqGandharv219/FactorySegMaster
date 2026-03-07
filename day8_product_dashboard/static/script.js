console.log("FactoryTwin v2.7: Logic initializing...");

let ws = null;
let wsPingTimer = null;
let activeTab = "image";
let lastProgressLog = "";

function log(msg) {
    console.log(`[DASHBOARD LOG] ${msg}`);
    const box = document.getElementById("event-log");
    if (!box) return;

    const line = document.createElement("div");
    line.className = "log-line";
    line.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
}

function setLogLines(logText) {
    const box = document.getElementById("event-log");
    if (!box || !logText) return;

    const lines = logText
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

    if (!lines.length) return;

    box.innerHTML = "";
    for (const lineText of lines) {
        const line = document.createElement("div");
        line.className = "log-line";
        line.innerText = lineText;
        box.appendChild(line);
    }
    box.scrollTop = box.scrollHeight;
}

function updateUIStats(stats) {
    if (!stats) return;

    const elements = {
        "stat-risk": stats.risk_score,
        "stat-workers": stats.worker_count,
        "stat-vehicles": stats.vehicle_count,
        "stat-violations": stats.violation_count
    };

    for (const [id, val] of Object.entries(elements)) {
        const el = document.getElementById(id);
        if (!el) continue;

        el.innerText = val;
        if (id === "stat-risk") {
            el.className = `stat-value ${val > 60 ? "red" : (val > 30 ? "accent" : "green")}`;
        }
    }
}

function setTab(tab) {
    activeTab = tab;

    document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`tab-${tab}`);
    if (activeBtn) activeBtn.classList.add("active");

    const imgSec = document.getElementById("image-section");
    const vidSec = document.getElementById("video-section");
    if (imgSec) imgSec.style.display = tab === "image" ? "block" : "none";
    if (vidSec) vidSec.style.display = tab === "video" ? "block" : "none";

    log(`Switched to ${tab.toUpperCase()} pipeline.`);
}

function previewMedia(type) {
    const inputId = type === "image" ? "img-input" : "vid-input";
    const input = document.getElementById(inputId);
    if (!input || !input.files[0]) return;

    const file = input.files[0];
    const placeholder = document.getElementById("viewer-placeholder");
    const outImg = document.getElementById("out-img");
    const outVid = document.getElementById("out-vid");

    if (placeholder) placeholder.style.display = "none";

    if (type === "image" && outImg) {
        if (outVid) {
            outVid.pause();
            outVid.style.display = "none";
        }
        outImg.style.display = "block";
        outImg.src = URL.createObjectURL(file);
        log("Frame staged for static inference.");
    } else if (type === "video" && outVid) {
        if (outImg) outImg.style.display = "none";
        outVid.style.display = "block";
        outVid.src = URL.createObjectURL(file);
        log("Sequence staged for temporal processing.");
    }
}

function setInputFile(input, file) {
    if (!input || !file || typeof DataTransfer === "undefined") return false;
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    input.files = dataTransfer.files;
    return true;
}

function getClipboardImageFromEvent(event) {
    const items = event?.clipboardData?.items || [];
    for (const item of items) {
        if (item.kind === "file" && item.type.startsWith("image/")) {
            return item.getAsFile();
        }
    }
    return null;
}

async function readClipboardImageFile() {
    if (!navigator.clipboard || !navigator.clipboard.read) {
        throw new Error("Clipboard read API is not available in this browser context.");
    }

    const clipboardItems = await navigator.clipboard.read();
    for (const item of clipboardItems) {
        const imageType = item.types.find((type) => type.startsWith("image/"));
        if (imageType) {
            const blob = await item.getType(imageType);
            return new File([blob], `clipboard-${Date.now()}.png`, { type: blob.type || imageType });
        }
    }

    return null;
}

function stageClipboardImage(file) {
    const imageInput = document.getElementById("img-input");
    if (!imageInput) return false;

    const assigned = setInputFile(imageInput, file);
    if (!assigned) {
        log("Clipboard image staging is unavailable in this browser.");
        return false;
    }

    previewMedia("image");
    log("Clipboard image captured and staged.");
    return true;
}

async function pasteImageFromClipboard() {
    try {
        const file = await readClipboardImageFile();
        if (!file) {
            alert("No image found in clipboard. Copy an image first, then paste.");
            return;
        }

        const staged = stageClipboardImage(file);
        if (!staged) {
            alert("Unable to stage clipboard image in this browser.");
        }
    } catch (err) {
        log(`Clipboard paste unavailable: ${err.message}`);
        alert("Clipboard access blocked. Use Ctrl+V in the page or drag/drop the image.");
    }
}

function bindDropZone(dropZoneId, inputId, type) {
    const zone = document.getElementById(dropZoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    const cancel = (event) => {
        event.preventDefault();
        event.stopPropagation();
    };

    ["dragenter", "dragover"].forEach((eventName) => {
        zone.addEventListener(eventName, (event) => {
            cancel(event);
            zone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        zone.addEventListener(eventName, (event) => {
            cancel(event);
            zone.classList.remove("drag-over");
        });
    });

    zone.addEventListener("drop", (event) => {
        const fileList = event.dataTransfer?.files;
        if (!fileList || fileList.length === 0) return;

        const file = fileList[0];
        const expected = type === "image" ? "image/" : "video/";
        if (!file.type.startsWith(expected)) {
            log(`Rejected ${type} drop: invalid file type ${file.type || "unknown"}.`);
            return;
        }

        if (!setInputFile(input, file)) {
            log("Drag-drop staging failed in this browser.");
            return;
        }

        previewMedia(type);
    });
}

function getRequestedVideoSeconds() {
    const el = document.getElementById("vid-seconds");
    if (!el) return 0;

    const raw = Number.parseFloat(el.value);
    if (!Number.isFinite(raw) || raw < 0) return 0;
    return raw;
}

function getOutputPreset() {
    const el = document.getElementById("vid-preset");
    const value = (el?.value || "original").toLowerCase();
    return ["original", "720p", "480p"].includes(value) ? value : "original";
}
function buildFormData(file, mediaType = "image") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("enable_ppe", document.getElementById("ppe-toggle")?.checked || false);
    formData.append("enable_sam2", document.getElementById("sam2-toggle")?.checked || false);

    if (mediaType === "video") {
        formData.append("process_seconds", getRequestedVideoSeconds());
        formData.append("output_preset", getOutputPreset());
    }

    return formData;
}

async function parseApiResponse(response) {
    const text = await response.text();
    let data = {};

    try {
        data = text ? JSON.parse(text) : {};
    } catch {
        data = { status: "error", message: text || "Unexpected server response" };
    }

    if (!response.ok) {
        const message = data.message || `HTTP ${response.status}`;
        throw new Error(message);
    }

    return data;
}

async function waitForMediaReady(url, attempts = 12, delayMs = 450) {
    for (let i = 0; i < attempts; i += 1) {
        try {
            const probe = await fetch(`${url}?probe=${Date.now()}_${i}`, {
                method: "HEAD",
                cache: "no-store"
            });

            if (probe.ok) {
                const contentLength = Number(probe.headers.get("content-length") || "0");
                if (contentLength > 0) return true;
            }
        } catch {
            // Retry until attempts are exhausted.
        }

        await new Promise((resolve) => setTimeout(resolve, delayMs));
    }

    return false;
}

function resetProgressUI() {
    const overlay = document.getElementById("progress-overlay");
    const fill = document.getElementById("progress-fill");
    const pctTxt = document.getElementById("progress-pct");
    const label = document.getElementById("progress-label");

    if (overlay) overlay.style.display = "block";
    if (fill) fill.style.width = "0%";
    if (pctTxt) pctTxt.innerText = "0%";
    if (label) label.innerText = "PROCESSING...";

    lastProgressLog = "";
}

async function processImage() {
    const file = document.getElementById("img-input")?.files[0];
    if (!file) return alert("Please select an image file first.");

    log("Initiating server-side inference...");

    try {
        const response = await fetch("/api/process-image", {
            method: "POST",
            body: buildFormData(file, "image")
        });
        const data = await parseApiResponse(response);

        if (data.status !== "success") {
            throw new Error(data.message || "Image processing failed");
        }

        const outImg = document.getElementById("out-img");
        const outVid = document.getElementById("out-vid");
        const placeholder = document.getElementById("viewer-placeholder");

        if (placeholder) placeholder.style.display = "none";
        if (outVid) {
            outVid.pause();
            outVid.style.display = "none";
        }
        if (outImg) {
            outImg.src = `${data.processed_url}?t=${Date.now()}`;
            outImg.style.display = "block";
        }

        updateUIStats(data.stats);
        setLogLines(data.logs);
        log("Inference complete. Dashboard updated.");
    } catch (err) {
        log(`Image pipeline error: ${err.message}`);
        console.error(err);
    }
}

async function processVideo() {
    const file = document.getElementById("vid-input")?.files[0];
    if (!file) return alert("Please select a video file first.");

    const requestedSeconds = getRequestedVideoSeconds();
    const outputPreset = getOutputPreset();
    log(`Starting background video processing task (${requestedSeconds > 0 ? `${requestedSeconds}s` : "full video"}, ${outputPreset})...`);
    resetProgressUI();

    try {
        const response = await fetch("/api/process-video", {
            method: "POST",
            body: buildFormData(file, "video")
        });
        const data = await parseApiResponse(response);

        if (data.status !== "started") {
            throw new Error(data.message || "Video task could not start");
        }

        startWebSocket(data.task_id);
    } catch (err) {
        const overlay = document.getElementById("progress-overlay");
        if (overlay) overlay.style.display = "none";
        log(`Video pipeline error: ${err.message}`);
        console.error(err);
    }
}

function clearWebSocket() {
    if (wsPingTimer) {
        clearInterval(wsPingTimer);
        wsPingTimer = null;
    }

    if (ws) {
        try {
            ws.close();
        } catch {
            // No-op.
        }
        ws = null;
    }
}

function startWebSocket(taskId) {
    clearWebSocket();

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/progress/${taskId}`);

    ws.onopen = () => {
        log("Progress channel established.");
        wsPingTimer = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send("ping");
            }
        }, 10000);
    };

    ws.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        if (!data || !data.type) return;

        if (data.type === "progress") {
            const total = Math.max(1, Number(data.total) || 1);
            const current = Math.min(Number(data.current) || 0, total);
            const pct = Math.round((current / total) * 100);

            const fill = document.getElementById("progress-fill");
            const pctTxt = document.getElementById("progress-pct");
            const label = document.getElementById("progress-label");

            if (fill) fill.style.width = `${pct}%`;
            if (pctTxt) pctTxt.innerText = `${pct}%`;
            if (label) label.innerText = `PROCESSING FRAME ${current}/${total}`;

            updateUIStats(data.stats);

            if (data.latest_log && data.latest_log !== lastProgressLog) {
                lastProgressLog = data.latest_log;
                log(data.latest_log);
            }
            return;
        }

        if (data.type === "complete") {
            log("Pipeline synchronized. Rendering output.");

            const overlay = document.getElementById("progress-overlay");
            const outImg = document.getElementById("out-img");
            const outVid = document.getElementById("out-vid");
            const placeholder = document.getElementById("viewer-placeholder");

            if (overlay) overlay.style.display = "none";
            if (placeholder) placeholder.style.display = "none";
            if (outImg) outImg.style.display = "none";

            if (outVid) {
                const ready = await waitForMediaReady(data.processed_url);
                if (!ready) {
                    log("Processed video file is delayed; attempting playback anyway.");
                }

                outVid.onerror = () => {
                    log("Processed video could not be decoded. Try another source clip or codec.");
                };
                outVid.src = `${data.processed_url}?t=${Date.now()}`;
                outVid.load();
                outVid.style.display = "block";
                outVid.play().catch(() => {
                    log("Output video ready. Press play to view.");
                });
            }

            if (data.codec || data.resolution) {
                log(`Output ready (${data.codec || "unknown codec"}, ${data.resolution || "unknown resolution"}).`);
            }

            updateUIStats(data.stats);
            setLogLines(data.logs);
            clearWebSocket();
            return;
        }

        if (data.type === "error") {
            const overlay = document.getElementById("progress-overlay");
            if (overlay) overlay.style.display = "none";
            log(`Video processing failed: ${data.message || "Unknown error"}`);
            clearWebSocket();
        }
    };

    ws.onerror = () => {
        const overlay = document.getElementById("progress-overlay");
        if (overlay) overlay.style.display = "none";
        log("WebSocket connection error during video processing.");
    };

    ws.onclose = () => {
        if (wsPingTimer) {
            clearInterval(wsPingTimer);
            wsPingTimer = null;
        }
        ws = null;
    };
}

document.addEventListener("DOMContentLoaded", () => {
    log("FactoryTwin v2.7 engine ready.");

    document.getElementById("tab-image")?.addEventListener("click", () => setTab("image"));
    document.getElementById("tab-video")?.addEventListener("click", () => setTab("video"));

    document.getElementById("img-input")?.addEventListener("change", () => previewMedia("image"));
    document.getElementById("vid-input")?.addEventListener("change", () => previewMedia("video"));

    bindDropZone("img-drop-zone", "img-input", "image");
    bindDropZone("vid-drop-zone", "vid-input", "video");

    window.addEventListener("paste", (event) => {
        if (activeTab !== "image") return;

        const file = getClipboardImageFromEvent(event);
        if (!file) return;

        event.preventDefault();
        stageClipboardImage(file);
    });
});

(() => {
    const canvas = document.getElementById("particles");
    if (!canvas) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const pointer = { x: 0, y: 0, active: false };
    const center = { x: 0, y: 0 };

    let width = 0;
    let height = 0;
    let dpr = 1;
    let dots = [];
    let rafId = null;
    let lastTime = 0;

    const mobile = window.matchMedia("(max-width: 920px)").matches;
    const config = {
        baseSpacing: mobile ? 32 : 24,
        minSpacing: mobile ? 26 : 19,
        maxSpacing: mobile ? 40 : 34,
        targetDots: mobile ? 740 : 1700,
        jitter: mobile ? 7 : 10,
        impactRadius: mobile ? 190 : 270,
        spring: mobile ? 0.022 : 0.017,
        pull: mobile ? 0.085 : 0.11,
        swirl: mobile ? 0.055 : 0.075,
        noise: mobile ? 0.03 : 0.05,
        damping: mobile ? 0.87 : 0.9,
        drift: mobile ? 0.12 : 0.17
    };

    const colorStops = [
        [255, 145, 70],
        [240, 176, 104],
        [105, 190, 182],
        [38, 133, 176]
    ];

    function lerp(a, b, t) {
        return a + (b - a) * t;
    }

    function sampleColor(t) {
        const clamped = Math.max(0, Math.min(1, t));
        const idx = clamped * (colorStops.length - 1);
        const lo = Math.floor(idx);
        const hi = Math.min(lo + 1, colorStops.length - 1);
        const part = idx - lo;

        return [
            lerp(colorStops[lo][0], colorStops[hi][0], part),
            lerp(colorStops[lo][1], colorStops[hi][1], part),
            lerp(colorStops[lo][2], colorStops[hi][2], part)
        ];
    }

    function buildDots() {
        dots = [];

        const area = Math.max(1, width * height);
        const adaptiveSpacing = Math.max(
            config.minSpacing,
            Math.min(config.maxSpacing, Math.sqrt(area / config.targetDots))
        );
        const spacing = Math.max(config.baseSpacing, adaptiveSpacing);
        const jitter = Math.min(config.jitter, spacing * 0.34);

        for (let y = spacing * 0.5; y <= height + spacing * 0.5; y += spacing) {
            for (let x = spacing * 0.5; x <= width + spacing * 0.5; x += spacing) {
                const jx = (Math.random() - 0.5) * jitter;
                const jy = (Math.random() - 0.5) * jitter;
                const baseX = x + jx;
                const baseY = y + jy;

                dots.push({
                    hx: baseX,
                    hy: baseY,
                    x: baseX,
                    y: baseY,
                    vx: 0,
                    vy: 0,
                    size: 0.75 + Math.random() * 0.9,
                    phase: Math.random() * Math.PI * 2,
                    spin: Math.random() < 0.5 ? -1 : 1
                });
            }
        }
    }

    function setCanvasSize() {
        dpr = Math.min(window.devicePixelRatio || 1, 1.8);
        width = Math.max(window.innerWidth, document.documentElement.clientWidth || 0);
        height = Math.max(window.innerHeight, document.documentElement.clientHeight || 0);

        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        if (!center.x && !center.y) {
            center.x = width * 0.55;
            center.y = height * 0.48;
        }

        buildDots();
    }

    function setPointer(x, y) {
        pointer.x = x;
        pointer.y = y;
        pointer.active = true;
    }

    function draw(ts) {
        const dt = Math.min(1.9, (ts - (lastTime || ts)) / 16.667 || 1);
        lastTime = ts;

        const idleX = width * 0.53 + Math.cos(ts * 0.00022) * width * config.drift;
        const idleY = height * 0.48 + Math.sin(ts * 0.00028) * height * (config.drift * 0.75);

        const targetX = pointer.active ? pointer.x : idleX;
        const targetY = pointer.active ? pointer.y : idleY;

        center.x += (targetX - center.x) * 0.08;
        center.y += (targetY - center.y) * 0.08;

        ctx.clearRect(0, 0, width, height);

        const fieldRadius = config.impactRadius * (pointer.active ? 1.06 : 0.84);
        const field = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, fieldRadius * 1.2);
        field.addColorStop(0, "rgba(255, 150, 70, 0.12)");
        field.addColorStop(0.45, "rgba(86, 168, 181, 0.1)");
        field.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = field;
        ctx.fillRect(0, 0, width, height);

        for (const dot of dots) {
            const toHomeX = dot.hx - dot.x;
            const toHomeY = dot.hy - dot.y;

            dot.vx += toHomeX * config.spring * dt;
            dot.vy += toHomeY * config.spring * dt;

            const dx = dot.x - center.x;
            const dy = dot.y - center.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 0.0001;

            let influence = 0;
            if (dist < fieldRadius) {
                influence = 1 - dist / fieldRadius;
                const radialX = dx / dist;
                const radialY = dy / dist;

                dot.vx += -radialX * influence * influence * config.pull * dt;
                dot.vy += -radialY * influence * influence * config.pull * dt;

                dot.vx += -radialY * influence * config.swirl * dot.spin * dt;
                dot.vy += radialX * influence * config.swirl * dot.spin * dt;

                const wobble = Math.sin(ts * 0.0018 + dot.phase + dist * 0.016) * config.noise;
                dot.vx += radialX * wobble * influence * dt;
                dot.vy += radialY * wobble * influence * dt;
            }

            dot.vx *= config.damping;
            dot.vy *= config.damping;

            dot.x += dot.vx * dt;
            dot.y += dot.vy * dt;

            const speed = Math.sqrt(dot.vx * dot.vx + dot.vy * dot.vy);
            const tone = 1 - influence;
            const rgb = sampleColor(tone);
            const alpha = 0.14 + influence * 0.66;

            const stretch = 1 + Math.min(1.2, speed * 2.8 + influence * 0.85);
            const radiusX = dot.size * stretch;
            const radiusY = dot.size * (0.9 - Math.min(0.4, speed));
            const angle = Math.atan2(dot.vy, dot.vx);

            if (influence > 0.33) {
                ctx.strokeStyle = `rgba(${rgb[0] | 0}, ${rgb[1] | 0}, ${rgb[2] | 0}, ${Math.min(0.22, influence * 0.24)})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(dot.hx, dot.hy);
                ctx.lineTo(dot.x, dot.y);
                ctx.stroke();
            }

            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.translate(dot.x, dot.y);
            ctx.rotate(angle);
            ctx.fillStyle = `rgb(${rgb[0] | 0}, ${rgb[1] | 0}, ${rgb[2] | 0})`;
            ctx.beginPath();
            ctx.ellipse(0, 0, radiusX, Math.max(0.28, radiusY), 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }

        rafId = requestAnimationFrame(draw);
    }

    function handleMouseMove(event) {
        setPointer(event.clientX, event.clientY);
    }

    function handleTouchMove(event) {
        const touch = event.touches[0];
        if (!touch) return;
        setPointer(touch.clientX, touch.clientY);
    }

    function handlePointerLeave() {
        pointer.active = false;
    }

    setCanvasSize();

    window.addEventListener("resize", setCanvasSize);
    window.addEventListener("orientationchange", setCanvasSize);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseleave", handlePointerLeave);
    window.addEventListener("touchstart", handleTouchMove, { passive: true });
    window.addEventListener("touchmove", handleTouchMove, { passive: true });
    window.addEventListener("touchend", handlePointerLeave);

    if (reduceMotion) {
        ctx.clearRect(0, 0, width, height);
        return;
    }

    rafId = requestAnimationFrame(draw);

    window.addEventListener("beforeunload", () => {
        if (rafId) cancelAnimationFrame(rafId);
    });
})();
