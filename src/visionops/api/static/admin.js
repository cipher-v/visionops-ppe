const list = document.querySelector("#job-list");
const form = document.querySelector("#upload-form");
const input = document.querySelector("#video-file");
const dropZone = document.querySelector("#drop-zone");
const fileName = document.querySelector("#file-name");
const message = document.querySelector("#upload-message");
const uploadStages = document.querySelector("#upload-stages");
let videos = JSON.parse(document.querySelector("#initial-videos").textContent);
let renderedSignature = "";
let trackedUploadId = null;

const escapeHtml = (value) => {
  const element = document.createElement("span");
  element.textContent = value ?? "";
  return element.innerHTML;
};

function elapsedLabel(createdAt) {
  const elapsed = Math.max(0, Math.floor((Date.now() - Date.parse(createdAt)) / 1000));
  const minutes = Math.floor(elapsed / 60);
  const seconds = String(elapsed % 60).padStart(2, "0");
  return `${minutes}:${seconds} elapsed`;
}

function processingPlaceholder(video) {
  if (video.status === "queued") {
    return '<div class="processing-placeholder"><strong>Queued</strong><span>Waiting for the video worker</span></div>';
  }
  if (video.status === "running") {
    return `<div class="processing-placeholder"><span class="spinner"></span><strong>Processing with VisionOps...</strong><span data-elapsed="${video.id}" data-created-at="${video.created_at}">${elapsedLabel(video.created_at)}</span></div>`;
  }
  if (video.status === "failed") {
    return `<div class="processing-placeholder"><strong>Processing failed</strong><span>${escapeHtml(video.error_message ?? "The video could not be processed.")}</span></div>`;
  }
  return '<div class="processing-placeholder"><strong>Finalizing preview...</strong><span>Preparing browser-compatible media</span></div>';
}

function render() {
  const signature = JSON.stringify(videos);
  if (signature === renderedSignature) return;
  renderedSignature = signature;
  if (!videos.length) {
    list.innerHTML = '<div class="empty-state compact"><h3>No processing jobs</h3><p>Upload the first demonstration video above.</p></div>';
    return;
  }
  list.innerHTML = videos.map((video) => {
    const canPreview = video.status === "completed" && video.video_url && video.original_url;
    const controls = canPreview
      ? `<button class="button small ${video.published ? "secondary" : "primary"}" data-action="${video.published ? "unpublish" : "publish"}" data-id="${video.id}">${video.published ? "Unpublish" : "Publish"}</button>`
      : "";
    const preview = canPreview
      ? `<div class="comparison-player admin-comparison" data-sync-player>
          <section class="video-panel"><div class="video-label">Original</div><video controls preload="metadata" playsinline data-video-role="original"><source src="${video.original_url}" type="video/mp4"></video></section>
          <section class="video-panel output-panel"><div class="video-label">VisionOps Output</div><video controls preload="metadata" playsinline muted data-video-role="output"><source src="${video.video_url}" type="video/mp4"></video></section>
        </div>`
      : processingPlaceholder(video);
    return `<article class="job-card">
      <div class="job-preview">${preview}</div>
      <div class="job-content">
        <div class="job-title"><div><span class="status-badge ${video.status}">${escapeHtml(video.status)}</span><h3>${escapeHtml(video.title)}</h3></div><span class="job-date">${escapeHtml(video.created_at.slice(0, 10))}</span></div>
        <p class="source-name">${escapeHtml(video.source_filename)}</p>
        ${video.error_message ? `<p class="notice error">${escapeHtml(video.error_message)}</p>` : ""}
        <div class="job-stats"><span><strong>${video.incident_count}</strong> incidents</span><span><strong>${video.summary.frames_processed ?? "—"}</strong> frames</span><span><strong>${video.summary.unique_person_tracks ?? "—"}</strong> tracks</span></div>
        <div class="job-actions">${controls}<button class="button small danger" data-action="delete" data-id="${video.id}" ${["queued", "running"].includes(video.status) ? "disabled" : ""}>Delete</button></div>
      </div>
    </article>`;
  }).join("");
  list.querySelectorAll("video").forEach((video) => video.load());
  window.VisionOpsSync?.initialize(list);
}

function setUploadProgress(activeStage) {
  const order = ["uploading", "uploaded", "queued", "processing", "completed"];
  uploadStages.hidden = false;
  const activeIndex = order.indexOf(activeStage);
  uploadStages.querySelectorAll("li").forEach((item) => {
    const index = order.indexOf(item.dataset.uploadStage);
    item.classList.toggle("done", index < activeIndex || activeStage === "completed");
    item.classList.toggle("active", index === activeIndex && activeStage !== "completed");
  });
}

function updateTrackedUpload() {
  if (!trackedUploadId) return;
  const video = videos.find((item) => item.id === trackedUploadId);
  if (!video) return;
  if (video.status === "queued") setUploadProgress("queued");
  if (video.status === "running") setUploadProgress("processing");
  if (video.status === "completed") {
    setUploadProgress("completed");
    message.textContent = "Processing complete. Preview is ready.";
    trackedUploadId = null;
  }
  if (video.status === "failed") {
    message.textContent = video.error_message ?? "Processing failed.";
    trackedUploadId = null;
  }
}

async function refresh() {
  const response = await fetch("/api/admin/videos", {headers: {Accept: "application/json"}});
  if (response.status === 401) { window.location = "/admin/login"; return; }
  if (response.ok) {
    videos = await response.json();
    render();
    updateTrackedUpload();
  }
}

input.addEventListener("change", () => { fileName.textContent = input.files[0]?.name ?? "No file selected"; });
["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("active"); }));
["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("active"); }));
dropZone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length) {
    const transfer = new DataTransfer();
    transfer.items.add(event.dataTransfer.files[0]);
    input.files = transfer.files;
    fileName.textContent = input.files[0].name;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setUploadProgress("uploading");
  message.textContent = "Uploading source video...";
  const response = await fetch("/api/admin/videos", {method: "POST", body: new FormData(form)});
  if (response.ok) {
    const accepted = await response.json();
    trackedUploadId = accepted.id;
    setUploadProgress("uploaded");
    message.textContent = "Upload complete. Job accepted.";
    form.reset();
    fileName.textContent = "No file selected";
    await refresh();
  } else {
    const body = await response.json().catch(() => ({}));
    message.textContent = body.error?.message ?? "Upload failed.";
    uploadStages.hidden = true;
  }
});

list.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const {action, id} = button.dataset;
  if (action === "delete" && !window.confirm("Delete this job and its generated output?")) return;
  button.disabled = true;
  const response = await fetch(`/api/admin/videos/${id}${action === "delete" ? "" : `/${action}`}`, {method: action === "delete" ? "DELETE" : "POST"});
  if (response.ok) await refresh(); else button.disabled = false;
});

render();
window.setInterval(refresh, 1500);
window.setInterval(() => {
  document.querySelectorAll("[data-elapsed]").forEach((element) => {
    element.textContent = elapsedLabel(element.dataset.createdAt);
  });
}, 1000);
