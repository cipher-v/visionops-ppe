(() => {
  const initialized = new WeakSet();
  const DRIFT_TOLERANCE_SECONDS = 0.25;

  function initialize(root = document) {
    root.querySelectorAll("[data-sync-player]").forEach((container) => {
      if (initialized.has(container)) return;
      const original = container.querySelector('[data-video-role="original"]');
      const output = container.querySelector('[data-video-role="output"]');
      if (!original || !output) return;
      initialized.add(container);
      output.muted = true;
      output.defaultMuted = true;

      let mirroring = false;
      let leader = original;
      const partner = (video) => video === original ? output : original;
      const ready = (video) => video.readyState >= HTMLMediaElement.HAVE_METADATA;
      const unlock = () => window.setTimeout(() => { mirroring = false; }, 0);
      const align = (source, target) => {
        if (!ready(source) || !ready(target) || !Number.isFinite(source.currentTime)) return;
        if (Math.abs(source.currentTime - target.currentTime) <= DRIFT_TOLERANCE_SECONDS) return;
        const duration = Number.isFinite(target.duration) ? target.duration : source.currentTime;
        target.currentTime = Math.min(source.currentTime, duration);
      };

      [original, output].forEach((video) => {
        video.addEventListener("play", () => {
          if (mirroring) return;
          leader = video;
          const other = partner(video);
          if (!ready(other)) return;
          mirroring = true;
          align(video, other);
          other.playbackRate = video.playbackRate;
          output.muted = true;
          other.play().catch(() => {}).finally(unlock);
        });
        video.addEventListener("pause", () => {
          if (mirroring) return;
          const other = partner(video);
          if (!ready(other) || other.paused) return;
          leader = video;
          mirroring = true;
          other.pause();
          unlock();
        });
        video.addEventListener("seeking", () => {
          if (mirroring) return;
          const other = partner(video);
          if (!ready(other)) return;
          leader = video;
          mirroring = true;
          align(video, other);
          unlock();
        });
        video.addEventListener("ratechange", () => {
          if (mirroring) return;
          const other = partner(video);
          if (!ready(other)) return;
          leader = video;
          mirroring = true;
          other.playbackRate = video.playbackRate;
          unlock();
        });
        video.addEventListener("canplay", () => {
          const other = partner(video);
          if (mirroring || !ready(other) || other.paused || !video.paused) return;
          mirroring = true;
          align(other, video);
          video.playbackRate = other.playbackRate;
          video.play().catch(() => {}).finally(unlock);
        });
      });

      output.addEventListener("volumechange", () => {
        if (!output.muted) output.muted = true;
      });

      const driftTimer = window.setInterval(() => {
        if (!container.isConnected) {
          window.clearInterval(driftTimer);
          return;
        }
        if (!ready(original) || !ready(output)) return;
        if (original.paused || output.paused || original.seeking || output.seeking) return;
        const other = partner(leader);
        if (Math.abs(leader.currentTime - other.currentTime) > DRIFT_TOLERANCE_SECONDS) {
          mirroring = true;
          align(leader, other);
          unlock();
        }
      }, 750);
    });
  }

  window.VisionOpsSync = {initialize};
  document.addEventListener("DOMContentLoaded", () => initialize());
})();
