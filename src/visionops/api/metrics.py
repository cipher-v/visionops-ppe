"""Low-cardinality Prometheus service metrics."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class ServiceMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "visionops_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.failures = Counter(
            "visionops_request_failures_total",
            "Failed HTTP requests",
            registry=self.registry,
        )
        self.images = Counter(
            "visionops_image_inferences_total",
            "Image inferences",
            registry=self.registry,
        )
        self.jobs_submitted = Counter(
            "visionops_video_jobs_submitted_total", "Video jobs", registry=self.registry
        )
        self.jobs_completed = Counter(
            "visionops_video_jobs_completed_total",
            "Completed jobs",
            ["status"],
            registry=self.registry,
        )
        self.active_jobs = Gauge(
            "visionops_active_video_jobs", "Active video jobs", registry=self.registry
        )
        self.detector_latency = Histogram(
            "visionops_detector_latency_seconds",
            "Detector latency",
            registry=self.registry,
        )
        self.pipeline_latency = Histogram(
            "visionops_pipeline_latency_seconds",
            "Video pipeline latency",
            registry=self.registry,
        )
        self.incidents = Counter(
            "visionops_incidents_emitted_total",
            "Incidents emitted",
            registry=self.registry,
        )
        self.drift_windows = Counter(
            "visionops_drift_windows_total",
            "Diagnostic drift windows",
            ["level"],
            registry=self.registry,
        )
