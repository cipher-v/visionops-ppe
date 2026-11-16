# Service design

FastAPI owns one `ProductionRuntime` through application lifespan. Startup strictly loads
the frozen detector manifest/config, verifies the weight SHA-256, validates tracker,
association and temporal configs, optionally loads the drift reference, creates one
PyTorch `DetectorBackend`, and performs one warmup. Startup fails rather than serving an
unready or mismatched model.

Image inference is stateless and runs through a bounded GPU semaphore. Video uploads enter
an in-process `ThreadPoolExecutor` limited to one worker; each job uses the same loaded
detector and the existing tracking, association, temporal and optional zone pipeline.
This is a single-node reference service, not a distributed scheduler. Completed structured
artifacts remain in the configured runtime directory until externally cleaned.

