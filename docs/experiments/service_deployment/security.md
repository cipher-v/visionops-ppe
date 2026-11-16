# Upload and operational security

Client filenames never become paths. The service generates UUID-based names in a
controlled runtime directory, checks an allow-listed extension, enforces a byte limit,
and then verifies decodability with OpenCV. Zone YAML is capped at 1 MiB and parsed by the
existing strict zone schema. Temporary video input is deleted after processing; completed
artifacts have documented local retention.

Logs contain event, route/job type, status and latency, but not upload bytes, frames,
credentials, filenames as metric labels, or arbitrary request secrets. API responses do
not expose model/config filesystem paths. This reference service has no authentication,
TLS, malware scanning, distributed rate limiting or durable queue; those belong at a
deployment gateway/platform boundary before untrusted public exposure.

Drift is explicitly diagnostic-only and is never allowed to stop inference, open an
incident, invalidate a prediction, retrain a model or page an operator.

