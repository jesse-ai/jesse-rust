# Jesse coordinator — bundled native backend

This is the Rust service on this development branch for the shared state and notifications Jesse uses.
It uses a versioned local protocol with JSON control frames, raw binary payloads, and a Python standard-library client.
It does not implement Redis, RESP, Lua, clustering, or persistence.

Existing Jesse Live releases keep importing the same core helper module. Those
helpers use only the native coordinator; neither `redis` nor `aioredis` is a
Jesse dependency, and there is no Redis backend or packaging extra. Optuna may
import an already installed Redis client for its optional journal backend; Jesse's SQLite/in-memory studies work without it.
Upgrading does not uninstall Redis packages already present in an environment.

## Automatic startup

The coordinator is compiled into the `jesse-rust` Python extension alongside the
indicators. Install Jesse normally and use `jesse run`; no coordinator settings,
separate executable installation or runtime compiler are needed. Imports, help
and version commands do not start services.

Jesse owns a dedicated child invoking `jesse_rust.run_coordinator()`. The service
binds an OS-assigned loopback port and authenticates with a generated token and
namespace. A successful protocol handshake pins its generation; workers inherit
that endpoint privately through their process environment. Independent Jesse
instances each own their own service. Legacy `REDIS_*` settings and the removed
`COORDINATION_BACKEND=redis` selector can remain in old project files harmlessly.

Diagnostics are drained to the project's `storage/logs/coordinator.log`, rotated
at 1 MiB. Startup readiness is bounded to twenty seconds. On ordinary shutdown,
Jesse drains/cancels its jobs before closing the ownership pipe. Closing stdin or
losing the parent exits the child on Windows and Unix. An unexpected child exit
stops the application with a failure status; it never silently replaces shared
state beneath running trading workers. A worker confirms a failed RPC through a
fresh handshake before treating service loss as cancellation. A refused connection
or changed generation stops it immediately; otherwise repeated inconclusive
failures over six seconds stop it, while explicit busy replies remain retryable.
Cancellation respects Jesse Live's persistency settings; it does not promise
exchange position closure or full live-trading failover.

## Development and external services

Run `./build-local.sh` from the parent Rust repository with
`JESSE_BUILD_PYTHON` set to your development interpreter. Existing maturin and
NumPy are required; the script does not upgrade dependencies. To build the
standalone diagnostic service, use `cargo build --release --locked` in this
folder. Normal distribution uses the compiled Python extension instead.

Advanced/test deployments may still set `COORDINATION_HOST`,
`COORDINATION_PORT`, `COORDINATION_NAMESPACE`, and `COORDINATION_TOKEN` in their
project environment. Such services are externally owned: start them yourself,
continuously drain their diagnostics, and stop all Jesse workers before replacing
them. Jesse does not stop externally managed services. Tokens require at least
32 bytes and are passed through the environment, never command-line arguments.

## Behavior and failure boundaries

- Byte values, sets, increments, expiration, expiring owner-checked locks, and
  prefix subscriptions cover the currently inspected core and live call sites.
- Each client keeps up to eight process-local RPC connections. Warm connections use one request round trip; first-use authentication precedes the body. Idle cached sockets retire lazily on next use after ten seconds, ahead of the server’s thirty-second idle deadline. EOF is checked before reuse. Spawned Jesse workers create fresh clients; using an inherited client after fork fails explicitly. Mutations are never retried after ambiguous network failures.
- Locks expire even while their body is running, as with the current usage.
  Callers must choose leases long enough for the protected operation. Releasing
  an expired lock cannot release its successor. There is no lease renewal yet.
- State is in memory. Restart loses caches, chart snapshots, counters and worker
  memberships; this is not a replacement for Redis persistence. PostgreSQL data
  is unaffected. Do not restart the coordinator under live trading sessions.
- Clients pin the server generation. The parent passes it to spawned Jesse
  workers. Old clients reject a new generation rather than silently writing to
  an empty service; restart the complete Jesse instance after a service restart.
  This does not constitute full live-trading failover or stop exchange activity.
- Subscriptions provide ordered, live-only delivery. They do not replay missed
  messages. A slow subscriber is disconnected rather than silently skipping a
  full queue. Jesse's existing listener reconnects after transient failures.
- On a new connection a 4 KiB authentication header is acknowledged before the client sends its bounded payload. Protocol version 2 is required; older prototype binaries/clients must be replaced together.
- Prefix matching supports the current `APP_PORT:channel:*` subscription only;
  it is not Redis glob matching. Namespaces isolate keys and channel delivery.
  Namespaces are not a security boundary between clients sharing a token.
- The listener is loopback-only and authenticated. This is not a remote-access
  service and provides no TLS. Protect the token and the project env file.

## Resource limits

The implementation caps JSON metadata frames at 16 MiB and binary payloads at 8 MiB, state accounting at 128 MiB,
keys at 100,000, connections at 256, event payloads at 8 MiB, and globally retained shared event frames at 64 MiB. Each subscriber can retain at most 16 MiB. A whole-frame two-second write deadline bounds slow-consumer retention. Queues are bounded by bytes, including a per-event overhead allowance, rather than a small event count. Subscriptions are capped at the smaller of 32 or half the adaptive connection cap, leaving capacity for RPCs. On supported 64-bit macOS/Linux targets the process descriptor limit lowers the connection cap, reserving sixteen descriptors for runtime use. Startup refuses fewer than four available client slots. The subscription reserve prevents subscriptions from exhausting RPC capacity; idle RPC pools can still delay new subscriptions until their idle deadline. A spare descriptor also provides best-effort busy rejection after unexpected Unix accept failures; accept errors are rate-limited in logs. The 256-connection cap accommodates high-core-count workers retaining idle pools; a fully occupied server still rejects new connections until sockets close or their 30-second idle deadline elapses. Values and events are transported as raw bytes; small set members still use hexadecimal strings. Synchronous calls can wait for a pool slot; asyncio callers should use the async publishing facade. State byte accounting includes conservative entry/member overhead and is not an RSS guarantee.
Busy rejection allows only 20 ms to read an authentication header and 20 ms to write a reply, keeping the accept loop responsive. Under overload a delayed header can therefore produce a generic connection error instead of a retryable busy reply. No request is executed on that path.

Connection or aggregate event-queue saturation is reported as a transient connection error (`busy:`). Requests retry only explicit busy rejections up to three attempts (10/20 ms backoff); a lock waiter may continue within its requested wait; ambiguous mutation failures are never retried. Explicit capacity/size errors propagate from `sync_publish`; network outages retain the existing logged-error behavior. Capacity errors do not silently evict shared state. Expired entries are swept
every second and removed on access. Snapshot/event limits need workload review
before a production release, particularly with large strategy chart payloads.

## Validation and scope

Run `python -m pytest tests/test_coordination.py` from the sibling
Jesse core repository, with this Rust repository alongside it (or set
`JESSE_COORDINATOR_SOURCE` to this directory). Tests build cached Cargo dependencies offline and start
their own local services on ephemeral ports. `cargo test --locked` also checks queue ownership and resource accounting. They never contact an existing
Redis or PostgreSQL instance. Set `JESSE_COORDINATION_REQUIRE_TESTS=1` to fail rather than skip when Cargo or its offline cache is unavailable (required in dedicated native CI).

The development lab uses a separate Conda environment, source worktrees, project,
and PostgreSQL cluster. No exchange credentials are copied into that lab.

The package CI checks wheel startup/parent-pipe cleanup on its supported targets.
Before a production release, complete the full supported-platform application
matrix and controlled long-running paper/live validation. Packaged component
checks alone do not certify live-trading reliability.

## License

The coordinator is distributed under the repository's MIT license. Preserve the
licenses of the resolved Cargo dependencies in packaged releases. Nothing here
copies Redis server implementation code.
