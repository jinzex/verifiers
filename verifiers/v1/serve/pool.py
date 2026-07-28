"""Env-server pool: a ROUTER broker over N backends.

A lone `EnvServer` runs every rollout as an `asyncio.Task` on one event loop, so
CPU-bound work (renderer tokenization, scoring) competes for that loop. v0 relieved
this with a router + worker pool; this reinstates it for v1.

A broker binds the client-facing ROUTER (the *same* wire protocol as a lone
`EnvServer`, so `EnvClient` is unchanged) and load-balances requests to the least-busy
worker over a `DEALER` per worker. Workers are either local spawned processes bound to
ipc addresses or independently managed remote EnvServers bound to TCP addresses. The
worker's `client_id` (its reply identity) is the broker's DEALER identity; the broker
holds the real client identity in `pending` and routes the reply back by `request_id`.
Remote pool startup waits for every backend; `health` is then answered inline and
everything else goes to a worker.

Local scaling is elastic but upscale-only: a new worker is spawned when in-flight requests
reach 90% of current capacity (`workers * multiplex`). Local workers are spawned
`spawn`-style (own env, own loop) and monitor a death pipe so an orphaned worker
self-exits if the broker dies. TODO: downscale idle workers, per-worker restart-on-death, stats/lag
monitors (v0 had them; omitted here — rollout errors are returned as data, not crashes,
so worker death is rare).
"""

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import threading
import uuid
from collections.abc import Callable

import msgpack
import zmq
import zmq.asyncio

from verifiers.v1.env import EnvConfig
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import HealthResponse, RunGroupRequest

logger = logging.getLogger(__name__)

_HEALTH = msgpack.packb(HealthResponse().model_dump(mode="json"), use_bin_type=True)


def _arm_teardown(death_pipe=None) -> None:
    """Arm a spawned process (serve_env broker/single server, or pool worker) for clean
    teardown: it inherits no signal handlers, so by default SIGTERM kills it abruptly, skipping
    asyncio.run()'s serving() cleanup and orphaning host tunnels (and sandboxes).

    - SIGTERM -> KeyboardInterrupt so the event loop runs its finallys (serve_env swallows it);
    - with `death_pipe`, self-SIGTERM when the parent dies (pipe EOF, even on its SIGKILL) so no
      child is orphaned (main -> serve_env and broker -> worker are both armed this way)."""

    def on_sigterm(*_) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    if death_pipe is None:
        return

    def _watch() -> None:
        with contextlib.suppress(Exception):
            death_pipe.recv()
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_watch, daemon=True).start()


class EnvServerPool:
    """ROUTER broker that dispatches requests to least-busy workers.

    Remote backends are fixed and independently managed. Local pools with `elastic=True`
    (default) start with a single worker and spawn another
    whenever in-flight requests reach 90% of current capacity (`workers * multiplex`), up
    to `max_workers`. Upscale-only for now — workers are never reclaimed. For local pools,
    `elastic=False` pre-spawns all `max_workers` upfront (the old fixed-pool behavior).
    The broker forwards opaque request frames, so workers can be `EnvServer` (v1) or
    `LegacyEnvServer` (v0) without the broker caring."""

    def __init__(
        self,
        server_kwargs: dict,
        max_workers: int | None,
        address: str,
        legacy: bool,
        log_setup: Callable[[], None] | None = None,
        multiplex: int = 128,
        elastic: bool = True,
        backend_addresses: list[str] | None = None,
    ) -> None:
        if backend_addresses is not None:
            if not backend_addresses:
                raise ValueError("backend_addresses must not be empty")
            if max_workers is None:
                max_workers = len(backend_addresses)
            elif max_workers != len(backend_addresses):
                raise ValueError("max_workers must match the number of backend_addresses")
            if elastic:
                raise ValueError("remote backend pools cannot be elastic")
        self.server_kwargs = server_kwargs
        self.max_workers = max_workers
        self.multiplex = multiplex
        self.elastic = elastic
        self.backend_addresses = backend_addresses
        self.legacy = legacy
        self.log_setup = log_setup
        self.session = uuid.uuid4().hex[:12]
        self.workers: list[dict] = []
        self._mpctx = mp.get_context("spawn")
        self._poller: zmq.asyncio.Poller | None = None

        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(address)
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)

    def _worker_path(self, i: int) -> str:
        return f"/tmp/vf-pool-{self.session}-{i}"

    def _add_worker(self, address: str, process=None, pipe=None) -> None:
        dealer = self.ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(address)  # connect before bind is fine — ZMQ queues
        self.workers.append(
            {
                "process": process,
                "dealer": dealer,
                "pipe": pipe,
                "active": 0,
                "index": len(self.workers),
                "address": address,
            }
        )
        if self._poller is not None:
            self._poller.register(dealer, zmq.POLLIN)

    def _spawn_worker(self) -> None:
        i = len(self.workers)  # upscale-only, so the next index is the current count
        address = f"ipc://{self._worker_path(i)}"
        parent_conn, child_conn = self._mpctx.Pipe()
        proc = self._mpctx.Process(
            target=serve_env,
            kwargs=dict(
                max_workers=1,
                address=address,
                death_pipe=child_conn,
                legacy=self.legacy,
                log_setup=self.log_setup,
                **self.server_kwargs,
            ),
            daemon=False,
        )
        proc.start()
        child_conn.close()  # parent keeps the write end (its close signals death)
        self._add_worker(address, process=proc, pipe=parent_conn)

    async def _wait_for_remote_backends(self, timeout: float = 120.0) -> None:
        payload = msgpack.packb({}, use_bin_type=True)

        async def wait(worker: dict) -> None:
            request_id = uuid.uuid4().bytes
            await worker["dealer"].send_multipart([request_id, b"health", payload])
            try:
                response_id, data = await asyncio.wait_for(
                    worker["dealer"].recv_multipart(), timeout=timeout
                )
            except TimeoutError as error:
                raise TimeoutError(
                    f"remote EnvServer did not become healthy: {worker['address']}"
                ) from error
            response = HealthResponse.model_validate(msgpack.unpackb(data, raw=False))
            if response_id != request_id or not response.success:
                raise RuntimeError(
                    f"remote EnvServer is not healthy: {worker['address']}"
                )

        await asyncio.gather(*(wait(worker) for worker in self.workers))

    def _maybe_scale_up(self, in_flight: int) -> None:
        """Spawn one more worker when in-flight rollout slots reach 90% of capacity.

        A new worker starts at `active=0`, so least-busy dispatch funnels the backlog to
        it as it comes online (a few seconds to load the env) — fine, since we only scale
        up once already saturated. `max_workers=None` scales without a cap."""
        if self.max_workers is not None and len(self.workers) >= self.max_workers:
            return
        if in_flight >= 0.9 * len(self.workers) * self.multiplex:
            self._spawn_worker()
            logger.info(
                "EnvServerPool scaled up to %d/%s workers (in_flight=%d)",
                len(self.workers),
                self._cap_str,
                in_flight,
            )

    @property
    def _cap_str(self) -> str:
        return "inf" if self.max_workers is None else str(self.max_workers)

    async def run(self) -> None:
        self._poller = zmq.asyncio.Poller()
        try:
            if self.backend_addresses is not None:
                for address in self.backend_addresses:
                    self._add_worker(address)
                await self._wait_for_remote_backends()
            else:
                # Elastic: start with one and scale up on demand. Otherwise pre-spawn the lot
                # (`max_workers` is a concrete count when elastic is off).
                for _ in range(1 if self.elastic else (self.max_workers or 1)):
                    self._spawn_worker()
        except BaseException:
            self._shutdown()
            raise
        self._poller.register(self.frontend, zmq.POLLIN)
        # request_id -> {client_id, worker, rollout_slots}
        pending: dict[bytes, dict] = {}
        logger.info(
            "EnvServerPool up: address=%s workers=%d/%s multiplex=%d elastic=%s",
            self.address,
            len(self.workers),
            self._cap_str,
            self.multiplex,
            self.elastic,
        )
        if self.backend_addresses is not None:
            logger.info("EnvServerPool remote backends: %s", self.backend_addresses)
        try:
            in_flight = 0
            while True:
                events = dict(await self._poller.poll())
                if self.frontend in events:
                    (
                        client_id,
                        request_id,
                        method,
                        payload,
                    ) = await self.frontend.recv_multipart()
                    if method == b"health":
                        await self.frontend.send_multipart(
                            [client_id, request_id, _HEALTH]
                        )
                    else:
                        # Pool capacity is measured in rollouts; one group request carries n.
                        rollout_slots = 1
                        if method == b"run_group":
                            with contextlib.suppress(Exception):
                                request = RunGroupRequest.model_validate(
                                    msgpack.unpackb(payload, raw=False)
                                )
                                rollout_slots = max(1, request.n)
                        worker = min(self.workers, key=lambda w: w["active"])
                        worker["active"] += rollout_slots
                        pending[request_id] = {
                            "client_id": client_id,
                            "worker": worker,
                            "rollout_slots": rollout_slots,
                        }
                        in_flight += rollout_slots
                        # forward without client_id — the DEALER identity is the worker's
                        # `client_id`; we hold the real one in `pending`.
                        await worker["dealer"].send_multipart(
                            [request_id, method, payload]
                        )
                        if self.elastic:
                            self._maybe_scale_up(in_flight)
                for w in self.workers:
                    if w["dealer"] in events:
                        request_id, data = await w["dealer"].recv_multipart(copy=False)
                        # Copy only the routing key; relay the response Frames unchanged.
                        entry = pending.pop(request_id.bytes, None)
                        if entry is None:
                            continue
                        entry["worker"]["active"] -= entry["rollout_slots"]
                        in_flight -= entry["rollout_slots"]
                        with contextlib.suppress(zmq.ZMQError):
                            await self.frontend.send_multipart(
                                [entry["client_id"], request_id, data], copy=False
                            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        for w in self.workers:
            if w["pipe"] is not None:
                with contextlib.suppress(Exception):
                    w["pipe"].close()
            if w["process"] is not None:
                with contextlib.suppress(Exception):
                    w["process"].terminate()
        for w in self.workers:
            if w["process"] is not None:
                with contextlib.suppress(Exception):
                    w["process"].join(timeout=10)
                if w["process"].is_alive():
                    with contextlib.suppress(Exception):
                        w["process"].kill()
            with contextlib.suppress(Exception):
                w["dealer"].close()
            if w["process"] is not None:
                with contextlib.suppress(OSError):
                    os.unlink(self._worker_path(w["index"]))
        self.frontend.close()
        self.ctx.term()
        logger.info("EnvServerPool down")


def env_config_data(env: EnvConfig) -> dict:
    """A picklable dict of a (possibly dynamically-narrowed, unpicklable) env config —
    ship this across a process boundary, then rebuild via `resolve_env_config`
    (re-narrowing to the env's concrete config class)."""
    return env.model_dump(mode="json")


def serve_env(
    *,
    max_workers: int | None,
    legacy: bool = False,
    address: str = "tcp://127.0.0.1:5000",
    address_queue=None,
    death_pipe=None,
    log_setup: Callable[[], None] | None = None,
    multiplex: int = 128,
    elastic: bool = True,
    backend_addresses: list[str] | None = None,
    **server_kwargs,
) -> None:
    """Serve one env over ZMQ, directly or through an `EnvServerPool` broker.
    Without `backend_addresses`, `max_workers <= 1` uses a single in-process `EnvServer`;
    larger values use local worker processes (`None` = unbounded). The frontend protocol
    is identical. Reports the bound address on `address_queue` (for a spawner that passed
    an OS-assigned `:0`).

    `backend_addresses` always selects a broker over fixed, independently managed remote
    EnvServers, including when only one backend is configured. Otherwise, `elastic`
    (default True) starts the pool at one worker and scales up to `max_workers` as load
    grows; `multiplex` is the per-worker capacity for the scale-up trigger.
    `elastic=False` pre-spawns all `max_workers`.

    A native env config may be passed as `config` (an object) or `config_data` (the
    picklable dict from `env_config_data`, for callers that spawn this function and so
    can't pickle a dynamically-narrowed config type); legacy passes `env_id`/`env_args`/
    `extra_env_kwargs`.

    `log_setup` (a picklable callable) configures logging for this process and every
    spawned worker — without it a spawned server inherits no handlers and its INFO logs
    (rollout start/done, the pool line) are silently dropped.

    `death_pipe` (when spawned by a parent, e.g. the eval main process) makes this server
    self-terminate if that parent dies abruptly — see `_arm_teardown`."""
    # Graceful SIGTERM (run asyncio teardown) + self-terminate if the parent dies. The
    # re-raised KeyboardInterrupt is swallowed below for a clean exit (no spurious traceback).
    _arm_teardown(death_pipe)
    if log_setup is not None:
        log_setup()
    try:
        if backend_addresses is not None or max_workers is None or max_workers > 1:
            if backend_addresses is None and (
                "config" in server_kwargs
            ):  # dict-ify for the workers (config_data is picklable)
                server_kwargs = {
                    "config_data": env_config_data(server_kwargs["config"])
                }
            pool = EnvServerPool(
                server_kwargs,
                max_workers,
                address,
                legacy,
                log_setup,
                multiplex,
                elastic,
                backend_addresses,
            )
            if address_queue is not None:
                address_queue.put(pool.address)
            asyncio.run(pool.run())
        else:
            from verifiers.v1.legacy import LegacyEnvServer

            if (
                "config_data" in server_kwargs
            ):  # rebuild the env config for an in-process server
                from verifiers.v1.loaders import resolve_env_config

                server_kwargs = {
                    "config": resolve_env_config(server_kwargs["config_data"])
                }
            cls = LegacyEnvServer if legacy else EnvServer
            cls.run_server(
                address=address, address_queue=address_queue, **server_kwargs
            )
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Env server failed")
        raise
