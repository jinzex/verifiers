import asyncio
import contextlib
from unittest.mock import Mock

import msgpack
import pytest
import zmq
import zmq.asyncio
from verifiers.v1.configs.env import (
    ElasticPoolConfig,
    RemotePoolConfig,
    StaticPoolConfig,
    pool_serve_kwargs,
)
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.pool import EnvServerPool
from verifiers.v1.serve.types import HealthResponse, InfoResponse


def test_pool_serve_kwargs() -> None:
    assert pool_serve_kwargs(StaticPoolConfig(num_workers=2)) == {
        "max_workers": 2,
        "elastic": False,
    }
    assert pool_serve_kwargs(ElasticPoolConfig(max_workers=3, multiplex=4)) == {
        "max_workers": 3,
        "multiplex": 4,
        "elastic": True,
    }
    addresses = ["tcp://node0:5101", "tcp://node1:5101"]
    assert pool_serve_kwargs(RemotePoolConfig(backend_addresses=addresses)) == {
        "max_workers": 2,
        "backend_addresses": addresses,
        "elastic": False,
    }


def test_local_pool_owns_spawned_workers() -> None:
    pool = EnvServerPool(
        server_kwargs={},
        max_workers=2,
        address="tcp://127.0.0.1:0",
        legacy=False,
    )
    parent_pipe, child_pipe, process = Mock(), Mock(), Mock()
    process.is_alive.return_value = False
    pool._mpctx = Mock()
    pool._mpctx.Pipe.return_value = (parent_pipe, child_pipe)
    pool._mpctx.Process.return_value = process

    try:
        pool._spawn_worker()
        process.start.assert_called_once()
        child_pipe.close.assert_called_once()
        assert pool.workers[0]["process"] is process
        assert pool.workers[0]["pipe"] is parent_pipe
    finally:
        pool._shutdown()
    parent_pipe.close.assert_called_once()
    process.terminate.assert_called_once()


def test_remote_pool_requires_a_backend() -> None:
    with pytest.raises(ValueError, match="backend_addresses must not be empty"):
        EnvServerPool(
            server_kwargs={},
            max_workers=0,
            address="tcp://127.0.0.1:0",
            legacy=False,
            backend_addresses=[],
        )

    addresses = ["tcp://node0:5101", "tcp://node1:5101"]
    with pytest.raises(ValueError, match="max_workers must match"):
        EnvServerPool(
            server_kwargs={},
            max_workers=1,
            address="tcp://127.0.0.1:0",
            legacy=False,
            elastic=False,
            backend_addresses=addresses,
        )
    with pytest.raises(ValueError, match="remote backend pools cannot be elastic"):
        EnvServerPool(
            server_kwargs={},
            max_workers=2,
            address="tcp://127.0.0.1:0",
            legacy=False,
            backend_addresses=addresses,
        )


@pytest.mark.asyncio
async def test_remote_pool_readiness_routing_and_shutdown() -> None:
    health_gate = asyncio.Event()
    release = asyncio.Event()
    received: asyncio.Queue[str] = asyncio.Queue()
    backend_tasks: list[asyncio.Task] = []
    backend_sockets = []
    backend_context = zmq.asyncio.Context()

    async def start_backend(name: str, wait_for_health: bool = False) -> str:
        socket = backend_context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind("tcp://127.0.0.1:0")
        address = socket.getsockopt_string(zmq.LAST_ENDPOINT)

        async def serve() -> None:
            while True:
                client_id, request_id, method, _ = await socket.recv_multipart()
                if method == b"health":
                    if wait_for_health:
                        await health_gate.wait()
                    response = HealthResponse()
                else:
                    await received.put(name)
                    await release.wait()
                    response = InfoResponse(error=name)
                data = msgpack.packb(response.model_dump(mode="python"), use_bin_type=True)
                await socket.send_multipart([client_id, request_id, data])

        backend_sockets.append(socket)
        backend_tasks.append(asyncio.create_task(serve()))
        return address

    addresses = [
        await start_backend("node0"),
        await start_backend("node1", wait_for_health=True),
    ]
    pool = EnvServerPool(
        server_kwargs={},
        max_workers=None,
        address="tcp://127.0.0.1:0",
        legacy=False,
        elastic=False,
        backend_addresses=addresses,
    )
    assert pool.max_workers == 2
    assert pool.elastic is False
    pool_task = asyncio.create_task(pool.run())
    env_client = EnvClient(pool.address)

    try:
        assert not await env_client.health(timeout=0.05)
        health_gate.set()
        await env_client.wait_for_server_startup(timeout=1, interval=0.01)

        requests = [asyncio.create_task(env_client.info()) for _ in range(2)]
        routed = await asyncio.wait_for(
            asyncio.gather(received.get(), received.get()), timeout=1
        )
        assert set(routed) == {"node0", "node1"}
        release.set()
        responses = await asyncio.wait_for(asyncio.gather(*requests), timeout=1)
        assert {response.error for response in responses} == {"node0", "node1"}

        pool_task.cancel()
        await pool_task
        assert all(not task.done() for task in backend_tasks)
    finally:
        release.set()
        if not pool_task.done():
            pool_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pool_task
        await env_client.close()
        for task in backend_tasks:
            task.cancel()
        await asyncio.gather(*backend_tasks, return_exceptions=True)
        for socket in backend_sockets:
            socket.close()
        backend_context.term()
