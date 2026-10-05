"""Check a shipped wheel's real coordinator and parent-pipe cleanup, without Jesse."""

import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading


def main() -> None:
    """Require authenticated readiness and clean exit from the installed Rust extension."""
    token = secrets.token_hex(32)
    process = subprocess.Popen(
        [sys.executable, '-u', '-c', 'import jesse_rust; jesse_rust.run_coordinator()'],
        env={**os.environ, 'JESSE_COORDINATION_TOKEN': token,
             'JESSE_COORDINATION_BIND': '127.0.0.1:0', 'JESSE_COORDINATION_PARENT_PIPE': '1'},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert process.stdout is not None and process.stdin is not None
    lines: queue.Queue[str] = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True).start()
    try:
        # A cold interpreter import must not hang a packaging CI job indefinitely.
        line = lines.get(timeout=20).strip()
        assert line.startswith('jesse-coordinator listening on 127.0.0.1:'), 'Packaged coordinator failed to start'
        port = int(line.rsplit(':', 1)[1])
        with socket.create_connection(('127.0.0.1', port), timeout=3) as connection:
            with connection.makefile('rb') as reader:
                request = {'version': 2, 'token': token, 'namespace': 'wheel-check',
                           'op': 'hello', 'payload_bytes': 0}
                connection.sendall(json.dumps(request).encode() + b'\n')
                assert json.loads(reader.readline()) == {'ok': True, 'value': 'ready'}
                connection.sendall(b'{}\n')
                reply = json.loads(reader.readline())
                assert reply['ok'] and reply['value']['protocol'] == 2
        # This is the same ownership pipe used by Jesse, including after a parent crash.
        process.stdin.close()
        assert process.wait(timeout=5) == 0, 'Coordinator did not exit after its owner closed'
        print('Packaged coordinator: authenticated readiness and parent-pipe cleanup passed')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


if __name__ == '__main__':
    main()
