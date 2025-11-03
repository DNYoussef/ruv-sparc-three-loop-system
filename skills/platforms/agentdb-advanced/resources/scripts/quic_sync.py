#!/usr/bin/env python3
"""
QUIC Synchronization Manager for AgentDB
Enables sub-millisecond latency synchronization between AgentDB instances
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from aioquic.asyncio import QuicConnectionProtocol, connect, serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived, ProtocolNegotiated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QUICConfig:
    """QUIC synchronization configuration"""
    host: str = "0.0.0.0"
    port: int = 4433
    peers: List[str] = None
    sync_interval: int = 1000  # ms
    batch_size: int = 100
    max_retries: int = 3
    compression: bool = True
    cert_file: str = "cert.pem"
    key_file: str = "key.pem"


class QUICAgentDBProtocol(QuicConnectionProtocol):
    """QUIC protocol handler for AgentDB synchronization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_patterns = []
        self.stream_map = {}

    def quic_event_received(self, event):
        """Handle QUIC events"""
        if isinstance(event, ProtocolNegotiated):
            logger.info(f"QUIC protocol negotiated: {event.alpn_protocol}")

        elif isinstance(event, StreamDataReceived):
            # Decode incoming pattern data
            data = event.data.decode('utf-8')
            try:
                pattern = json.loads(data)
                logger.info(f"Received pattern sync: {pattern.get('id', 'unknown')}")
                # Process pattern insertion
                asyncio.create_task(self.process_pattern(pattern))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode pattern data: {e}")

    async def process_pattern(self, pattern: Dict):
        """Process incoming pattern synchronization"""
        # Insert pattern into local AgentDB
        # This would be integrated with your AgentDB adapter
        logger.info(f"Processing pattern: {pattern.get('type', 'unknown')}")


class QUICSyncManager:
    """Manages QUIC synchronization between AgentDB nodes"""

    def __init__(self, config: QUICConfig):
        self.config = config
        self.connections = {}
        self.sync_queue = asyncio.Queue()
        self.running = False

    async def start_server(self):
        """Start QUIC server for receiving synchronization"""
        configuration = QuicConfiguration(
            is_client=False,
            alpn_protocols=["agentdb-sync-v1"],
        )
        configuration.load_cert_chain(
            self.config.cert_file,
            self.config.key_file
        )

        logger.info(f"Starting QUIC server on {self.config.host}:{self.config.port}")

        await serve(
            self.config.host,
            self.config.port,
            configuration=configuration,
            create_protocol=QUICAgentDBProtocol,
        )

    async def connect_to_peer(self, peer: str):
        """Establish QUIC connection to a peer"""
        host, port = peer.split(':')
        port = int(port)

        configuration = QuicConfiguration(
            is_client=True,
            alpn_protocols=["agentdb-sync-v1"],
        )
        configuration.verify_mode = False  # For dev; use proper certs in production

        logger.info(f"Connecting to peer {peer}")

        async with connect(
            host,
            port,
            configuration=configuration,
            create_protocol=QUICAgentDBProtocol,
        ) as protocol:
            self.connections[peer] = protocol
            logger.info(f"Connected to peer {peer}")

    async def broadcast_pattern(self, pattern: Dict):
        """Broadcast pattern to all connected peers"""
        pattern_data = json.dumps(pattern).encode('utf-8')

        for peer, protocol in self.connections.items():
            try:
                stream_id = protocol._quic.get_next_available_stream_id()
                protocol._quic.send_stream_data(stream_id, pattern_data, end_stream=True)
                protocol.transmit()
                logger.info(f"Broadcasted pattern to {peer}")
            except Exception as e:
                logger.error(f"Failed to broadcast to {peer}: {e}")

    async def sync_loop(self):
        """Main synchronization loop"""
        self.running = True
        logger.info("Starting synchronization loop")

        while self.running:
            try:
                # Batch patterns for efficient synchronization
                patterns = []
                for _ in range(self.config.batch_size):
                    try:
                        pattern = await asyncio.wait_for(
                            self.sync_queue.get(),
                            timeout=self.config.sync_interval / 1000
                        )
                        patterns.append(pattern)
                    except asyncio.TimeoutError:
                        break

                # Broadcast batched patterns
                if patterns:
                    for pattern in patterns:
                        await self.broadcast_pattern(pattern)

            except Exception as e:
                logger.error(f"Sync loop error: {e}")

            await asyncio.sleep(self.config.sync_interval / 1000)

    async def queue_pattern(self, pattern: Dict):
        """Queue pattern for synchronization"""
        await self.sync_queue.put(pattern)

    async def start(self):
        """Start QUIC synchronization manager"""
        # Start server
        server_task = asyncio.create_task(self.start_server())

        # Connect to peers
        if self.config.peers:
            for peer in self.config.peers:
                await self.connect_to_peer(peer)

        # Start sync loop
        sync_task = asyncio.create_task(self.sync_loop())

        logger.info("QUIC Sync Manager started")

        # Keep running
        await asyncio.gather(server_task, sync_task)

    def stop(self):
        """Stop synchronization"""
        self.running = False
        logger.info("QUIC Sync Manager stopped")


async def main():
    """Example usage"""
    config = QUICConfig(
        host="0.0.0.0",
        port=4433,
        peers=["192.168.1.10:4433", "192.168.1.11:4433"],
        sync_interval=1000,
        batch_size=100
    )

    manager = QUICSyncManager(config)
    await manager.start()


if __name__ == "__main__":
    asyncio.run(main())
