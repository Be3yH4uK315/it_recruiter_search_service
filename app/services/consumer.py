import asyncio
import json
import aio_pika
import logging
from concurrent.futures import ThreadPoolExecutor
from app.core.config import settings
from app.ml_models import indexer

logger = logging.getLogger(__name__)

DLX_NAME = f"{settings.CANDIDATE_EXCHANGE_NAME}.dlx"
DLQ_NAME = f"{settings.CANDIDATE_EXCHANGE_NAME}.dlq"

class RabbitMQConsumer:
    def __init__(self):
        self.connection_string = (
            f"amqp://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PASS}@"
            f"{settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}/"
        )
        self.connection = None
        self.channel = None
        self.task = None
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def check_connection(self):
        try:
            if not self.connection or self.connection.is_closed:
                await self.connect()
            return True
        except Exception as e:
            logger.error(f"RabbitMQ connection check failed: {e}")
            return False

    async def connect(self, retries=5, backoff=2):
        for attempt in range(retries):
            try:
                logger.info(f"Connecting to RabbitMQ (attempt {attempt + 1})...")
                self.connection = await aio_pika.connect_robust(self.connection_string)
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=1)
                logger.info("Successfully connected to RabbitMQ.")
                return
            except Exception as e:
                logger.error(f"Connection failed: {e}. Retrying in {backoff ** attempt} seconds...")
                await asyncio.sleep(backoff ** attempt)
        raise Exception("Failed to connect to RabbitMQ after retries.")

    async def on_message(self, message: aio_pika.IncomingMessage):
        try:
            routing_key = message.routing_key
            body = message.body.decode()
            data = json.loads(body)

            logger.info(f"Received message with routing key '{routing_key}'")

            if routing_key in ["candidate.created", "candidate.updated"]:
                await indexer.index_document(data)
                future = self.executor.submit(indexer.upsert_vector, data)
                await asyncio.wrap_future(future)
            elif routing_key == "candidate.deleted":
                candidate_id = data.get("id")
                if candidate_id:
                    await indexer.delete_document(candidate_id)
                    future = self.executor.submit(indexer.delete_vector, candidate_id)
                    await asyncio.wrap_future(future)
                else:
                    logger.error("Error: 'id' not found in delete message")
            
            await message.ack()
            logger.info(f"Message with routing key '{routing_key}' processed successfully.")

        except Exception as e:
            logger.error(f"Error processing message: {e}. Rejecting and sending to DLQ.")
            await message.reject(requeue=False)

    async def consume(self):
        if not self.connection:
            await self.connect()

        dlx_exchange = await self.channel.declare_exchange(
            DLX_NAME, aio_pika.ExchangeType.TOPIC, durable=True
        )
        dlq_queue = await self.channel.declare_queue(DLQ_NAME, durable=True)
        await dlq_queue.bind(dlx_exchange, routing_key="#")

        exchange = await self.channel.declare_exchange(
            settings.CANDIDATE_EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True
        )

        queue = await self.channel.declare_queue(
            durable=True,
            arguments={
                "x-dead-letter-exchange": DLX_NAME,
                "x-dead-letter-routing-key": "#"
            }
        )

        await queue.bind(exchange, routing_key="candidate.*")

        logger.info("Starting to consume messages with DLQ configured...")
        await queue.consume(self.on_message)

    def start_consuming(self):
        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self.consume())
        logger.info("RabbitMQ consumer task created.")

    async def close(self):
        if self.task and not self.task.done():
            self.task.cancel()
        if self.connection:
            await self.connection.close()
        self.executor.shutdown(wait=True)
        logger.info("RabbitMQ connection closed.")

consumer = RabbitMQConsumer()