import asyncio
import json
import aio_pika
import logging
from app.core.config import settings
from app.ml_models import indexer

logger = logging.getLogger(__name__)

class RabbitMQConsumer:
    def __init__(self):
        self.connection_string = (
            f"amqp://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PASS}@"
            f"{settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}/"
        )
        self.connection = None
        self.channel = None
        self.task = None

    async def connect(self):
        logger.info("Connecting to RabbitMQ...")
        self.connection = await aio_pika.connect_robust(self.connection_string)
        self.channel = await self.connection.channel()
        logger.info("Successfully connected to RabbitMQ.")

    async def on_message(self, message: aio_pika.IncomingMessage):
        async with message.process():
            try:
                routing_key = message.routing_key
                body = message.body.decode()
                data = json.loads(body)

                logger.info(f"Received message with routing key '{routing_key}'")

                if routing_key in ["candidate.created", "candidate.updated"]:
                    await indexer.index_document(data)
                    indexer.upsert_vector(data)
                elif routing_key == "candidate.deleted":
                    candidate_id = data.get("id")
                    if candidate_id:
                        await indexer.delete_document(candidate_id)
                        indexer.delete_vector(candidate_id)
                    else:
                        logger.error("Error: 'id' not found in delete message")

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def consume(self):
        if not self.connection:
            await self.connect()

        exchange = await self.channel.declare_exchange(
            settings.CANDIDATE_EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True
        )
        queue = await self.channel.declare_queue(durable=True)
        await queue.bind(exchange, routing_key="candidate.*")

        logger.info("Starting to consume messages...")
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
        logger.info("RabbitMQ connection closed.")

consumer = RabbitMQConsumer()