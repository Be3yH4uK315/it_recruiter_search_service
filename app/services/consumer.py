# app/services/consumer.py

import asyncio
import json
import aio_pika
from app.core.config import (
    RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USER, RABBITMQ_PASS,
    CANDIDATE_EXCHANGE_NAME
)
from app.services.indexer import indexer


class RabbitMQConsumer:
    def __init__(self):
        self.connection_string = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"
        self.connection = None
        self.channel = None
        self.task = None

    async def connect(self):
        print("Connecting to RabbitMQ...")
        self.connection = await aio_pika.connect_robust(self.connection_string)
        self.channel = await self.connection.channel()
        print("Successfully connected to RabbitMQ.")

    async def on_message(self, message: aio_pika.IncomingMessage):
        async with message.process():
            try:
                routing_key = message.routing_key
                body = message.body.decode()
                data = json.loads(body)

                print(f"Received message with routing key '{routing_key}'")

                if routing_key in ["candidate.created", "candidate.updated"]:
                    indexer.index_document(data)
                elif routing_key == "candidate.deleted":
                    candidate_id = data.get("id")
                    if candidate_id:
                        indexer.delete_document(candidate_id)
                    else:
                        print("Error: 'id' not found in delete message")

            except Exception as e:
                print(f"Error processing message: {e}")

    async def consume(self):
        if not self.connection:
            await self.connect()

        exchange = await self.channel.declare_exchange(
            CANDIDATE_EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True
        )

        queue = await self.channel.declare_queue(durable=True)

        await queue.bind(exchange, routing_key="candidate.*")

        print("Starting to consume messages...")
        await queue.consume(self.on_message)

    def start_consuming(self):
        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self.consume())
        print("RabbitMQ consumer task created.")

    async def close(self):
        if self.task and not self.task.done():
            self.task.cancel()
        if self.connection:
            await self.connection.close()
        print("RabbitMQ connection closed.")


consumer = RabbitMQConsumer()