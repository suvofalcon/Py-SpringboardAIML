
# Import Kafka Consumer
from kafka import KafkaConsumer

# Initialize COnsumer
consumer = KafkaConsumer(
    'names', 
    bootstrap_servers=['localhost:9093', 'localhost:9094', 'localhost:9094'],
    group_id = 'names-consumer-group'
)

for message in consumer:
    print(message)