
# Get the Kafka Producer
import time
from kafka import KafkaProducer

# Initialize Producer
producer = KafkaProducer(bootstrap_servers=['localhost:9093', 'localhost:9094', 'localhost:9094'])

# Producer sends message
producer.send('test', b'Hello from the Python Producer')

# We will put a delay, beacause otherwise process will get killed, before the message gets send
time.sleep(20)
