
# Get the Kafka Producer
import time
from kafka import KafkaProducer
from faker import Faker

# We will use Faker package to send some random names
fake = Faker()

# Initialize Producer
producer = KafkaProducer(bootstrap_servers=['localhost:9093', 'localhost:9094', 'localhost:9094'])

for _ in range(100):
    # Producer sends message
    name = fake.name()
    producer.send('names', name.encode('utf-8'))
    print(name)
    
# We will put a delay, beacause otherwise process will get killed, before the message gets send
time.sleep(20)
