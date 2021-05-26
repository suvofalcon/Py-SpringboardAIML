## We will create topic with multiple partitions

#### Start Zookeeper
<i>zookeeper-server-start.sh $KAFKA_CONFIGS/zookeeper.properties</i>

#### Start Kafka Broker
<i>kafka-server-start.sh $KAFKA_CONFIGS/server.properties</i>

#### Create a Kafka Topic
<i>afka-topics.sh --bootstrap-server localhost:9092 --create --replication-factor 1 --partitions 3 --topic animals</i>

#### Start the Console Producer
<i>kafka-console-producer.sh --broker-list localhost:9092 --topic animals</i>

#### Start Console Consumer and consume messages from beginning
<i>kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic animals --from-beginning</i>

Since consumers are reading messages from multiple partitions .. hence the messages are appearing in different order

#### Start Console Consumer and consume messages from beginning from specific partition
<i>kafka-console-consumer.sh --bootstrap-server localhost:9092 --partition 2 --topic animals --from-beginning</i>

#### Reading messages from specific offset in specific partition
<i>kafka-console-consumer.sh --bootstrap-server localhost:9092 --partition 2 --topic animals --offset 0</i>

#### Reading messages from specific offset from all partitions
<i>This cannot be done, as partiton is required when specifying offset</i>

#### Reading details about topic and _consumer_offsets topic
<i>kafka-topics.sh --bootstrap-server localhost:9092 --list</i>  -- Will give us the system topic and the user's topic
<i>afka-topics.sh --bootstrap-server localhost:9092 --describe --topic animals</i> -- Details about a topic