## Kafka Consumer Groups

#### Start the zookeeper
<i>zookeeper-server-start.sh $KAFKA_CONFIGS/zookeeper.properties</i>

#### Start Kafka Broker (For simplicity, we will just have a single broker)
<i>kafka-server-start.sh $KAFKA_CONFIGS/server.properties</i>

#### Create a Kafka Topic
<i>kafka-topics.sh --bootstrap-server localhost:9092 --create --replication-factor 1 --partitions 3 --topic numbers</i>

**list down all topics**
*kafka-topics.sh --bootstrap-server localhost:9092 --list*

#### Start the Console Producer
<i>kafka-console-producer.sh --broker-list localhost:9092 --topic numbers</i>

#### Start Console Consumer and consume messages from beginning
<i>kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic numbers --from-beginning</i>

#### List all consumer groups
<i>kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list</i>

#### Consumer group details
*kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group console-consumer-39908 --describe*

#### Start Console consumer with specific consumer group
*kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic numbers --group numbers-group --from-beginning*

#### Start Console consumer and read from specific partition
*kafka-console-consumer.sh --bootstrap-server localhost:9092 --partition 2 --from-beginning --topic numbers*
