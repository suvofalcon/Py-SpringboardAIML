## Multiple Brokers and Topic with Replication

Every broker on the same computer must have 

1. Unique port
2. Unique broker ID
3. Unique Log directory

#### Start the zookeeper
<i>zookeeper-server-start.sh $KAFKA_CONFIGS/zookeeper.properties</i>

#### Start the three brokers with the three configuration files we created
*kafka-server-start.sh $KAFKA_CONFIGS/server1.properties*
*kafka-server-start.sh $KAFKA_CONFIGS/server2.properties* 
*kafka-server-start.sh $KAFKA_CONFIGS/server3.properties* 

#### Get information from zookeeper about active broker ids
*zookeeper-shell.sh localhost:2181 ls /brokers/ids*

#### Create a new topic with replication factor 3
kafka-topics.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --create --replication-factor 3 --partitions 7 --topic months

The total quantity of partitions in the topic in this setup wil  be number of partitions X replication factor (7 X 3 = 21)

**list down all topics**
*kafka-topics.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --list*

Note - __consumer_offsets topic is created after the first connection to the cluster and after we consume the first message from the cluster

**Details about specific topic**
*kafka-topics.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --describe --topic cars*

#### Start Console producer
<i>kafka-console-producer.sh --broker-list localhost:9093, localhost:9094, localhost:9095 --topic cars</i>

#### Start Console Consumer
*kafka-console-consumer.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic cars --from-beginning*

#### Start Consumer and read messages from specific partition
*kafka-console-consumer.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic months --partition 3 --from-beginning*

#### Start Consumer and read messages from specific partition and specific offset
*kafka-console-consumer.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic months --partition 2 --offset 1*
*kafka-console-consumer.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic months --partition 1 --offset 2*


