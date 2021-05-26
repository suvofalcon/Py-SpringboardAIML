## Kafka Cluster with Multiple Brokers

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

#### Get Information from zookeeper about specific broker by id
*zookeeper-shell.sh localhost:2181 get /brokers/ids/1*
*zookeeper-shell.sh localhost:2181 get /brokers/ids/2*
*zookeeper-shell.sh localhost:2181 get /brokers/ids/3*

#### Create a topic with replication 1 and partitions 5 in any of the specified brokers
*kafka-topics.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --create --replication-factor 1 --partitions 5 --topic cars*

-- Please note 5 partitions will be created across three brokers

#### Start Console producer
<i>kafka-console-producer.sh --broker-list localhost:9093, localhost:9094, localhost:9095 --topic cars</i>

*kafka-console-producer.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic cars* -- Preferable

#### Start Console Consumer
*kafka-console-consumer.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic cars --from-beginning*

#### Details about topic in the cluster

**list down all topics**
*kafka-topics.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --list*

**Details about specific topic**
*kafka-topics.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --describe --topic cars*

#### Simulating Broker failure in the cluster






