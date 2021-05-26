## Kafka Performance Test

#### Start the zookeeper
<i>zookeeper-server-start.sh $KAFKA_CONFIGS/zookeeper.properties</i>

#### Start the three brokers with the three configuration files we created
*kafka-server-start.sh $KAFKA_CONFIGS/server1.properties*
*kafka-server-start.sh $KAFKA_CONFIGS/server2.properties* 
*kafka-server-start.sh $KAFKA_CONFIGS/server3.properties* 

#### Create a Kafka Topic
<i>kafka-topics.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --create --replication-factor 3 --partitions 100 --topic perf</i>

#### Delete a particular topic
*kafka-topics.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic perf --delete*


#### We will create the consumer for this topic
*kafka-console-consumer.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic cars --from-beginning*

#### Start Kafka Producer Performance Test
*kafka-producer-perf-test.sh --topic perf2 --num-records 1000 --throughput 100 --record-size 1000 --producer-props acks=all bootstrap.servers=localhost:9093,localhost:9094,localhost:9095*

#### Start Kafka Consumer Performance Test
*kafka-consumer-perf-test.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic perf --messages 1000000*

#### Simulating lag

1. Create a new topic
kafka-topics.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --create --replication-factor 3 --partitions 3 --topic perf2

2. Start Two Console Consumer with Specific Consumer group
*kafka-console-consumer.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic perf2 --group perf --from-beginning*
*kafka-console-consumer.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --topic perf2 --group perf --from-beginning*

3. Read Details about this consumer group
*kafka-consumer-groups.sh --bootstrap-server localhost:9093, localhost:9094, localhost:9095 --group perf --describe*

4. Start a console producer and produce messages at some rate
*kafka-producer-perf-test.sh --topic perf2 --num-records 1000 --throughput 10 --record-size 100000 --producer-props acks=all bootstrap.servers=localhost:9093,localhost:9094,localhost:9095*

