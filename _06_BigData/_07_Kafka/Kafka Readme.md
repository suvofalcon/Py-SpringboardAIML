## What is Apache Kafka

![What is Kafka](resources/Kafka-1.png)

YouTube is an example of publish-subscribe messaging system

## Some Basic Commands of Apache Kafka

![Basic Commands of Kafka](resources/Kafka-2.png)

## Apache Kafka Broker

![Kafka Broker](resources/Kafka-3.png)

In every publish subscribe system, message should be stored somewhere. Publishers should be able to send messages and consumers should be able to receive those subscribed messages.

Brokers are responsible for all of these. Publishers in Apache Kafka are called producers and subscribers are called Consumers. There are can multiple Kafka brokers working indepedently. Kafka brokers stores messages in files in hard-drives. There can be multiple producers and multiple consumers as below. 

![Multiple Producers and Consumers](resources/Kafka-4.png)

These Producers and Consumers, produce and receive messages simultaneously. This also makes the Broker single point of failure, if the broker fails then producers and consumers message interaction will not work

Therefore Broker clusters are used.

## Broker Cluster

![Broker Cluster](resources/Kafka-5.png)

Multiple Producers and Multiple Consumers can interact with different brokers inside the Broker cluster

One producer can send messages to multiple brokers and hence each  of the broker can store part of the messages

How does the Broker Synchronize and Communicate amongst themseleves inside a Broker Cluster?

## Zookeeper

![Zookeeper](resources/Kafka-6.png)

There is just a single controller in a kafka cluster

## Zookeeper Cluster (ensemble)

![Zookeeper Cluster](resources/Kafka-7.png)

It is recommended to have odd number of servers in a zookeeper ensemble.
In every zookeeper cluster, we setup something called quorum , which is the minimum number of servers that should be up and running in order to form operational cluster.

![Zookeeper Quorum](resources/Kafka-8.png)

## Multiple Kafka Clusters

![Multiple Kafka Cluster](resources/Kafka-9.png)

Every cluster is a separate entity, however it is possible to have data synchronization between different Kafka Clusters

## Defaiult Kafka and Zookeeper Ports

![Default Ports](resources/Kafka-10.png)

If multiple zookeepers servers are running on a single computer, three different ports are needed and the same cab be specified by adjusting the configuration files

Same relates to Kafka Brokers

If Brokers should be publicly accessible, we need to adjust "advertised.listeners" property in Broker Config.

## Kafka Topic

![Kafka Topic](resources/Kafka-11.png)

Messages are stored by Kafka Brokers by topic. Every topic must have a unique name. Every topic must have a unique name in a kafka cluster.

Every message inside a topic must have a unique number called offset.

