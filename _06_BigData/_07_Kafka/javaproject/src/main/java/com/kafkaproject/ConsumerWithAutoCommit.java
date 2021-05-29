package com.kafkaproject;

import java.time.Duration;
import java.util.Arrays;
import java.util.Properties;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;

public class ConsumerWithAutoCommit {

    public static void main(String[] args) {
        
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9093, localhost:9094, localhost:9095");
        props.put("group.id", "first-group"); // This is mandatory if autocommit is set to true
        props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
        props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");

        props.put("enable.auto.commit", "true");
        props.put("auto.commit.interval.ms", "1000");

        String topics[] = {"numbers"};

        KafkaConsumer<String, String> consumer = new KafkaConsumer<String, String>(props);
        consumer.subscribe(Arrays.asList(topics));

        try{
            while (true) {
                ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
                for (ConsumerRecord<String, String> record : records){
                    String message = String.format("offset = %d, key = %s, value = %s, partition= %s%n", record.offset(), record.key(), record.value(), record.partition());
                    System.out.println(message);
                }
            }
        } catch (Exception e){
            e.printStackTrace();
        } finally{
            consumer.close();
        }
    }
}
