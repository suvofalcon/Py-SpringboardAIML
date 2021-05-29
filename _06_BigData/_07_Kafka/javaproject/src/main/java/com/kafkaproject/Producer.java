package com.kafkaproject;

import java.util.Properties;
import java.util.Date;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;

public class Producer {
    
    public static void main(String[] args) {
        
        String clientId = "my-producer";

        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9093, localhost:9094, localhost:9095");
        props.put("acks", "all");
        props.put("client-id", clientId);

        /*
        Every Message in Kafka is sent as a sequence of bytes and they are stored in the brokers as sequence of bytes.
        That is why we need serializers on Producers and desrializers on Consumers in order to correctly encode/decode 
        message keys and values (every message consists of values and optional key)
        */
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

        // Depending on the data types we have specified in the KafkaProducer intializer, we need to specify similar Serializers
        KafkaProducer<String, String> producer = new KafkaProducer<String, String>(props);
        
        int numOfRecords = 100;
        String topic = "numbers";

        //EXAMPLE -1 (Numbers as Strings for key and value without any delay)
        for (int i = 0; i < numOfRecords; i++) {
            System.out.println("Message "+ i +" was just sent at "+ new Date());
            producer.send(new ProducerRecord<String,String>(topic, Integer.toString(i), Integer.toString(i)));
        }
        producer.close();

        //EXAMPLE - 2 (Formatted string as message and messages are sent with 300 ms delay - 3 messages/second)
        // try{
        //     for (int index = 0; index < numOfRecords; index++) {
        //         String message = String.format("Producer %s has sent message %s at %s", clientId, index, new Date());
        //         System.out.println(message);
        //         producer.send(new ProducerRecord<String,String>(topic, Integer.toString(index), message));
        //         // Messages will be sent at 300 ms delay
        //         Thread.sleep(300);
        //     }
        // }catch(Exception e){
        //     e.printStackTrace();
        // } finally{
        //     producer.close(); // we need to close the producer
        // }
    }
}
