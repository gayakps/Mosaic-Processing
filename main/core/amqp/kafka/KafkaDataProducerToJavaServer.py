from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: x.encode('utf-8'))

producer.send('testTopic', value='Hello, Kafka!')
producer.flush()
producer.close()