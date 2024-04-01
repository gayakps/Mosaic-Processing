import pika
import json

from core.config import option

# RabbitMQ 서버에 연결
connection = pika.BlockingConnection(pika.ConnectionParameters(option.host))
channel = connection.channel()
# 메시지를 전송할 큐 선언
channel.queue_declare(queue=option.queue_name, durable=True)
# 전송할 메시지. 예를 들어, 비즈니스 로직에 필요한 데이터를 JSON 형태로 전송할 수 있습니다.
message = json.dumps({'type': 'process_data', 'data': 'This is a sample data'})

send_message_amount = 0


def sendToJavaServer(json_message):
    global send_message_amount

    send_message_amount = send_message_amount + 1
    # 전송할 메시지. 예를 들어, 비즈니스 로직에 필요한 데이터를 JSON 형태로 전송할 수 있습니다.
    print(f'Sending message: {json_message} ( Amount : {send_message_amount} )')
    channel.basic_publish(exchange='',
                          routing_key=f'{option.queue_name}',
                          body=json_message)

