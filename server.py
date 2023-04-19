import socket
import threading

HOST = '127.0.0.1'
PORT = 500
LISTENER_LIMIT = 5
active_clients = []

# Function to listen for upcoming messages from a client
def listen_for_messages(client, username):

    while 1:

        message = client.recv(2048).decode('utf-8')
        if message != '':

            final_msg = username + '~' + message
            send_message_to_all(final_msg)
        else:
            print(f"The message send from client {username} is empty")


# Function to send message to a single client
def send_message_to_client(client, message):

    client.sendall(message.encode())


def send_message_to_all(message):

    for user in active_clients:

        send_message_to_client(user[1], message)


# Function to handle client
def client_handler(client):

    # Server will listen for client message that will contain the username
    while 1:

        username = client.recv(2048).decode('utf-8')
        if username != "":
            active_clients.append((username, client))
            break

        else:
            print("Client username is empty")

    threading.Thread(target=listen_for_messages, args=(client, username, )).start()

def main():

    #Creating the socket class object
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        #blinding server for a connection
        server.bind((HOST, PORT))
        print(f"Running server on {HOST} {PORT}")
    except:
        print(f"Unable to bind to host {HOST} and port {PORT}")

    #set server limit
    server.listen(LISTENER_LIMIT)

    #This while loop will keep listening to client server
    while 1:

       client, address = server.accept()
       print(f"Successfully connected to client {address[0]} {address[1]}")

       threading.Thread(target=client_handler, args= (client, )).start()


if __name__ == '__main__':
    main()