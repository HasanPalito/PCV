import socket

address = ('127.0.0.1', 5066)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(address)
def move(roll,pitch,yaw,mouth=0,eye=0):

    msg = '%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'% (roll,pitch,yaw,0,mouth,eye,0,0)
    #(roll, pitch, yaw, min_ear, mar, mdst)
    s.send(bytes(msg, "utf-8"))
