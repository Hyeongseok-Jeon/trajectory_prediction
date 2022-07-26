
#region Pickle interface
import pickle

def load_pickle(path):
    f = open(path, "rb")
    data = pickle.load(f)
    f.close()
    return data

def write_pickle(path, data):
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()
#endregion

#region C Types Structures
import ctypes
import numpy as np

# Lane Link c structure
COORD_MAX_SIZE = 550
CONN_MAX_SIZE = 10

class lane_link_t(ctypes.Structure):
    _fields_ = [ 
                # Connection information
                ("next_ids_count", ctypes.c_uint32),
                ("next_ids", ctypes.ARRAY(ctypes.c_uint64, CONN_MAX_SIZE)),
                
                ("prev_ids_count", ctypes.c_uint32),
                ("prev_ids", ctypes.ARRAY(ctypes.c_uint64, CONN_MAX_SIZE)),
                
                # Reference point
                ("ref_lat", ctypes.c_double),
                ("ref_lng", ctypes.c_double),
                
                # Points
                ("points_count", ctypes.c_uint32),
                ("points_x", ctypes.ARRAY(ctypes.c_double, COORD_MAX_SIZE)),
                ("points_y", ctypes.ARRAY(ctypes.c_double, COORD_MAX_SIZE)),

                # ID
                ("id", ctypes.c_uint64)
            ];
# Message
MESSAGE_OK = 1
MESSAGE_MAP = 2
MESSAGE_TRAJ = 3
MESSAGE_OBJ = 4
MESSAGE_EGO = 5
class message_t(ctypes.Structure):
    _fields_ = [
                ("id", ctypes.c_uint32),
                ("size", ctypes.c_uint32)
            ];

#endregion

#region Pipes Structure
import sys
import win32pipe, win32file, pywintypes

class Pipes:
    def __init__(self, in_name, out_name, buffer_size, connect_first=True):
        print("in_name: %s, out_name: %s"%(in_name, out_name))
        # Read Pipe
        self.inPipe = win32pipe.CreateNamedPipe(
                        r'{}'.format(in_name),
                        win32pipe.PIPE_ACCESS_DUPLEX,
                        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                        1, buffer_size, buffer_size,
                        win32pipe.NMPWAIT_USE_DEFAULT_WAIT,
                        None)
        
        # Wait for connection
        if not connect_first:
            print("Waiting for connection")
            win32pipe.ConnectNamedPipe(self.inPipe, None)
            # Receive Start Message
            status, msg = self.read_pipe(6)
            if status==0:
                print("Receive Message:", msg)
            # Opening outgoing pipe
            self.outPipe = win32file.CreateFile(
                            r'{}'.format(out_name),
                            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                            0,
                            None,
                            win32file.OPEN_EXISTING,
                            0,
                            None
                        )

            res = win32pipe.SetNamedPipeHandleState(self.outPipe, win32pipe.PIPE_READMODE_BYTE, None, None)
                
            print("Open Connection!")

            # Send Start Message
            self.write_pipe("Start\0".encode("ascii"))
            print("Write Start Message")

        if connect_first:
            
             # Opening outgoing pipe
            self.outPipe = win32file.CreateFile(
                            r'{}'.format(out_name),
                            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                            0,
                            None,
                            win32file.OPEN_EXISTING,
                            0,
                            None
                        )

            res = win32pipe.SetNamedPipeHandleState(self.outPipe, win32pipe.PIPE_READMODE_BYTE, None, None)
                
            print("Open Connection!")

            # Send Start Message
            self.write_pipe("Start\0".encode("ascii"))
            print("Write Start Message")

            print("Waiting for connection")
            win32pipe.ConnectNamedPipe(self.inPipe, None)

            # Receive Start Message
            status, msg = self.read_pipe(6)
            if status==0:
                print("Receive Message:", msg)

    # Read from inPipe
    def read_pipe(self, buffer_size):
        return win32file.ReadFile(self.inPipe, buffer_size)

    # Write to outPipe
    def write_pipe(self, message):
        win32file.WriteFile(self.outPipe, message)

#endregion


import string
import random

from datetime import datetime
def id_generator():
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S.%f")

