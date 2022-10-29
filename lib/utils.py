
#region Pickle interface
import pickle
import numpy as np

def load_pickle(path):
    f = open(path, "rb")
    data = pickle.load(f)
    f.close()
    return data

def write_pickle(path, data):
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()

def ENUtoWGS(RefPosLat, RefPosLong, ENUInputPnt_north, ENUInputPnt_east):

    f64_KappaLat = 0.0
    f64_KappaLon = 0.0
    f64_ENUEast = 0.0
    f64_ENUNorth = 0.0

    f64_M = 0.0
    f64_N = 0.0

    F64_TMC_DEG2RAD = 0.01745329251994
    F64_TMC_RAD2DEG = 57.2957795130824
    F64_TMC_GEOD_A = 6378137.0
    F64_TMC_GEOD_E2 = 0.00669437999014
    F64_TMC_EPS = 0.00000001
    F64_TMC_WGS84_SCALE_FACTOR = 1.0e-7

    f64_WGS84Latitude_rad = F64_TMC_DEG2RAD * RefPosLat
    f64_WGS84Longitude_rad = F64_TMC_DEG2RAD * RefPosLong

    f64_SinLatitue_rad = np.sin(f64_WGS84Latitude_rad)
    f64_CosLatitue_rad = np.cos(f64_WGS84Latitude_rad)
    f64_SquareSinLatitue_rad = f64_SinLatitue_rad * f64_SinLatitue_rad

    f64_Denominator = np.sqrt(1.0 - (F64_TMC_GEOD_E2 * f64_SquareSinLatitue_rad))
    f64_CubicDenominator = f64_Denominator * f64_Denominator * f64_Denominator
    f64_NxCosLatitue = 0.0

    f64_M = F64_TMC_GEOD_A * (1.0 - F64_TMC_GEOD_E2) / f64_CubicDenominator
    f64_N = F64_TMC_GEOD_A / f64_Denominator

    f64_KappaLat = 1.0 / f64_M * F64_TMC_RAD2DEG
    f64_NxCosLatitue = f64_N * f64_CosLatitue_rad

    if f64_NxCosLatitue == 0.0:
        f64_NxCosLatitue = 0.000000000000001

    f64_KappaLon = 1.0 / f64_NxCosLatitue * F64_TMC_RAD2DEG

    f64_Latitude_deg = RefPosLat + (f64_KappaLat * ENUInputPnt_north)
    f64_Longitude_deg = RefPosLong + (f64_KappaLon * ENUInputPnt_east)

    return [f64_Latitude_deg, f64_Longitude_deg]
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

