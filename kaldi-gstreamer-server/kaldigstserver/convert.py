import os
import struct
import binascii


def do_plotting(input_float):

    import matplotlib.pyplot as plt
    
    plt.plot(input_float)
    plt.show()


def F32LE_to_S16LE(input_bytes):
    input_float = []    
    index = 0

    if len(input_bytes) % 4 != 0:
        print('WARNING! Not divisible by 4')

    while index < len(input_bytes) - 3:
        
        number = struct.unpack('<f', input_bytes[index: index + 4])
        
        input_float.append(number[0])
        index += 4
    

    if index != len(input_bytes) - 4 and index != len(input_bytes):
        print("Alignment issue!!!! Got " + str(len(input_bytes) - index))
        
    
    output_int = []
    output_bytes = bytearray()
    for i in input_float:
        value = int(i * (2**15))
        output_int.append(value)
        
        value = max(min(value, 2 ** 15 - 10), -2** 15 + 10)

        thing = struct.pack('<h', value)
        #print(thing)
        output_bytes.extend(thing)

    return output_bytes


if __name__ == '__main__':
    with open('../peterson_float.raw', 'rb') as f:
        loaded_bytes = f.read()

    print(len(F32LE_to_S16LE(loaded_bytes)))


'''                

input_float = []
with open('input_894080.raw', 'rb') as f:
    
    tmp = f.read(4)
    while tmp:
        number = struct.unpack('<f', tmp)
        tmp = f.read(4)
        #print(number)
        input_float.append(number[0])

#print(input_float)

#plt.plot(input_float)
#plt.show()

output_int = []
output_bytes = bytearray()
for i in input_float:
    value = int(i * (2**15))
    output_int.append(value)
    
    thing = struct.pack('<h', value)
    #print(thing)
    output_bytes.extend(thing)

print(len(output_bytes))

#plt.plot(output_int)
#plt.show()
'''



