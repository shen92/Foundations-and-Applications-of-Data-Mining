import sys
import time
import binascii
import statistics
import csv

from blackbox import BlackBox

PARAM_A = [ 30011, 30013, 30029, 30047, 30059, 30071, 30089, 30091, 30097, 30103,
            30109, 30113, 30119, 30133, 30137, 30139, 30161, 30169, 30181, 30187,
            30197, 30203, 30211, 30223, 30241, 30253, 30259, 30269, 30271, 30293,
            30307, 30313, 30319, 30323, 30341, 30347, 30367, 30389, 30391, 30403,
            30427, 30431, 30449, 30467, 30469, 30491, 30493, 30497, 30509, 30517,
            30529, 30539, 30553, 30557, 30559, 30577, 30593, 30631, 30637, 30643,
            30649, 30661, 30671, 30677, 30689, 30697, 30703, 30707, 30713, 30727,
            30757, 30763, 30773, 30781, 30803, 30809, 30817, 30829, 30839, 30841,
            30851, 30853, 30859, 30869, 30871, 30881, 30893, 30911, 30931, 30937,
            30941, 30949, 30971, 30977, 30983, 31013, 31019, 31033, 31039, 31051,
            31063, 31069, 31079, 31081, 31091, 31121, 31123, 31139, 31147, 31151,
            31153, 31159, 31177, 31181, 31183, 31189, 31193, 31219, 31223, 31231,
            31237, 31247, 31249, 31253, 31259, 31267, 31271, 31277, 31307, 31319 ]

PARAM_B = [ 71263, 71287, 71293, 71317, 71327, 71329, 71333, 71339, 71341, 71347,
            71353, 71359, 71363, 71387, 71389, 71399, 71411, 71413, 71419, 71429,
            71437, 71443, 71453, 71471, 71473, 71479, 71483, 71503, 71527, 71537,
            71549, 71551, 71563, 71569, 71593, 71597, 71633, 71647, 71663, 71671,
            71693, 71699, 71707, 71711, 71713, 71719, 71741, 71761, 71777, 71789,
            71807, 71809, 71821, 71837, 71843, 71849, 71861, 71867, 71879, 71881 ]

P = 92143
M = 69997
NUM_HASH = 54
NUM_PARTITION = 9
PARTITION_SIZE = 6
hash_function_list = list()

'''
    The input is a string s: user_id
    The output is a list of hash values
'''
def myhashs(s):
    result = []
    user_id_int = int(binascii.hexlify(s.encode("utf8")), 16)
    for i in range(NUM_HASH):
        result.append(hash_function_list[i](user_id_int, i))
    return result

if __name__ == '__main__':
    start_time = time.time()

    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]
    
    '''
        A list of hash functions: hash_function_list
    '''
    for i in range(NUM_HASH):
        # f(x) = ((ax + b) % p) % m
        # a = PARAM_A[3 * i % len(PARAM_A)]
        # b = PARAM_B[-(5 * i % len(PARAM_B))]
        # p = P
        # m = M
        hash_function = lambda x, i: ((PARAM_A[i % len(PARAM_B)] * x + PARAM_B[(i + 1 % len(PARAM_A))]) % P) % M
        hash_function_list.append(hash_function)

    '''
        1. Hash every element a to a sufficiently long bit-string (e.g., h(a) = 1100 – 4 bits)
        2. Maintain R = length of longest trailing zeros among all bit-strings (e.g., R = 2)
        3. Estimate count = 2 ^ R, e.g., 2 ^ 2= 4
    '''
    bx = BlackBox()
    distinct_elements_list = list()
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_filename, stream_size)

        # for hash function at i, the max num of tailing zeros for each batch bin strings 
        max_num_trailing_zero_list = list(0 for i in range(NUM_HASH))
        for user_id in stream_users:
            hash_values = myhashs(user_id)
            
            # for a user_id, get its num of tailing zeros in each hash group
            for i in range(len(hash_values)):
                hash_value = hash_values[i]
                value_bin = bin(hash_value)
                num_trailing_zero = len(value_bin) - len(value_bin.rstrip('0'))

                # update num of tailing zeros in each hash group
                if num_trailing_zero > max_num_trailing_zero_list[i]:
                    max_num_trailing_zero_list[i] = num_trailing_zero
                    
        # for hash function at i, power of 2 of the max num of tailing zeros
        estimate_count_list = list()
        for max_num_trailing_zero in max_num_trailing_zero_list:
            estimate_count_list.append(2 ** max_num_trailing_zero)
        
        '''
            Combine solutions:
            1. partition distinct_elements_list into NUM_PARTITION groups
            2. for each partition, calculate mean
            3. for reduced result, use median
        '''
        partitioned_estimate_count_list = list([] for i in range(NUM_PARTITION))
        for i in range(NUM_HASH):
            partition_index = i % NUM_PARTITION
            partitioned_estimate_count_list[partition_index].append(estimate_count_list[i])
        partitioned_estimate_mean_list = list()
        for partition in partitioned_estimate_count_list:
            partitioned_estimate_mean_list.append(sum(partition) / PARTITION_SIZE)
        distinct_elements_list.append(round(statistics.median(partitioned_estimate_mean_list)))

    with open(output_filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["Time", "Ground Truth", "Estimation"])
        for i in range(len(distinct_elements_list)):    
            csv_writer.writerow([i, stream_size, distinct_elements_list[i]])
    csv_file.close()
    
    print(sum(distinct_elements_list) / (stream_size * num_of_asks))
    print("Duration:", str(time.time() - start_time))