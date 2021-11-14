import sys
import time
import binascii
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
NUM_HASH = 2
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
        A list A of n bits, initially all 0’s: A[0..n-1]
    '''
    A = list(0 for i in range(M))

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
        A set S of objects: previous_user_set
    '''
    previous_user_set = set()
    
    '''
        Construction: for each object o in S,
        – Apply each hash function hj to o
        – If hj(o) = i, set A[i] = 1 (if it was 0)

        Application: check if new object o’ is in S
        – Hash o’ using each hash function
        – If for some hash function hj(o’) = i and A[i] = 0,
        stop and report o’ not in S
    '''
    bx = BlackBox()
    FPR_list = list()
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_filename, stream_size)
        
        num_false_exist = 0
        num_true_not_exist = 0
        
        for i in range(len(stream_users)):
            user_id = stream_users[i]
            hash_values = myhashs(user_id)

            predict_exist = False
            match_count = 0
            for values in hash_values:
                # if any value not exist, the user is PREDICTED as not exist
                if A[values] != 0:
                    match_count += 1
            if match_count == NUM_HASH:
                predict_exist = True
            
            if user_id not in previous_user_set:
                # num_true_not_exist (True Negative): the user is ACTUALLY as not exist
                num_true_not_exist += 1
                # num_not_exist (False Positive): user ACTUALLY not exist, but PREDICTED as exist
                if predict_exist:
                    num_false_exist += 1

        # FPR = FP / (FP + TN)
        # Positive: exist, Negative: not exist
        FPR = num_false_exist / (num_false_exist + num_true_not_exist)
        FPR_list.append(FPR)
        
        # Update A for current batch
        for i in range(len(stream_users)):
            user_id = stream_users[i]
            hash_values = myhashs(user_id)
            
            for values in hash_values:
                A[values] = 1

    with open(output_filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["Time", "FPR"])
        for i in range(len(FPR_list)):    
            csv_writer.writerow([i, FPR_list[i]])
    csv_file.close()

    print("Duration:", str(time.time() - start_time))