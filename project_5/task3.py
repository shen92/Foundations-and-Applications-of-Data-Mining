import sys
import time
import random
import binascii
import csv

from blackbox import BlackBox

SEED = 553

if __name__ == '__main__':
    start_time = time.time()

    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]

    random.seed(SEED)

    bx = BlackBox()
    
    '''
        1. Store all the first s elements of the stream to S (n <= s)
        2. Suppose we have seen n-1 elements, and now the nth element arrives (n > s)
            - With probability s/n, keep the nth element, else discard it
            - If we picked the nth element, then it replaces one of the s elements in the sample S, picked uniformly at random
    '''
    s = 100
    n = 0
    reservoir = list()
    with open(output_filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["seqnum", "0_id", "20_id", "40_id", "80_id"])
        for i in range(num_of_asks):
            stream_users = bx.ask(input_filename, stream_size)
            
            # n <= s
            if i == 0:
                for user in stream_users:
                    n += 1
                    reservoir.append(user)
            # n > s
            else:
                for user in stream_users:
                    n += 1
                    p_keep_user =  random.random()
                    if p_keep_user < s / n:
                        target_position = random.randint(0, 99)
                        reservoir[target_position] = user
                    
            csv_writer.writerow([100 * (i + 1), reservoir[0], reservoir[20], reservoir[40], reservoir[60], reservoir[80]])

    csv_file.close()


    print("Duration:", str(time.time() - start_time))