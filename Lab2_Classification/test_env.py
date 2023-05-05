from sys import path
import os
print(path)
print(os.getcwd())


# import Lab1_Regression.env_test
in_channel = 64
block_num=4
for in_channel, block_num in zip(
                [in_channel, in_channel * 2, in_channel * 4, in_channel * 8],
                 range( 1, block_num + 1),
            ):
    print(in_channel, block_num)
print()
