# Check number of combinations, just to be sure.

import itertools

def list_of_combs(arr):
    """returns a list of all subsets of a list"""
    
    combs = []
    for i in range(1, len(arr)+1):
        listing = [list(x) for x in itertools.combinations(arr, i)]
        combs.extend(listing)
    return combs

# Not used, does not produce a good list..
#
#for l in range(1, len(features)+1):
#    for subset in itertools.combinations(features, l):
#        print(subset)
#        
#comb = list()
#
#for i in range(0,len(features)):
#    for a in itertools.combinations(features,i+1):
#        comb.append(a)

combinations = list_of_combs(features)
for i in range(len(combinations)):
    print(combinations[i])

print('\nNumber of combinations:',len(combinations))


eng_1_3 = ['ae1',
             'ae3',
             'me1',
             'me3']
eng_2_4 = ['ae2',
             'ae4',
             'me2',
             'me4']

test_13 = list()
test_24 = list()

# For the first range where only one features available, combination range 0-4

for j in range(4):
    temp_list13 = list()
    temp_list24 = list()
    for i in range(len(eng_1_3)):
        temp_list13.append(str(eng_1_3[i] + '_' + str(combinations[j][0])))
        temp_list24.append(str(eng_2_4[i] + '_' + str(combinations[j][0])))
    test_13.append(temp_list13)
    test_24.append(temp_list24)

    
# Next range, two features. Combination Range 4-10 
    
for j in range(4,10):
    temp_list13 = list()
    temp_list24 = list()
    for i in range(len(eng_1_3)):
        temp_list13.append([str(eng_1_3[i] + '_' + str(combinations[j][0] )),
                           str(eng_1_3[i] + '_' + str(combinations[j][1]  ))])
        temp_list24.append([str(eng_2_4[i] + '_' + str(combinations[j][0] )),
                           str(eng_2_4[i] + '_' + str(combinations[j][1]  ))])
    test_13.append(temp_list13)
    test_24.append(temp_list24)


for j in range(10,14):
    temp_list13 = list()
    temp_list24 = list()
    for i in range(len(eng_1_3)):
        temp_list13.append([str(eng_1_3[i] + '_' + str(combinations[j][0] )),
                           str(eng_1_3[i] + '_' + str(combinations[j][1]  )),
                           str(eng_1_3[i] + '_' + str(combinations[j][2]))])
        temp_list24.append([str(eng_2_4[i] + '_' + str(combinations[j][0] )),
                           str(eng_2_4[i] + '_' + str(combinations[j][1]  )),
                           str(eng_2_4[i] + '_' + str(combinations[j][2]  ))])
    test_13.append(temp_list13)
    test_24.append(temp_list24)



# And the last is with all variables

j = 14
for i in range(len(eng_1_3)):
    temp_list13 = list()
    temp_list24 = list()
    temp_list13.append([str(eng_1_3[i] + '_' + str(combinations[j][0] )),
                        str(eng_1_3[i] + '_' + str(combinations[j][1]  )),
                        str(eng_1_3[i] + '_' + str(combinations[j][2])) ,
                        str(eng_1_3[i] + '_' + str(combinations[j][3]))])
    temp_list24.append([str(eng_2_4[i] + '_' + str(combinations[j][0] )),
                        str(eng_2_4[i] + '_' + str(combinations[j][1]  )),
                        str(eng_2_4[i] + '_' + str(combinations[j][2]  )),
                        str(eng_2_4[i] + '_' + str(combinations[j][3] ))])
    test_13.append(temp_list13)
    test_24.append(temp_list24)
