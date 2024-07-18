"""
Created on Tue Jul  9 17:46:03 2024

@author: 44775
"""
from lattice_dna_model import *

t = lattice_dna(100, 100, 100, [-25, -50, 0], [25, -50, 0], 4)

zerovec = np.array([0,0,0])

altstartvec = np.array([-10,-10,-10])

#testlist = [zerovec, zerovec+vector_list[0], zerovec+2*vector_list[0], zerovec+3*vector_list[0]]
#testlist = [zerovec, zerovec+1*vector_list[0], zerovec+1*vector_list[0], zerovec+2*vector_list[0]]
#testlist = [np.array([0, 0,  0]), np.array([0, 1, -1]), np.array([1, 1, 0]), np.array([2, 2, 0])]
#testlist = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([1, 1, 0]), np.array([2, 2, 0])]
#testlist = [np.array([1, 0, 1]), np.array([0, 0, 0]), np.array([1, 1, 0]), np.array([2, 2, 0])]
testlist = [np.array([1, -1, 0]), np.array([0, 0, 0]), np.array([1, 1, 0]), np.array([2, 2, 0])]
#t.dnastr_A = [zerovec, zerovec+vector_list[0], zerovec+2*vector_list[0], zerovec+vector_list[0]]

testlistB = [altstartvec, altstartvec+vector_list[0], altstartvec+vector_list[0], altstartvec+2*vector_list[0]]
#t.dnastr_B = [altstartvec, altstartvec+vector_list[0], altstartvec+vector_list[0], altstartvec+2*vector_list[0]]

for test_index in range(t.lengths):
    print('OLD PARAM')
    print('strintact:',t.check_strintact(testlist, test_index),t.check_strintact(testlistB, test_index))
    print('excvol:',t.check_excvol_gen2(testlist, testlistB, test_index))
    
    print()
    print('NEW PARAM')
    newtestlist, newtestlistB = t.propose_change(testlist, testlistB, test_index)
    #newtestlist = [zerovec, zerovec+vector_list[0], zerovec+2*vector_list[0], zerovec+3*vector_list[0]]
    print('strintact:',t.check_strintact(newtestlist, test_index), t.check_strintact(newtestlistB, test_index))
    print('excvol:',t.check_excvol_gen2(newtestlist, newtestlistB, test_index))
    
    
    
    if t.check_excvol_gen2(newtestlist,newtestlistB,test_index) == False or t.check_strintact(newtestlist,test_index) == False or t.check_strintact(newtestlistB,test_index) == False or t.check_inbox(newtestlist[test_index]) == False or t.check_inbox(newtestlistB[test_index]) == False:
        print('FAILED ...')
        
    if not t.check_excvol_gen2(newtestlist,newtestlistB,test_index):
        print('FAILED excvol')
        
    if not t.check_strintact(newtestlist,test_index):
        print('FAILED strintact A')
        
    if not t.check_strintact(newtestlistB,test_index):
        print('FAILED strintact B')
        
    if not t.check_inbox(newtestlist[test_index]) and not t.check_inbox([newtestlistB,test_index]):
        print('FAILED inbox')
        
    if t.check_excvol_gen2(newtestlist,newtestlistB,test_index) and t.check_strintact(newtestlist,test_index) and t.check_strintact(newtestlistB,test_index) and t.check_inbox(newtestlist[test_index]) and t.check_inbox(newtestlistB[test_index]):
        print('PASSED')
        newtestlist, newtestlistB = t.propose_change(newtestlist, newtestlistB, test_index)


#print(vector_list[9])
nsteps = 100000

#for i, item in enumerate(range(nsteps)):
#    t.montecarlostep_gen2()

#t.proj_2d(fullbox = True )
#t.proj_2d(fullbox = False)

#t.proj_3d()
